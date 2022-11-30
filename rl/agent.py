import jax
import jax.numpy as jnp
import rlax
import optax
import chex
import haiku as hk
import numpy as np 

from rl.network import build_dqn


@chex.dataclass
class ActorState:
    count: int


@chex.dataclass
class LearnerState:
    count: int
    opt_state: optax.OptState


@chex.dataclass
class Params:
    online: hk.Params
    target: hk.Params


@chex.dataclass
class ActorOutput:
    q_values: chex.Array
    actions: chex.Array


class DQN_Agent:
    def __init__(
        self,
        observation_spec,
        action_spec,
        num_hidden_units,
        epsilon,
        learning_rate,
        discount_gamma,
        target_update_period,
        n_agents
    ):
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._epsilon = epsilon
        self._discount_gamma = discount_gamma
        self._target_update_period = target_update_period
        self._n_agents = n_agents

        # Neural net and optimiser.
        # TODO (edan) : make generic 
        self._network = build_dqn(num_hidden_units, action_spec["0"].num_values)
        
        self._optimizer = optax.adam(learning_rate)
        # Jitting for speed.
        self.actor_step = jax.jit(self.actor_step)
        self.learner_step = jax.jit(self.learner_step)

    def initial_params(self, key):
        # TODO (edan): make generic 
        sample_input = jnp.zeros(self._observation_spec["0"]["flat"].shape)
        params = self._network.init(key, sample_input)
        return Params(online=params, target=params)

    def initial_actor_state(self):
        actor_count = jnp.zeros((), dtype=jnp.float32)
        return ActorState(count=actor_count)

    def initial_learner_state(self, params):
        learner_count = jnp.zeros((), dtype=jnp.float32)
        opt_state = self._optimizer.init(params.online)
        return LearnerState(count=learner_count, opt_state=opt_state)

    def select_action(self, params, key, obs, mask, eval = False):
        
        actor_output = self.actor_step(params, obs, key, eval)
        masked_q_values = jnp.where(mask,actor_output.q_values,jnp.finfo(actor_output.q_values.dtype).min)
        actor_output = actor_output.replace(q_values = masked_q_values) 
        return actor_output.actions


    def actor_step(self, params, observation, key, eval=False):
        q_values = self._network.apply(params.online, observation)
        train_action = rlax.epsilon_greedy(self._epsilon).sample(key, q_values)
        eval_action = rlax.greedy().sample(key, q_values)

        action = jax.lax.select(eval, eval_action, train_action)

        return ActorOutput(actions=action, q_values=q_values)

    def learner_step(self, params, data, learner_state):
        target_params = optax.periodic_update(
            params.online, params.target, learner_state.count, self._target_update_period
        )
        obs_tm1, a_tm1, r_t, discount_t, obs_t = data
        for i in range(self._n_agents):
            
            dloss_dtheta = jax.grad(self._loss)(params.online, target_params, obs_tm1[f"{i}"]["flat"], a_tm1[f"{i}"], r_t[f"{i}"], discount_t, obs_t[f"{i}"]["flat"])

            updates, opt_state = self._optimizer.update(
                dloss_dtheta, learner_state.opt_state
            )

            online_params = optax.apply_updates(params.online, updates)

            learner_state = LearnerState(count=learner_state.count + 1, opt_state=opt_state)

            params = Params(online=online_params, target=target_params)

        return params, learner_state

    def _loss(self, params, target_params, obs_tm1, a_tm1, r_t, discount_t, obs_t):
        q_tm1 = self._network.apply(params, obs_tm1)
        q_t = self._network.apply(target_params, obs_t)
        td_error = jax.vmap(rlax.q_learning)(
            q_tm1, a_tm1, r_t, self._discount_gamma * discount_t, q_t
        )
        return jnp.mean(rlax.l2_loss(td_error))
