import jax
import haiku as hk
import numpy as np
import matplotlib.pyplot as plt

from ai_economist import foundation


from config import basic_env_config
from rl.replay_buffer import ReplayBuffer, stack_trees
from rl.wrapper import GymWrapper
from rl.agent import DQN_Agent
from economy_utils import do_plot, sample_random_actions
import plotting


def eval_agent(agent : DQN_Agent, environment, params, eval_episodes, rng_key):
    

    episode_returns = []
    for eval_episode in range(eval_episodes):
        returns = 0
        # Prepare agent, environment and accumulator for a new episode.
        timestep = environment.reset()
        
        while not timestep.last():
            
            # Get Action
            actions = {a_idx: agent.select_action(params, rng_key, timestep.observation[a_idx]["flat"], timestep.observation[a_idx]['action_mask']) for a_idx in agent.agent_indices}

            # Agent-environment interaction.
            timestep = environment.step(actions)
            # env.render()
            returns += np.sum(list(timestep.reward.values()))
           
        
        episode_returns.append(returns)

    return np.mean(episode_returns)

def run_loop(
    agent: DQN_Agent, environment, seed, replay_buffer, batch_size, train_episodes, eval_period, eval_episodes
):

    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    params = agent.initial_params(next(rng))
    learner_state = agent.initial_learner_state(params)
    

    print(f"Training agent for {train_episodes} episodes")
    
    evaluation_scores = []
    for train_episode in range(train_episodes):
        print("Episode:",train_episode)
        
        # Prepare agent, environment and accumulator for a new episode.
        timestep = environment.reset()
        replay_buffer.push(timestep, None)

        while not timestep.last():
            
            # Get Actions
            actions = {a_idx: agent.select_action(params, next(rng), timestep.observation[a_idx]["flat"], timestep.observation[a_idx]['action_mask']) for a_idx in agent.agent_indices}

            # Agent-environment interaction.
            timestep = environment.step(actions)
            
            # Accumulate experience.
            replay_buffer.push(timestep, actions)

            # Learning.
            if replay_buffer.is_ready(batch_size):
                params, learner_state, loss = agent.learner_step(
                    params, replay_buffer.sample(batch_size), learner_state
                )
                print("Loss:",loss)

        if (train_episode+1)%eval_period==0:
            average_eval_reward = eval_agent(agent, environment, params, eval_episodes, next(rng))
            
            evaluation_scores.append(average_eval_reward)
    
    average_eval_reward = eval_agent(agent,environment,params, eval_episodes, next(rng))
    evaluation_scores.append(average_eval_reward)

    return params, evaluation_scores
    

def play_final_episode(env, agent : DQN_Agent, params, rng_key, random_actions = False):

    # Reset
    timestep = env.reset(force_dense_logging=True)

    # Interaction loop (w/ plotting)
    for t in range(env.episode_length):
        if random_actions:
            actions = sample_random_actions(env,timestep.observation)
        else:
            actions = {a_idx: agent.select_action(params, rng_key, timestep.observation[a_idx]["flat"], timestep.observation[a_idx]['action_mask']) for a_idx in agent.agent_indices}

        timestep = env.step(actions)


if __name__ == "__main__":
    
    env = foundation.make_env_instance(**basic_env_config)

    env = GymWrapper(env)
    
    replay_buffer = ReplayBuffer(capacity=100000)

    dqn_agent = DQN_Agent(
    observation_spec=env.observation_spec(),
    action_spec=env.action_spec(),
    num_hidden_units=256,
    epsilon=0.1,
    learning_rate=1e-3,
    discount_gamma=0.99,
    target_update_period=50,
    n_agents = env._environment.n_agents)

    params, evaluation_scores = run_loop(
        agent=dqn_agent,
        environment=env,
        seed=42,
        replay_buffer=replay_buffer,
        batch_size=128,
        train_episodes=10,
        eval_period = 100,
        eval_episodes = 10,
    )

    
    play_final_episode(env, dqn_agent, params, jax.random.PRNGKey(42),True)
    dense_log = env._environment.previous_episode_dense_log
    (fig0, fig1, fig2), incomes, endows, c_trades, all_builds = plotting.breakdown(dense_log)

    # print(evaluation_scores)
    # plt.plot(np.array(evaluation_scores))
    plt.show()

