import jax
import jax.numpy as jnp
import haiku as hk
import gym
import numpy as np

import matplotlib.pyplot as plt

from rl.replay_buffer import ReplayBuffer
from rl.wrapper import GymWrapper
from rl.agent import DQN_Agent
import matplotlib.pyplot as plt

def eval_agent(agent, environment, params, actor_state, eval_episodes, rng):
        
    episode_returns = []
    for eval_episode in range(eval_episodes):
        returns = 0
        # Prepare agent, environment and accumulator for a new episode.
        timestep = environment.reset()
        
        while not timestep.last():
            
            # Get Action
            actor_output, actor_state = agent.actor_step(
                params, timestep, actor_state, rng, eval=True
            )

            # Agent-environment interaction.
            action = int(actor_output.actions)
            timestep = environment.step(action)
            # env.render()
            returns += timestep.reward
        
        episode_returns.append(returns)

    return np.mean(episode_returns)

def run_loop(
    agent: DQN_Agent, environment, seed, replay_buffer, batch_size, train_episodes, eval_period, eval_episodes
):

    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    params = agent.initial_params(next(rng))
    learner_state = agent.initial_learner_state(params)
    actor_state = agent.initial_actor_state()

    print(f"Training agent for {train_episodes} episodes")
    
    evaluation_scores = []
    for train_episode in range(train_episodes):
        print("Episode:",train_episode)
        
        # Prepare agent, environment and accumulator for a new episode.
        timestep = environment.reset()
        replay_buffer.push(timestep, None)

        while not timestep.last():
            
            # Get Action
            actor_output, actor_state = agent.actor_step(
                params, timestep, actor_state, next(rng)
            )

            # Agent-environment interaction.
            action = int(actor_output.actions)
            timestep = environment.step(action)

            # Accumulate experience.
            replay_buffer.push(timestep, action)

            # Learning.
            if replay_buffer.is_ready(batch_size):
                params, learner_state = agent.learner_step(
                    params, replay_buffer.sample(batch_size), learner_state
                )

        if (train_episode+1)%eval_period==0:
            average_eval_reward = eval_agent(agent, environment, params, actor_state, eval_episodes, next(rng))
            evaluation_scores.append(average_eval_reward)
    
    average_eval_reward = eval_agent(agent,environment,params, actor_state, eval_episodes, next(rng))
    evaluation_scores.append(average_eval_reward)
    return evaluation_scores
    
   

if __name__ == "__main__":
    
    env = GymWrapper(gym.make("CartPole-v1"))
    
    replay_buffer = ReplayBuffer(capacity=50000)

    dqn_agent = DQN_Agent(
    observation_spec=env.observation_spec(),
    action_spec=env.action_spec(),
    num_hidden_units=128,
    epsilon=0.1,
    learning_rate=1e-3,
    discount_gamma=0.99,
    target_update_period=50)

    run_data = run_loop(
        agent=dqn_agent,
        environment=env,
        seed=42,
        replay_buffer=replay_buffer,
        batch_size=64,
        train_episodes=500,
        eval_period = 100,
        eval_episodes = 100
    )

    plt.plot(np.arange(len(run_data)),np.array(run_data))
    plt.show()

