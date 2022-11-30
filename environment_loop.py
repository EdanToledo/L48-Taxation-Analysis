import jax
import haiku as hk
import numpy as np
import matplotlib.pyplot as plt
from ai_economist import foundation


from config import basic_env_config
from rl.replay_buffer import ReplayBuffer, stack_trees
from rl.wrapper import GymWrapper
from rl.agent import DQN_Agent
import matplotlib.pyplot as plt

def eval_agent(agent : DQN_Agent, environment, params, eval_episodes, rng_key):
    agent_indices = [f"{i}" for i in range(agent._n_agents)]

    episode_returns = []
    for eval_episode in range(eval_episodes):
        returns = {a_idx : 0 for a_idx in agent_indices}
        # Prepare agent, environment and accumulator for a new episode.
        timestep = environment.reset()
        
        while not timestep.last():
            
            # Get Action
            actions = {a_idx: agent.select_action(params, rng_key, timestep.observation[a_idx]["flat"], timestep.observation[a_idx]['action_mask']) for a_idx in agent_indices}

            # Agent-environment interaction.
            timestep = environment.step(actions)
            # env.render()
            returns = {a_idx : timestep.reward[a_idx] + returns[a_idx] for a_idx in agent_indices}
        
        episode_returns.append(returns)
    
    episode_returns = stack_trees(episode_returns)
    rewards = jax.tree_map(lambda x : np.mean(x), episode_returns)

    return np.mean([reward for reward in rewards.values()])

def run_loop(
    agent: DQN_Agent, environment, seed, replay_buffer, batch_size, train_episodes, eval_period, eval_episodes
):

    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    params = agent.initial_params(next(rng))
    learner_state = agent.initial_learner_state(params)
    agent_indices = [f"{i}" for i in range(agent._n_agents)]

    print(f"Training agent for {train_episodes} episodes")
    
    evaluation_scores = []
    for train_episode in range(train_episodes):
        print("Episode:",train_episode)
        
        # Prepare agent, environment and accumulator for a new episode.
        timestep = environment.reset()
        replay_buffer.push(timestep, None)

        while not timestep.last():
            
            # Get Actions
            actions = {a_idx: agent.select_action(params, next(rng), timestep.observation[a_idx]["flat"], timestep.observation[a_idx]['action_mask']) for a_idx in agent_indices}

            # Agent-environment interaction.
            timestep = environment.step(actions)
            
            # Accumulate experience.
            replay_buffer.push(timestep, actions)

            # Learning.
            if replay_buffer.is_ready(batch_size):
                params, learner_state = agent.learner_step(
                    params, replay_buffer.sample(batch_size), learner_state
                )

        if (train_episode+1)%eval_period==0:
            average_eval_reward = eval_agent(agent, environment, params, eval_episodes, next(rng))
            print("Eval Reward:",average_eval_reward)
            evaluation_scores.append(average_eval_reward)
    
    average_eval_reward = eval_agent(agent,environment,params, eval_episodes, next(rng))
    evaluation_scores.append(average_eval_reward)

    return evaluation_scores
    
   

if __name__ == "__main__":
    
    env = foundation.make_env_instance(**basic_env_config)
    
   

    env = GymWrapper(env)
    
    replay_buffer = ReplayBuffer(capacity=100000)

    dqn_agent = DQN_Agent(
    observation_spec=env.observation_spec(),
    action_spec=env.action_spec(),
    num_hidden_units=128,
    epsilon=0.1,
    learning_rate=1e-3,
    discount_gamma=0.99,
    target_update_period=100,
    n_agents = env._environment.n_agents)

    run_data = run_loop(
        agent=dqn_agent,
        environment=env,
        seed=42,
        replay_buffer=replay_buffer,
        batch_size=128,
        train_episodes=500,
        eval_period = 50,
        eval_episodes = 100,
    )
    
    print(run_data)
    # plt.plot(np.arange(len(run_data)),np.array(run_data))
    # plt.show()

