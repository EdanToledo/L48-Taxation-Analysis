import numpy as np
import matplotlib.pyplot as plt
import plotting
from IPython import display

def income_earned(labor, skill):
    """Income is amount of work (labor) times skill."""
    return labor * skill

def utility(labor, skill):
    """Utility is convex increasing in income and linearly decreasing in amount of work (labor)."""

    def isoelastic_utility(z, eta=0.35):
        """Utility gained from income z: https://en.wikipedia.org/wiki/Isoelastic_utility"""
        return (z**(1-eta) - 1) / (1 - eta)
    
    income = income_earned(labor, skill)
    utility_from_income = isoelastic_utility(income)
    disutility_from_labor = labor
    
    # Total utility is utility from income minus disutility incurred from working
    return utility_from_income - disutility_from_labor



def plot_utility_curve(skill):
    """Plot the curve relating labor performed to total utility."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    labor_array = np.linspace(0, 1000, 501)
    utility_array = utility(labor_array, skill)
    ax.plot(labor_array, utility_array, label="Skill = {}".format(skill))
    ax.plot(labor_array[np.argmax(utility_array)], np.max(utility_array), 'k*', markersize=10)

def do_plot(env, ax, fig):
    """Plots world state during episode sampling."""
    plotting.plot_env_state(env, ax)
    ax.set_aspect('equal')
    display.display(fig)
    display.clear_output(wait=True)

def sample_random_action(agent, mask):
    """Sample random UNMASKED action(s) for agent."""
    # Return a list of actions: 1 for each action subspace
    if agent.multi_action_mode:
        split_masks = np.split(mask, agent.action_spaces.cumsum()[:-1])
        return [np.random.choice(np.arange(len(m_)), p=m_/m_.sum()) for m_ in split_masks]

    # Return a single action
    else:
        return np.random.choice(np.arange(agent.action_spaces), p=mask/mask.sum())

def sample_random_actions(env, obs):
    """Samples random UNMASKED actions for each agent in obs."""
        
    actions = {
        a_idx: sample_random_action(env.get_agent(a_idx), a_obs['action_mask'])
        for a_idx, a_obs in obs.items()
    }

    return actions