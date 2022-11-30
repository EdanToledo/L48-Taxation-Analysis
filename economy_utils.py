import numpy as np
import matplotlib.pyplot as plt

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
    