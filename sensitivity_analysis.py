from emukit.core import ContinuousParameter, DiscreteParameter, ParameterSpace
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.sensitivity.monte_carlo import MonteCarloSensitivity
import numpy as np

# Sensitivity analysis implemented using the following tutorial https://emukit.github.io/sensitivity-analysis/

def pass_values(file_name):
    file = open(file_name, 'r')

    line = file.readline().strip()
    num_var = len(line.split()) - 2

    X, Y = np.empty([0, num_var], float), np.empty([0,], float)

    while True:
        line = file.readline().strip()

        if not line:
            break

        variables = line.split()
        Y = np.append(Y, [float(variables[1])])
        converted = np.array(variables[2:]).astype(float)
        X = np.append(X, [converted], axis=0)

    file.close()
    return X, Y

X, Y = pass_values("bo_evaluations_3.txt")
# print("X", X)
# print("Y", Y)
# Normalize the data
Y = Y*(-1)
Y = Y - Y.mean()
Y = Y/ Y.std()
# Format Y to be the corret shape
Y = Y[:,None]


# Treat the marginal tax value of each bracket as a seperate variable
num_tax_brackets = 3
parameters = []

for i in range(num_tax_brackets):
    parameters.append(ContinuousParameter('bracket_rates_' + str(i), 0, 1))

parameters.append(DiscreteParameter('bracket_cutoffs', [0,1]))

space = ParameterSpace(parameters)


# For testing stuff with a predefined function
# X = np.random.randn(18, 8)
# Y = np.random.randn(18)[:, None]*9
# print("X", X)
# print("Y", Y)

# Do sensitivity analysis using GP surrogate model

# space = ParameterSpace([ContinuousParameter('x1', -np.pi, np.pi),
#                         ContinuousParameter('x2', -np.pi, np.pi),
#                         ContinuousParameter('x3', -np.pi, np.pi)])

# from emukit.core.initial_designs.random_design import RandomDesign
# from emukit.test_functions.sensitivity import Ishigami
# ishigami = Ishigami(a=5, b=0.1)
# target_simulator = ishigami.fidelity1

# desing = RandomDesign(space)
# X = desing.get_samples(500)
# Y  = target_simulator(X)[:,None]

model_gpy = GPRegression(X,Y)
model_emukit = GPyModelWrapper(model_gpy)
model_emukit.optimize()

# from emukit.core.initial_designs.random_design import RandomDesign
# desing = RandomDesign(space)
# X_i = desing.get_samples(10000)
# Y_i = model_emukit.predict(X_i)
# print(X_i)
# print(np.max(Y_i[0]))
# print(np.min(Y_i[0]))
# print(np.mean(Y_i[1]))

senstivity = MonteCarloSensitivity(model = model_emukit, input_domain = space)
main_effects, total_effects, m = senstivity.compute_effects(num_monte_carlo_points = 10000)
print(main_effects)
print(total_effects)
print(m)