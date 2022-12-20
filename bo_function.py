import time
import ray
import logging
import os
import sys
import yaml

from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import NoopLogger, pretty_print
from rllib_code.env_wrapper import RLlibEnvWrapper
from rllib_code.training_script import (
    custom_log_creator,
    maybe_store_dense_log,
    process_args,
    set_up_dirs_and_maybe_restore,
)

from GPyOpt.methods import BayesianOptimization
from ai_economist import foundation
from ai_economist.foundation.base.base_env import BaseEnvironment, scenario_registry
import numpy as np

# State Dicts
state_dict = {}


def build_trainer(run_configuration):
    """Finalize the trainer config by combining the sub-configs."""
    trainer_config = run_configuration.get("trainer")

    # === Env ===
    env_config = {
        "env_config_dict": run_configuration.get("env"),
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
    }

    # === Seed ===
    if trainer_config["seed"] is None:
        try:
            start_seed = int(run_configuration["metadata"]["launch_time"])
        except KeyError:
            start_seed = int(time.time())
    else:
        start_seed = int(trainer_config["seed"])

    final_seed = int(start_seed % (2 ** 16)) * 1000
    logger.info("seed (final): %s", final_seed)

    # === Multiagent Policies ===
    dummy_env = RLlibEnvWrapper(env_config)

    # Policy tuples for agent/planner policy types
    agent_policy_tuple = (
        None,
        dummy_env.observation_space,
        dummy_env.action_space,
        run_configuration.get("agent_policy"),
    )
    planner_policy_tuple = (
        None,
        dummy_env.observation_space_pl,
        dummy_env.action_space_pl,
        run_configuration.get("planner_policy"),
    )

    policies = {"a": agent_policy_tuple, "p": planner_policy_tuple}

    def policy_mapping_fun(i):
        if str(i).isdigit() or i == "a":
            return "a"
        return "p"

    # Which policies to train
    if run_configuration["general"]["train_planner"]:
        policies_to_train = ["a", "p"]
    else:
        policies_to_train = ["a"]

    # === Finalize and create ===
    trainer_config.update(
        {
            "env_config": env_config,
            "seed": final_seed,
            "multiagent": {
                "policies": policies,
                "policies_to_train": policies_to_train,
                "policy_mapping_fn": policy_mapping_fun,
            },
            "metrics_smoothing_episodes": trainer_config.get("num_workers")
            * trainer_config.get("num_envs_per_worker"),
        }
    )

    def logger_creator(config):
        return NoopLogger({}, "/tmp")

    ppo_trainer = PPOTrainer(
        env=RLlibEnvWrapper,
        config=trainer_config,
        logger_creator=custom_log_creator("./experiment_results", "econ_exp"),
    )

    return ppo_trainer


def load_config(run_dir):
    config_path = os.path.join(run_dir, "config.yaml")
    assert os.path.isdir(run_dir)
    assert os.path.isfile(config_path)

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    return run_configuration


def economic_sim_func_eval(run_dir, run_config):

    # Creates a trainer object using the new run configuration we feed it.
    # Essentially, what we can do is use the GP to change the parameters in
    # the config and then run this to evaluate it.
    trainer = build_trainer(run_config)

    # Set up directories for logging and saving. Restore if this has already been
    # done (indicating that we're restarting a crashed run). Or, if appropriate,
    # load in starting model weights for the agent and/or planner.
    # This is used to load in the free market agent weights when specified.
    (
        dense_log_dir,
        ckpt_dir,
        restore_from_crashed_run,
        step_last_ckpt,
        num_parallel_episodes_done,
    ) = set_up_dirs_and_maybe_restore(run_dir, run_config, trainer)

    # ======================
    # === Start training ===
    # ======================
    dense_log_frequency = run_config["env"].get("dense_log_frequency", 0)
    global_step = int(step_last_ckpt)

    while num_parallel_episodes_done < run_config["general"]["episodes"]:

        # Training
        result = trainer.train()

        # === Counters++ ===
        num_parallel_episodes_done = result["episodes_total"]
        global_step = result["timesteps_total"]
        curr_iter = result["training_iteration"]

    return result["policy_reward_mean"]["p"]


def f(x):
    """
    Wrapper around `economic_sim_func_eval` used as the target function to optimise
    with BO.

    Args:
        x (np.array): The bracket intervals for the number of brackets specified by
                      the outer optimisation loop.
    """
    run_dir = run_dir = "./rllib_code/phase2"
    run_config = load_config(run_dir)

   
    # Change configuration to make tax brackets flexible
    run_config["env"]["components"][-1]["PeriodicBracketTax"]["tax_model"] = "fixed-bracket-rates"
    run_config["env"]["components"][-1]["PeriodicBracketTax"]["n_brackets"] = len(x[0][0:-1])
    run_config["env"]["components"][-1]["PeriodicBracketTax"]["fixed_bracket_rates"] = list(x[0][0:-1])
    if x[0][-1] == 0:
        run_config["env"]["components"][-1]["PeriodicBracketTax"]["bracket_spacing"] = "linear"
    else:
        run_config["env"]["components"][-1]["PeriodicBracketTax"]["bracket_spacing"] = "log"

    # We use 2000 from seeing free-market agents achieving this value
    run_config["env"]["components"][-1]["PeriodicBracketTax"]["top_bracket_cutoff"] = 2000

    return economic_sim_func_eval(run_dir, run_config)


def optimise_brackets(n_brackets):
    """
    Given a number of brackets generated by an outer BO loop,
    optimise the brackets that we already have.
    """
    global state_dict

    # TODO: Domain is currently arbitrary and needs to be experimentally determined (i.e. see what
    # values for brackets would be realistic).

    # Performs a single BO
    domain = [
        {
            "name": "bracket_rates",
            "type": "continuous",
            "dimensionality": int(n_brackets),
            "domain": (0, 1),
        },
        {
            "name": "bracket_cutoffs",
            "type": "discrete",
            "domain": (0, 1),
            "dimensionality": 1,
        },
    ]

    opt = BayesianOptimization(f=f, domain=domain)
    opt.run_optimization(max_iter=30, max_time=3600)

    # Compute the results of this
    ins, outs = opt.get_evaluations()[0], opt.get_evaluations()[1]
    state_dict[n_brackets] = [ins, outs]
    result = np.min(outs)
    return result


def bayesian_optimisation():
    """
    Objective function we are trying to optimise.
    """

    # Optimise over the number of brackets – max taken to be 100.
    domain = [
        {
            "name": "size",
            "type": "discrete",
            "dimensionality": 1,
            "domain": tuple(i for i in range(1, 101)),
        }
    ]
    opt = BayesianOptimization(f=optimise_brackets, domain=domain)
    opt.run_optimization(max_iter=11)

    # Compute the results of this
    return opt


if __name__ == "__main__":

    ray.init(log_to_driver=False, include_webui=False)
    logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)

    # Perform BO
    opt = bayesian_optimisation()

    print("Opt Eval:",opt.get_evaluations())
    opt.save_report("BO_Report.txt")
    # economic_sim_func_eval(run_dir, config)
    ray.shutdown()  # shutdown Ray after use
