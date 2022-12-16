import time
from ray.tune.logger import NoopLogger, pretty_print
from rllib_code.env_wrapper import RLlibEnvWrapper
from rllib_code.training_script import maybe_store_dense_log, process_args, set_up_dirs_and_maybe_restore
import ray
import logging
import os
import sys
from ray.rllib.agents.ppo import PPOTrainer
import yaml



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
        env=RLlibEnvWrapper, config=trainer_config, logger_creator=logger_creator
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
    
    # Creates a trainer object using the new run configuration we feed it. Essentially, what we can do is use the GP to change the parameters in the config and then run this to evaluate it.
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

        # We probably dont need to log this stuff
        # logger.info(
        #     "Iter %d: steps this-iter %d total %d -> %d/%d episodes done",
        #     curr_iter,
        #     result["timesteps_this_iter"],
        #     global_step,
        #     num_parallel_episodes_done,
        #     run_config["general"]["episodes"],
        # )
        # if curr_iter == 1 or result["episodes_this_iter"] > 0:
        #     logger.info(pretty_print(result))

        if result["policy_reward_mean"]:
            print("agent reward mean:", result["policy_reward_mean"]["a"])
            print("government reward mean:", result["policy_reward_mean"]["p"])
            print("episode reward mean:", result["episode_reward_mean"])
        
        
        # === Dense logging ===
        # maybe_store_dense_log(trainer, result, dense_log_frequency, dense_log_dir)

    return result["policy_reward_mean"]["p"]

        

    


if __name__ == "__main__":

    ray.init(log_to_driver=False)
    logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)


    run_dir = "./rllib_code/phase1"
    config = load_config(run_dir)
    economic_sim_func_eval(run_dir, config)
    ray.shutdown()  # shutdown Ray after use
