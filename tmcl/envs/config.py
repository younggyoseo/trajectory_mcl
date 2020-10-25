from tmcl.envs.normalized_env import normalize
from tmcl.envs import *


def get_environment_config(config):
    if config["dataset"] == "halfcheetah":
        train_mass_scale_set = [0.25, 0.5, 1.5, 2.5]
        train_damping_scale_set = [1.0]
        config["num_test"] = 2
        config["test_range"] = [
            [[0.1, 0.15], [1.0]],
            [[2.75, 3.0], [1.0]],
        ]
        env = HalfCheetahEnv(
            mass_scale_set=train_mass_scale_set,
            damping_scale_set=train_damping_scale_set,
        )
        env.seed(config["seed"])
        env = normalize(env)
        config["n_itr"] = 10
        config["total_test"] = 10
        config["test_num_rollouts"] = 10
        config["max_path_length"] = 1000
        config["simulation_param_dim"] = 2

    elif config["dataset"] == "cripple_ant":
        cripple_set = [0, 1, 2]
        extreme_set = [0]
        config["num_test"] = 1
        config["test_range"] = [
            [[3], [0]],
        ]
        env = CrippleAntEnv(cripple_set=cripple_set, extreme_set=extreme_set)
        env.seed(config["seed"])
        env = normalize(env)
        config["n_itr"] = 10
        config["total_test"] = 10
        config["test_num_rollouts"] = 10
        config["max_path_length"] = 1000
        config["simulation_param_dim"] = 1

    elif config["dataset"] == "slim_humanoid":
        train_mass_scale_set = [0.8, 0.9, 1.0, 1.15, 1.25]
        train_damping_scale_set = [1.0]
        config["num_test"] = 2
        config["test_range"] = [
            [[0.6, 0.7], [1.0]],
            [[1.5, 1.6], [1.0]],
        ]
        env = SlimHumanoidEnv(
            mass_scale_set=train_mass_scale_set,
            damping_scale_set=train_damping_scale_set,
        )
        env.seed(config["seed"])
        env = normalize(env)
        config["n_itr"] = 10
        config["total_test"] = 10
        config["test_num_rollouts"] = 10
        config["max_path_length"] = 1000
        config["simulation_param_dim"] = 2

    elif config["dataset"] == "hopper":
        train_mass_scale_set = [0.5, 0.75, 1.0, 1.25, 1.5]
        train_damping_scale_set = [1.0]
        config["num_test"] = 2
        config["test_range"] = [
            [[0.25, 0.375], [1.0]],
            [[1.75, 2.0], [1.0]],
        ]
        env = HopperEnv(
            mass_scale_set=train_mass_scale_set,
            damping_scale_set=train_damping_scale_set,
        )
        env.seed(config["seed"])
        env = normalize(env)
        config["total_test"] = 10
        config["test_num_rollouts"] = 10
        config["max_path_length"] = 500
        config["simulation_param_dim"] = 2

    else:
        raise ValueError(config["dataset"])

    return env, config
