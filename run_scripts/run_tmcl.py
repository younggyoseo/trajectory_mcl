from tmcl.dynamics.tmcl import MCLMultiHeadedCaDMDynamicsModel
from tmcl.trainers.mb_trainer import Trainer
from tmcl.policies.mpc_controller import MPCController
from tmcl.samplers.sampler import Sampler
from tmcl.logger import logger
from tmcl.envs.normalized_env import normalize
from tmcl.utils.utils import ClassEncoder
from tmcl.samplers.model_sample_processor import ModelSampleProcessor
from tmcl.envs.config import get_environment_config

from tensorboardX import SummaryWriter
import json
import os
import gym
import argparse


def run_experiment(config):
    env, config = get_environment_config(config)

    # Save final config after editing config with respect to each environment.
    EXP_NAME = config["save_name"]
    EXP_NAME += (
        "hidden_" + str(config["dim_hidden"]) + "_lr_" + str(config["learning_rate"])
    )
    EXP_NAME += "_horizon_" + str(config["horizon"]) + "_seed_" + str(config["seed"])

    exp_dir = os.getcwd() + "/data/" + EXP_NAME + "/" + config.get("exp_name", "")
    logger.configure(
        dir=exp_dir,
        format_strs=["stdout", "log", "csv"],
        snapshot_mode="last",
        only_test=config["only_test_flag"],
    )
    json.dump(
        config,
        open(exp_dir + "/params.json", "w"),
        indent=2,
        sort_keys=True,
        cls=ClassEncoder,
    )
    writer = SummaryWriter(exp_dir)

    dynamics_model = MCLMultiHeadedCaDMDynamicsModel(
        name="dyn_model",
        env=env,
        learning_rate=config["learning_rate"],
        hidden_sizes=config["hidden_sizes"],
        valid_split_ratio=config["valid_split_ratio"],
        rolling_average_persitency=config["rolling_average_persitency"],
        hidden_nonlinearity=config["hidden_nonlinearity"],
        traj_batch_size=config["traj_batch_size"],
        sample_batch_size=config["sample_batch_size"],
        segment_size=config["segment_size"],
        normalize_input=config["normalize_flag"],
        n_forwards=config["horizon"],
        n_candidates=config["n_candidates"],
        ensemble_size=config["ensemble_size"],
        head_size=config["head_size"],
        n_particles=config["n_particles"],
        use_cem=config["use_cem"],
        deterministic=config["deterministic"],
        weight_decays=config["weight_decays"],
        weight_decay_coeff=config["weight_decay_coeff"],
        ie_itrs=config["ie_itrs"],
        use_ie=config["use_ie"],
        use_simulation_param=config["use_simulation_param"],
        simulation_param_dim=config["simulation_param_dim"],
        sep_layer_size=config["sep_layer_size"],
        cp_hidden_sizes=config["context_hidden_sizes"],
        context_weight_decays=config["context_weight_decays"],
        context_out_dim=config["context_out_dim"],
        context_hidden_nonlinearity=config["context_hidden_nonlinearity"],
        history_length=config["history_length"],
        future_length=config["future_length"],
        state_diff=config["state_diff"],
        back_coeff=config["back_coeff"],
        use_global_head=config["use_global_head"],
        non_adaptive_planning=config["non_adaptive_planning"],
    )

    policy = MPCController(
        name="policy",
        env=env,
        dynamics_model=dynamics_model,
        discount=config["discount"],
        n_candidates=config["n_candidates"],
        horizon=config["horizon"],
        use_cem=config["use_cem"],
        num_rollouts=config["num_rollouts"],
        mcl_cadm=True,
    )

    sampler = Sampler(
        env=env,
        policy=policy,
        num_rollouts=config["num_rollouts"],
        max_path_length=config["max_path_length"],
        n_parallel=config["n_parallel"],
        random_flag=True,
        use_cem=config["use_cem"],
        horizon=config["horizon"],
        state_diff=config["state_diff"],
        history_length=config["history_length"],
        mcl_cadm=True,
    )

    sample_processor = ModelSampleProcessor(
        recurrent=True,  # MCL
        writer=writer,
        context=True,
        future_length=config["future_length"],
    )

    algo = Trainer(
        env=env,
        env_flag=config["dataset"],
        policy=policy,
        dynamics_model=dynamics_model,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config["n_itr"],
        initial_random_samples=config["initial_random_samples"],
        dynamics_model_max_epochs=config["dynamic_model_epochs"],
        num_test=config["num_test"],
        test_range=config["test_range"],
        total_test=config["total_test"],
        test_max_epochs=config["max_path_length"],
        no_test_flag=config["no_test_flag"],
        only_test_flag=config["only_test_flag"],
        use_cem=config["use_cem"],
        horizon=config["horizon"],
        writer=writer,
        mcl_cadm=True,
        history_length=config["history_length"],
        state_diff=config["state_diff"],
    )
    algo.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory-wise MCL")
    parser.add_argument("--save_name", default="TMCL/", help="experiments name")
    parser.add_argument("--seed", type=int, default=0, help="random_seed")
    parser.add_argument("--dataset", default="halfcheetah", help="environment flag")
    parser.add_argument(
        "--hidden_size", type=int, default=200, help="size of hidden feature"
    )
    parser.add_argument(
        "--traj_batch_size", type=int, default=250, help="batch size (trajectory)"
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=256, help="batch size (sample)"
    )
    parser.add_argument("--segment_size", type=int, default=10, help="segment size")
    parser.add_argument(
        "--n_epochs", type=int, default=50, help="training epochs per iteration"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning_rate")
    parser.add_argument("--horizon", type=int, default=30, help="horrizon for planning")
    parser.add_argument(
        "--normalize_flag", action="store_true", help="flag to normalize"
    )
    parser.add_argument("--total_test", type=int, default=20, help="# of test")
    parser.add_argument(
        "--n_candidate", type=int, default=200, help="candidate for planning"
    )
    parser.add_argument(
        "--no_test_flag", action="store_true", help="flag to disable test"
    )
    parser.add_argument(
        "--only_test_flag", action="store_true", help="flag to enable only test"
    )
    parser.add_argument(
        "--ensemble_size", type=int, default=5, help="size of ensembles"
    )
    parser.add_argument("--head_size", type=int, default=3, help="size of heads")
    parser.add_argument(
        "--n_particles",
        type=int,
        default=20,
        help="size of particles in trajectory sampling",
    )
    parser.add_argument("--policy_type", type=str, default="CEM", help="Policy Type")
    parser.add_argument(
        "--deterministic_flag",
        type=int,
        default=0,
        help="flag to use deterministic dynamics model",
    )
    parser.add_argument(
        "--use_ie_flag", type=int, default=1, help="flag to use ie loss with ie_itrs"
    )
    parser.add_argument(
        "--ie_itrs", type=int, default=3, help="epochs to train with IE loss"
    )
    parser.add_argument(
        "--sim_param_flag",
        type=int,
        default=0,
        help="flag to use simulation parameter as an input",
    )
    parser.add_argument(
        "--sep_layer_size",
        type=int,
        default=0,
        help="size of separated layers in multiheaded architecture",
    )
    parser.add_argument(
        "--tag", type=str, default="", help="additional tag for save directory.."
    )
    parser.add_argument(
        "--history_length",
        type=int,
        default=10,
        help="history length for adaptive planning",
    )
    parser.add_argument(
        "--state_diff",
        type=int,
        default=1,
        help="flag to use state difference for history",
    )
    parser.add_argument(
        "--back_coeff", type=float, default=0.5, help="coefficient for backward loss"
    )
    parser.add_argument(
        "--future_length", type=int, default=5, help="length of future timesteps"
    )
    parser.add_argument(
        "--context_out_dim", type=int, default=10, help="dimension of context vector"
    )
    parser.add_argument(
        "--use_global_head_flag",
        type=int,
        default=1,
        help="use global head for context encoder",
    )
    parser.add_argument(
        "--non_adaptive_planning_flag",
        type=int,
        default=0,
        help="non-adaptive planning. just average",
    )

    args = parser.parse_args()

    if args.normalize_flag:
        args.save_name = "/NORMALIZED/" + args.save_name
    else:
        args.save_name = "/RAW/" + args.save_name

    if args.dataset == "hopper":
        args.save_name = "/HOPPER/" + args.save_name
    elif args.dataset == "halfcheetah":
        args.save_name = "/HALFCHEETAH/" + args.save_name
    elif args.dataset == "cripple_ant":
        args.save_name = "/CRIPPLE_ANT/" + args.save_name
    elif args.dataset == "slim_humanoid":
        args.save_name = "/SLIM_HUMANOID/" + args.save_name
    else:
        raise ValueError(args.dataset)

    if args.deterministic_flag == 0:
        args.save_name += "PROB/"
    else:
        args.save_name += "DET/"

    if args.policy_type in ["RS", "CEM"]:
        args.save_name += "{}/".format(args.policy_type)
        args.save_name += "CAND_{}/".format(args.n_candidate)
    else:
        raise ValueError(args.policy_type)

    # MCL Path
    if args.sim_param_flag == 1:
        args.save_name += "SIM_PARAM/"
    args.save_name += "T_BATCH_{}/".format(args.traj_batch_size)
    args.save_name += "S_BATCH_{}/".format(args.segment_size)
    args.save_name += "T_S_BATCH_{}/".format(args.sample_batch_size)
    args.save_name += "IE_{}/".format(args.ie_itrs)
    args.save_name += "EPOCH_{}/".format(args.n_epochs)
    args.save_name += "ENS_{}/".format(args.ensemble_size)
    args.save_name += "HEAD_{}/".format(args.head_size)
    args.save_name += "SEP_{}/".format(args.sep_layer_size)
    if args.use_ie_flag == 1:
        args.save_name += "USE_IE/"
    if args.non_adaptive_planning_flag == 1:
        args.save_name += "NON_ADAPTIVE/"

    # CaDM Path
    args.save_name += "H_{}/".format(args.history_length)
    args.save_name += "F_{}/".format(args.future_length)
    args.save_name += "BACK_COEFF_{}/".format(args.back_coeff)
    if args.state_diff:
        args.save_name += "DIFF/"
    else:
        args.save_name += "WHOLE/"
    if args.use_global_head_flag == 1:
        args.save_name += "GLOBAL_HEAD/"

    if args.tag != "":
        args.save_name += "tag_{}/".format(args.tag)

    config = {
        # Policy
        "n_candidates": args.n_candidate,
        "horizon": args.horizon,
        # Policy - CEM Hyperparameters
        "use_cem": args.policy_type == "CEM",
        # Environments
        "dataset": args.dataset,
        "normalize_flag": args.normalize_flag,
        "seed": args.seed,
        # Sampling
        "max_path_length": 200,
        "num_rollouts": 10,
        "n_parallel": 5,
        "initial_random_samples": True,
        # Training Hyperparameters
        "n_itr": 10,
        "learning_rate": args.lr,
        "traj_batch_size": args.traj_batch_size,
        "segment_size": args.segment_size,
        "sample_batch_size": args.sample_batch_size,
        "dynamic_model_epochs": args.n_epochs,
        "valid_split_ratio": 0.0,
        "rolling_average_persitency": 0.99,
        # Testing Hyperparameters
        "total_test": args.total_test,
        "no_test_flag": args.no_test_flag,
        "only_test_flag": args.only_test_flag,
        # Dynamics Model Hyperparameters
        "dim_hidden": args.hidden_size,
        "hidden_sizes": (args.hidden_size,) * 4,
        "hidden_nonlinearity": "swish",
        "deterministic": (args.deterministic_flag > 0),
        "weight_decays": (0.000025, 0.00005, 0.000075, 0.000075, 0.0001),
        "weight_decay_coeff": 1.0,
        # PE-TS Hyperparameters
        "ensemble_size": args.ensemble_size,
        "n_particles": args.n_particles,
        "head_size": args.head_size,
        "sep_layer_size": args.sep_layer_size,
        # CaDM Hyperparameters
        "context_hidden_sizes": (256, 128, 64),
        "context_weight_decays": (0.000025, 0.00005, 0.000075),
        "context_out_dim": args.context_out_dim,
        "context_hidden_nonlinearity": "relu",
        "history_length": args.history_length,
        "future_length": args.future_length,
        "state_diff": args.state_diff,
        "back_coeff": args.back_coeff,
        "use_global_head": (args.use_global_head_flag > 0),
        # MCL hyperparameters
        "ie_itrs": args.ie_itrs,
        "use_ie": (args.use_ie_flag > 0),
        "use_simulation_param": (args.sim_param_flag > 0),
        # Ablation
        "non_adaptive_planning": (args.non_adaptive_planning_flag > 0),
        #  Other
        "save_name": args.save_name,
        "discount": 1.0,
        "tag": args.tag,
    }

    run_experiment(config)
