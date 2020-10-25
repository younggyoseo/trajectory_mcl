from tmcl.policies.base import Policy
from tmcl.utils.serializable import Serializable
import numpy as np


class MPCController(Policy, Serializable):
    def __init__(
        self,
        name,
        env,
        dynamics_model,
        reward_model=None,
        discount=1,
        use_cem=False,
        n_candidates=1024,
        horizon=10,
        use_reward_model=False,
        num_rollouts=10,
        context=False,
        mcl_cadm=False,
    ):
        self.dynamics_model = dynamics_model  # dynamics_model
        self.reward_model = reward_model  # None
        self.discount = discount  # 1
        self.n_candidates = n_candidates  # 2000
        self.horizon = horizon  # 30
        self.use_cem = use_cem  # False
        self.env = env  # OurHalfCheetahEnv
        self.use_reward_model = use_reward_model  # False
        self.context = context
        self.mcl_cadm = mcl_cadm

        self.unwrapped_env = env  # OurHalfCheetahEnv
        while hasattr(self.unwrapped_env, "wrapped_env"):
            self.unwrapped_env = self.unwrapped_env.wrapped_env

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, "reward"), "env must have a reward function"

        Serializable.quick_init(self, locals())
        super(MPCController, self).__init__(env=env)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation, init_mean=None, init_var=None):
        if observation.ndim == 1:
            observation = observation[None]

        if self.use_cem:
            action = self.get_cem_gpu_action(observation, init_mean, init_var)
        else:
            # override with random shooting on GPU
            action = self.get_rs_gpu_action(observation)

        return action, dict()

    def get_actions(
        self,
        observations,
        cp_obs=None,
        cp_act=None,
        history_obs=None,
        history_act=None,
        history_delta=None,
        init_mean=None,
        init_var=None,
        sim_params=None,
    ):
        if self.mcl_cadm:
            if self.use_cem:
                actions = self.get_cem_gpu_action(
                    observations,
                    cp_obs=cp_obs,
                    cp_act=cp_act,
                    init_mean=init_mean,
                    init_var=init_var,
                    history_obs=history_obs,
                    history_act=history_act,
                    history_delta=history_delta,
                    sim_params=sim_params,
                )
            else:
                actions = self.get_rs_gpu_action(
                    observations,
                    cp_obs=cp_obs,
                    cp_act=cp_act,
                    history_obs=history_obs,
                    history_act=history_act,
                    history_delta=history_delta,
                    sim_params=sim_params,
                )
        elif self.context:
            if self.use_cem:
                actions = self.get_cem_gpu_action(
                    observations,
                    init_mean=init_mean,
                    init_var=init_var,
                    cp_obs=cp_obs,
                    cp_act=cp_act,
                    sim_params=sim_params,
                )
            else:
                # override with random shooting on GPU
                actions = self.get_rs_gpu_action(
                    observations, cp_obs=cp_obs, cp_act=cp_act, sim_params=sim_params,
                )
        else:
            if self.use_cem:
                actions = self.get_cem_gpu_action(
                    observations,
                    init_mean=init_mean,
                    init_var=init_var,
                    sim_params=sim_params,
                )
            else:
                # override with random shooting on GPU
                actions = self.get_rs_gpu_action(observations, sim_params=sim_params,)

        return actions, dict()

    def get_random_action(self, n):
        if len(self.unwrapped_env.action_space.shape) == 0:
            return np.random.randint(self.unwrapped_env.action_space.n, size=n)
        else:
            return np.random.uniform(
                low=self.action_space.low,
                high=self.action_space.high,
                size=(n,) + self.action_space.low.shape,
            )

    def get_rs_gpu_action(
        self,
        observations,
        cp_obs=None,
        cp_act=None,
        history_obs=None,
        history_act=None,
        history_delta=None,
        sim_params=None,
    ):
        if self.mcl_cadm:
            action = self.dynamics_model.get_action(
                obs=observations,
                cp_obs=cp_obs,
                cp_act=cp_act,
                history_obs=history_obs,
                history_act=history_act,
                history_delta=history_delta,
                sim_params=sim_params,
            )
        elif self.context:
            action = self.dynamics_model.get_action(
                obs=observations, cp_obs=cp_obs, cp_act=cp_act, sim_params=sim_params
            )
        else:
            action = self.dynamics_model.get_action(
                obs=observations, sim_params=sim_params
            )
        return action

    def get_cem_gpu_action(
        self,
        observations,
        init_mean,
        init_var,
        cp_obs=None,
        cp_act=None,
        history_obs=None,
        history_act=None,
        history_delta=None,
        sim_params=None,
    ):
        if self.mcl_cadm:
            action = self.dynamics_model.get_action(
                obs=observations,
                cp_obs=cp_obs,
                cp_act=cp_act,
                history_obs=history_obs,
                history_act=history_act,
                history_delta=history_delta,
                cem_init_mean=init_mean,
                cem_init_var=init_var,
                sim_params=sim_params,
            )
        elif self.context:
            action = self.dynamics_model.get_action(
                obs=observations,
                cp_obs=cp_obs,
                cp_act=cp_act,
                cem_init_mean=init_mean,
                cem_init_var=init_var,
                sim_params=sim_params,
            )
        else:
            action = self.dynamics_model.get_action(
                obs=observations,
                cem_init_mean=init_mean,
                cem_init_var=init_var,
                sim_params=sim_params,
            )
        return action

    def reset(self, dones=None):
        pass

