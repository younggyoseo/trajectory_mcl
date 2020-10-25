import os
import numpy as np
import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env


class CrippleAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, cripple_set=[0, 1, 2, 3], extreme_set=[0]):
        self.cripple_mask = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/ant.xml" % dir_path, 5)

        self.n_possible_cripple = 4
        self.cripple_mask = np.ones(self.n_possible_cripple)
        self.cripple_set = cripple_set
        self.extreme_set = extreme_set

        self.cripple_dict = {
            0: [2, 3],  # front L
            1: [4, 5],  # front R
            2: [6, 7],  # back L
            3: [0, 1],  # back R
        }

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()

        utils.EzPickle.__init__(self, cripple_set, extreme_set)

    def _set_observation_space(self, observation):
        super(CrippleAntEnv, self)._set_observation_space(observation)
        proc_observation = self.obs_preproc(observation[None])
        self.proc_observation_space_dims = proc_observation.shape[-1]

    def step(self, a):
        self.xposbefore = self.get_body_com("torso")[0]
        if self.cripple_mask is None:
            a = a
        else:
            a = self.cripple_mask * a
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        reward_ctrl = 0.0
        reward_run = (xposafter - self.xposbefore) / self.dt
        reward_contact = 0.0
        reward_survive = 0.05
        reward = reward_run + reward_ctrl + reward_contact + reward_survive

        done = False
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=reward_run,
                reward_ctrl=reward_ctrl,
                reward_contact=reward_contact,
                reward_survive=reward_survive,
            ),
        )

    def seed(self, seed=None):
        if seed is None:
            self._seed = 0
        else:
            self._seed = seed
        super().seed(seed)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                self.sim.data.get_body_xmat("torso").flat,
                self.get_body_com("torso"),
            ]
        )

    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.xposbefore = self.get_body_com("torso")[0]

        self.change_env()
        return self._get_obs()

    def reward(self, obs, act, next_obs):
        reward_ctrl = 0.0
        vel = (next_obs[..., -3] - obs[..., -3]) / self.dt
        reward_run = vel

        reward_contact = 0.0
        reward_survive = 0.05
        reward = reward_run + reward_ctrl + reward_contact + reward_survive

        return reward

    def tf_reward_fn(self):
        def _thunk(obs, act, next_obs):
            reward_ctrl = 0.0
            vel = (next_obs[..., -3] - obs[..., -3]) / self.dt
            reward_run = vel

            reward_contact = 0.0
            reward_survive = 0.05
            reward = reward_run + reward_ctrl + reward_contact + reward_survive
            return reward

        return _thunk

    def set_crippled_joint(self, value):
        self.cripple_mask = np.ones(self.action_space.shape)
        if value == 0:
            self.cripple_mask[2] = 0
            self.cripple_mask[3] = 0
        elif value == 1:
            self.cripple_mask[4] = 0
            self.cripple_mask[5] = 0
        elif value == 2:
            self.cripple_mask[6] = 0
            self.cripple_mask[7] = 0
        elif value == 3:
            self.cripple_mask[0] = 0
            self.cripple_mask[1] = 0
        elif value == -1:
            pass

        self.crippled_leg = value

        geom_rgba = self._init_geom_rgba.copy()
        if self.crippled_leg == 0:
            geom_rgba[3, :3] = np.array([1, 0, 0])
            geom_rgba[4, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 1:
            geom_rgba[6, :3] = np.array([1, 0, 0])
            geom_rgba[7, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 2:
            geom_rgba[9, :3] = np.array([1, 0, 0])
            geom_rgba[10, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 3:
            geom_rgba[12, :3] = np.array([1, 0, 0])
            geom_rgba[13, :3] = np.array([1, 0, 0])
        self.model.geom_rgba[:] = geom_rgba

        # Make the removed leg not affect anything
        temp_size = self._init_geom_size.copy()
        temp_pos = self._init_geom_pos.copy()

        if self.crippled_leg == 0:
            # Top half
            temp_size[3, 0] = temp_size[3, 0] / 2
            temp_size[3, 1] = temp_size[3, 1] / 2
            # Bottom half
            temp_size[4, 0] = temp_size[4, 0] / 2
            temp_size[4, 1] = temp_size[4, 1] / 2
            temp_pos[4, :] = temp_pos[3, :]

        elif self.crippled_leg == 1:
            # Top half
            temp_size[6, 0] = temp_size[6, 0] / 2
            temp_size[6, 1] = temp_size[6, 1] / 2
            # Bottom half
            temp_size[7, 0] = temp_size[7, 0] / 2
            temp_size[7, 1] = temp_size[7, 1] / 2
            temp_pos[7, :] = temp_pos[6, :]

        elif self.crippled_leg == 2:
            # Top half
            temp_size[9, 0] = temp_size[9, 0] / 2
            temp_size[9, 1] = temp_size[9, 1] / 2
            # Bottom half
            temp_size[10, 0] = temp_size[10, 0] / 2
            temp_size[10, 1] = temp_size[10, 1] / 2
            temp_pos[10, :] = temp_pos[9, :]

        elif self.crippled_leg == 3:
            # Top half
            temp_size[12, 0] = temp_size[12, 0] / 2
            temp_size[12, 1] = temp_size[12, 1] / 2
            # Bottom half
            temp_size[13, 0] = temp_size[13, 0] / 2
            temp_size[13, 1] = temp_size[13, 1] / 2
            temp_pos[13, :] = temp_pos[12, :]

        self.model.geom_size[:] = temp_size
        self.model.geom_pos[:] = temp_pos

    def change_env(self):
        if self.extreme_set == [0]:
            self.crippled_joint = np.array([self.np_random.choice(self.cripple_set)])
        elif self.extreme_set == [1]:
            self.crippled_joint = self.np_random.choice(
                self.cripple_set, 2, replace=False
            )
        else:
            raise ValueError(self.extreme_set)

        self.cripple_mask = np.ones(self.action_space.shape)
        total_crippled_joints = []
        for j in self.crippled_joint:
            total_crippled_joints += self.cripple_dict[j]
        self.cripple_mask[total_crippled_joints] = 0

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_sim_parameters(self):
        return np.array([self.crippled_joint]).reshape(-1)

    def num_modifiable_parameters(self):
        return 1

    def log_diagnostics(self, paths, prefix):
        return
