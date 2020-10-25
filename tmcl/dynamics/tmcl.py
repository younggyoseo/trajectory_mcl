from tmcl.dynamics.core.layers import MCLMultiHeadedCaDMEnsembleMLP
from tmcl.dynamics.core.layers import MultiHeadedEnsembleContextPredictor
from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
from tmcl.utils.serializable import Serializable
from tmcl.utils import tensor_utils
from tmcl.logger import logger
import time
import joblib
import os
import os.path as osp

tfd = tfp.distributions


class MCLMultiHeadedCaDMDynamicsModel(Serializable):
    """
    Class for MLP continous dynamics model
    """

    _activations = {
        None: tf.identity,
        "relu": tf.nn.relu,
        "tanh": tf.tanh,
        "sigmoid": tf.sigmoid,
        "softmax": tf.nn.softmax,
        "swish": lambda x: x * tf.sigmoid(x),
    }

    def __init__(
        self,
        name,
        env,
        hidden_sizes=(200, 200, 200, 200),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=None,
        traj_batch_size=10,
        sample_batch_size=32,
        segment_size=128,
        learning_rate=0.001,
        normalize_input=True,
        optimizer=tf.train.AdamOptimizer,
        valid_split_ratio=0.2,
        rolling_average_persitency=0.99,
        n_forwards=30,
        n_candidates=2500,
        ensemble_size=5,
        head_size=3,
        n_particles=20,
        use_cem=False,
        deterministic=False,
        weight_decays=(0.0, 0.0, 0.0, 0.0, 0.0),
        weight_decay_coeff=0.0,
        ie_itrs=1,
        use_ie=False,
        use_simulation_param=False,
        simulation_param_dim=1,
        sep_layer_size=0,
        cp_hidden_sizes=(256, 128, 64),
        context_weight_decays=(0.0, 0.0, 0.0, 0.0),
        context_out_dim=10,
        context_hidden_nonlinearity=tf.nn.relu,
        history_length=10,
        future_length=10,
        state_diff=False,
        back_coeff=0.0,
        use_global_head=False,
        non_adaptive_planning=False,
    ):

        Serializable.quick_init(self, locals())

        # Default Attributes
        self.env = env
        self.name = name
        self._dataset = None

        # Dynamics Model Attributes
        self.deterministic = deterministic

        # MPC Attributes
        self.n_forwards = n_forwards
        self.n_candidates = n_candidates
        self.use_cem = use_cem

        # Training Attributes
        self.weight_decays = weight_decays
        self.weight_decay_coeff = weight_decay_coeff
        self.normalization = None
        self.normalize_input = normalize_input
        self.traj_batch_size = traj_batch_size
        self.sample_batch_size = sample_batch_size
        self.segment_size = segment_size
        self.learning_rate = learning_rate
        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

        # PE-TS Attributes
        self.ensemble_size = ensemble_size
        self.n_particles = n_particles

        # MCL Attributes
        self.ie_itrs = ie_itrs
        self.use_ie = use_ie
        self.sep_layer_size = sep_layer_size
        self.use_simulation_param = use_simulation_param
        self.simulation_param_dim = simulation_param_dim
        self.head_size = head_size
        self.use_global_head = use_global_head
        self.non_adaptive_planning = non_adaptive_planning

        # CaDM Attributes
        self.cp_hidden_sizes = cp_hidden_sizes
        self.context_out_dim = context_out_dim
        self.history_length = history_length
        self.future_length = future_length
        self.context_weight_decays = context_weight_decays
        self.state_diff = state_diff
        self.back_coeff = back_coeff

        # Dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape[0]
        self.proc_obs_space_dims = proc_obs_space_dims = env.proc_observation_space_dims
        if len(env.action_space.shape) == 0:
            self.action_space_dims = action_space_dims = env.action_space.n
            self.discrete = True
        else:
            self.action_space_dims = action_space_dims = env.action_space.shape[0]
            self.discrete = False

        hidden_nonlinearity = self._activations[hidden_nonlinearity]
        output_nonlinearity = self._activations[output_nonlinearity]

        with tf.variable_scope(name):
            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.obs_next_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.cp_obs_ph = tf.placeholder(
                tf.float32, shape=(None, obs_space_dims * self.history_length)
            )
            self.cp_act_ph = tf.placeholder(
                tf.float32, shape=(None, action_space_dims * self.history_length)
            )
            self.simulation_param_ph = tf.placeholder(
                tf.float32, shape=(None, simulation_param_dim)
            )

            self.bs_obs_ph = tf.placeholder(
                tf.float32, shape=(ensemble_size, None, None, obs_space_dims)
            )  # [ensemble_size, trajectory, path_length, obs_space_dims]
            self.bs_obs_next_ph = tf.placeholder(
                tf.float32, shape=(ensemble_size, None, None, obs_space_dims)
            )
            self.bs_act_ph = tf.placeholder(
                tf.float32, shape=(ensemble_size, None, None, action_space_dims)
            )
            self.bs_delta_ph = tf.placeholder(
                tf.float32, shape=(ensemble_size, None, None, obs_space_dims)
            )
            self.bs_back_delta_ph = tf.placeholder(
                tf.float32, shape=(ensemble_size, None, None, obs_space_dims)
            )
            self.bs_cp_obs_ph = tf.placeholder(
                tf.float32,
                shape=(ensemble_size, None, None, obs_space_dims * self.history_length),
            )
            self.bs_cp_act_ph = tf.placeholder(
                tf.float32,
                shape=(
                    ensemble_size,
                    None,
                    None,
                    action_space_dims * self.history_length,
                ),
            )
            self.bs_simulation_param_ph = tf.placeholder(
                tf.float32, shape=(ensemble_size, None, None, simulation_param_dim)
            )

            self.norm_obs_mean_ph = tf.placeholder(
                tf.float32, shape=(proc_obs_space_dims,)
            )
            self.norm_obs_std_ph = tf.placeholder(
                tf.float32, shape=(proc_obs_space_dims,)
            )
            self.norm_act_mean_ph = tf.placeholder(
                tf.float32, shape=(action_space_dims,)
            )
            self.norm_act_std_ph = tf.placeholder(
                tf.float32, shape=(action_space_dims,)
            )
            self.norm_delta_mean_ph = tf.placeholder(
                tf.float32, shape=(obs_space_dims,)
            )
            self.norm_delta_std_ph = tf.placeholder(tf.float32, shape=(obs_space_dims,))
            self.norm_cp_obs_mean_ph = tf.placeholder(
                tf.float32, shape=(obs_space_dims * self.history_length,)
            )
            self.norm_cp_obs_std_ph = tf.placeholder(
                tf.float32, shape=(obs_space_dims * self.history_length,)
            )
            self.norm_cp_act_mean_ph = tf.placeholder(
                tf.float32, shape=(action_space_dims * self.history_length,)
            )
            self.norm_cp_act_std_ph = tf.placeholder(
                tf.float32, shape=(action_space_dims * self.history_length,)
            )
            self.norm_back_delta_mean_ph = tf.placeholder(
                tf.float32, shape=(obs_space_dims,)
            )
            self.norm_back_delta_std_ph = tf.placeholder(
                tf.float32, shape=(obs_space_dims,)
            )

            self.cem_init_mean_ph = tf.placeholder(
                tf.float32, shape=(None, self.n_forwards, action_space_dims)
            )
            self.cem_init_var_ph = tf.placeholder(
                tf.float32, shape=(None, self.n_forwards, action_space_dims)
            )

            self.history_obs_ph = tf.placeholder(
                tf.float32, shape=(None, None, obs_space_dims)
            )  # [batch_size, history_length, obs_space_dims]
            self.history_act_ph = tf.placeholder(
                tf.float32, shape=(None, None, action_space_dims)
            )
            self.history_delta_ph = tf.placeholder(
                tf.float32, shape=(None, None, obs_space_dims)
            )

            self.min_traj_idxs_ph = tf.placeholder(
                tf.int32, shape=(ensemble_size, None)
            )  # [ensemble_size, trajectory]
            self.min_traj_back_idxs_ph = tf.placeholder(
                tf.int32, shape=(ensemble_size, None)
            )  # [ensemble_size, trajectory]

            traj_size_tensor = tf.shape(self.bs_obs_ph)[1]
            traj_length_tensor = tf.shape(self.bs_obs_ph)[2]

            bs_obs = tf.reshape(self.bs_obs_ph, [ensemble_size, -1, obs_space_dims])
            bs_obs_next = tf.reshape(
                self.bs_obs_next_ph, [ensemble_size, -1, obs_space_dims]
            )
            bs_act = tf.reshape(self.bs_act_ph, [ensemble_size, -1, action_space_dims])
            bs_cp_obs = tf.reshape(
                self.bs_cp_obs_ph,
                [ensemble_size, -1, obs_space_dims * self.history_length],
            )
            bs_cp_act = tf.reshape(
                self.bs_cp_act_ph,
                [ensemble_size, -1, action_space_dims * self.history_length],
            )
            bs_sim_param = tf.reshape(
                self.bs_simulation_param_ph, [ensemble_size, -1, simulation_param_dim]
            )

            with tf.variable_scope("context_model"):
                cp = MultiHeadedEnsembleContextPredictor(
                    name,
                    output_dim=0,
                    input_dim=0,
                    context_dim=(obs_space_dims + action_space_dims)
                    * self.history_length,
                    context_hidden_sizes=self.cp_hidden_sizes,
                    output_nonlinearity=output_nonlinearity,
                    ensemble_size=self.ensemble_size,
                    context_weight_decays=self.context_weight_decays,
                    bs_input_cp_obs_var=bs_cp_obs,
                    bs_input_cp_act_var=bs_cp_act,
                    norm_cp_obs_mean_var=self.norm_cp_obs_mean_ph,
                    norm_cp_obs_std_var=self.norm_cp_obs_std_ph,
                    norm_cp_act_mean_var=self.norm_cp_act_mean_ph,
                    norm_cp_act_std_var=self.norm_cp_act_std_ph,
                    context_out_dim=self.context_out_dim,
                    head_size=self.head_size,
                    use_global_head=self.use_global_head,
                )
                self.bs_cp_var = (
                    cp.context_output_var
                )  # [head_size, ensemble_size, None, context_dim]

            # create MLP
            with tf.variable_scope("ff_model"):
                mlp = MCLMultiHeadedCaDMEnsembleMLP(
                    name,
                    input_dim=0,
                    output_dim=obs_space_dims,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    input_obs_dim=obs_space_dims,
                    input_act_dim=action_space_dims,
                    input_obs_var=self.obs_ph,
                    input_act_var=self.act_ph,
                    input_history_obs_var=self.history_obs_ph,
                    input_history_act_var=self.history_act_ph,
                    input_history_delta_var=self.history_delta_ph,
                    n_forwards=self.n_forwards,
                    reward_fn=env.tf_reward_fn(),
                    n_candidates=self.n_candidates,
                    discrete=self.discrete,
                    bs_input_obs_var=bs_obs,
                    bs_input_act_var=bs_act,
                    ensemble_size=self.ensemble_size,
                    head_size=self.head_size,
                    n_particles=self.n_particles,
                    # Normalization
                    norm_obs_mean_var=self.norm_obs_mean_ph,
                    norm_obs_std_var=self.norm_obs_std_ph,
                    norm_act_mean_var=self.norm_act_mean_ph,
                    norm_act_std_var=self.norm_act_std_ph,
                    norm_delta_mean_var=self.norm_delta_mean_ph,
                    norm_delta_std_var=self.norm_delta_std_ph,
                    norm_cp_obs_mean_var=self.norm_cp_obs_mean_ph,
                    norm_cp_obs_std_var=self.norm_cp_obs_std_ph,
                    norm_cp_act_mean_var=self.norm_cp_act_mean_ph,
                    norm_cp_act_std_var=self.norm_cp_act_std_ph,
                    norm_back_delta_mean_var=None,
                    norm_back_delta_std_var=None,
                    obs_preproc_fn=env.obs_preproc,
                    obs_postproc_fn=env.obs_postproc,
                    use_cem=self.use_cem,
                    cem_init_mean_var=self.cem_init_mean_ph,
                    cem_init_var_var=self.cem_init_var_ph,
                    deterministic=self.deterministic,
                    weight_decays=self.weight_decays,
                    use_simulation_param=self.use_simulation_param,
                    simulation_param_dim=self.simulation_param_dim,
                    simulation_param_var=self.simulation_param_ph,
                    bs_input_sim_param_var=bs_sim_param,
                    sep_layer_size=self.sep_layer_size,
                    # CaDM
                    context_obs_var=self.cp_obs_ph,
                    context_act_var=self.cp_act_ph,
                    cp_forward=cp.forward,
                    bs_input_cp_var=self.bs_cp_var,
                    context_out_dim=self.context_out_dim,
                    build_policy_graph=True,
                    # Ablation
                    non_adaptive_planning=self.non_adaptive_planning,
                )

            with tf.variable_scope("bb_model"):
                back_mlp = MCLMultiHeadedCaDMEnsembleMLP(
                    name,
                    input_dim=0,
                    output_dim=obs_space_dims,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    input_obs_dim=obs_space_dims,
                    input_act_dim=action_space_dims,
                    input_obs_var=self.obs_next_ph,
                    input_act_var=self.act_ph,
                    input_history_obs_var=self.history_obs_ph,
                    input_history_act_var=self.history_act_ph,
                    input_history_delta_var=self.history_delta_ph,
                    n_forwards=self.n_forwards,
                    reward_fn=env.tf_reward_fn(),
                    n_candidates=self.n_candidates,
                    discrete=self.discrete,
                    bs_input_obs_var=bs_obs_next,
                    bs_input_act_var=bs_act,
                    ensemble_size=self.ensemble_size,
                    head_size=self.head_size,
                    n_particles=self.n_particles,
                    # Normalization
                    norm_obs_mean_var=self.norm_obs_mean_ph,
                    norm_obs_std_var=self.norm_obs_std_ph,
                    norm_act_mean_var=self.norm_act_mean_ph,
                    norm_act_std_var=self.norm_act_std_ph,
                    norm_delta_mean_var=None,
                    norm_delta_std_var=None,
                    norm_cp_obs_mean_var=self.norm_cp_obs_mean_ph,
                    norm_cp_obs_std_var=self.norm_cp_obs_std_ph,
                    norm_cp_act_mean_var=self.norm_cp_act_mean_ph,
                    norm_cp_act_std_var=self.norm_cp_act_std_ph,
                    norm_back_delta_mean_var=self.norm_back_delta_mean_ph,
                    norm_back_delta_std_var=self.norm_back_delta_std_ph,
                    obs_preproc_fn=env.obs_preproc,
                    obs_postproc_fn=env.obs_postproc,
                    use_cem=self.use_cem,
                    cem_init_mean_var=self.cem_init_mean_ph,
                    cem_init_var_var=self.cem_init_var_ph,
                    deterministic=self.deterministic,
                    weight_decays=self.weight_decays,
                    use_simulation_param=self.use_simulation_param,
                    simulation_param_dim=self.simulation_param_dim,
                    simulation_param_var=self.simulation_param_ph,
                    bs_input_sim_param_var=bs_sim_param,
                    sep_layer_size=self.sep_layer_size,
                    # CaDM
                    context_obs_var=self.cp_obs_ph,
                    context_act_var=self.cp_act_ph,
                    cp_forward=None,
                    bs_input_cp_var=self.bs_cp_var,
                    context_out_dim=self.context_out_dim,
                    build_policy_graph=False,
                    non_adaptive_planning=False,
                )

            self.params = tf.trainable_variables()
            self.delta_pred = mlp.output_var
            self.embedding = mlp.embedding

            # 1. Forward Dynamics Prediction Loss
            # Outputs from Dynamics Model are normalized delta predictions
            mu, logvar = (
                mlp.mu,
                mlp.logvar,
            )  # [head_size, ensemble_size, traj_size, traj_length, obs_space_dims]

            bs_normalized_delta = normalize(
                self.bs_delta_ph, self.norm_delta_mean_ph, self.norm_delta_std_ph
            )
            bs_mu = tf.reshape(
                mu,
                [
                    head_size,
                    ensemble_size,
                    traj_size_tensor,
                    traj_length_tensor,
                    obs_space_dims,
                ],
            )
            bs_logvar = tf.reshape(
                logvar,
                [
                    head_size,
                    ensemble_size,
                    traj_size_tensor,
                    traj_length_tensor,
                    obs_space_dims,
                ],
            )

            bs_traj_losses = tf.reduce_mean(
                tf.reduce_mean(tf.square(bs_mu - bs_normalized_delta), axis=-1), axis=-1
            )  # [head_size, ensemble_size, traj_size]
            bs_traj_losses = tf.transpose(
                bs_traj_losses, [1, 2, 0]
            )  # [ensemble_size, traj_size, head_size]

            self.min_traj_idxs = tf.reshape(
                tf.nn.top_k(-1.0 * bs_traj_losses)[1], [ensemble_size, traj_size_tensor]
            )

            flat_min_traj_idxs = tf.reshape(
                self.min_traj_idxs_ph, [-1]
            )  # [ensemble_size * traj_size]
            flat_bs_traj_losses = tf.reshape(
                bs_traj_losses, [-1, head_size]
            )  # [ensemble_size * traj_size, head_size]
            nd_idxs = tf.transpose(
                tf.stack(
                    [tf.range(tf.shape(flat_min_traj_idxs)[0]), flat_min_traj_idxs],
                    axis=0,
                )
            )  # [ensemble_size * traj_size, 2]
            min_traj_losses = tf.gather_nd(
                flat_bs_traj_losses, nd_idxs
            )  # [ensemble_size * traj_size]
            min_traj_losses = tf.reshape(
                min_traj_losses, [ensemble_size, traj_size_tensor]
            )

            # Define IE Loss
            ie_mse_loss = tf.reduce_mean(bs_traj_losses, axis=[1, 2])

            # max_traj_losses, self.min_traj_idxs = tf.nn.top_k(-1. * bs_traj_losses) # [ensemble_size, traj_size]
            # min_traj_losses = -1. * max_traj_losses # [ensemble_size, traj_size]

            mse_loss = 0.0
            for head_idx in range(self.head_size):
                head_idx_bool = tf.cast(
                    tf.equal(self.min_traj_idxs_ph, head_idx), tf.float32
                )
                mse_loss += tf.reduce_sum(
                    min_traj_losses * head_idx_bool, axis=1
                ) / tf.maximum(tf.reduce_sum(head_idx_bool, axis=1), 1.0)

            self.mse_loss = tf.reduce_sum(mse_loss)
            self.ie_mse_loss = tf.reduce_sum(ie_mse_loss)
            self.norm_pred_error = tf.reduce_mean(min_traj_losses)

            # 2. Backward Dynamics Prediction Loss
            # Outputs from Dynamics Model are normalized delta predictions
            back_mu, back_logvar = (
                back_mlp.mu,
                back_mlp.logvar,
            )  # [head_size, ensemble_size, traj_size, traj_length, obs_space_dims]

            bs_normalized_back_delta = normalize(
                self.bs_back_delta_ph,
                self.norm_back_delta_mean_ph,
                self.norm_back_delta_std_ph,
            )
            bs_back_mu = tf.reshape(
                back_mu,
                [
                    head_size,
                    ensemble_size,
                    traj_size_tensor,
                    traj_length_tensor,
                    obs_space_dims,
                ],
            )
            bs_back_logvar = tf.reshape(
                back_logvar,
                [
                    head_size,
                    ensemble_size,
                    traj_size_tensor,
                    traj_length_tensor,
                    obs_space_dims,
                ],
            )

            bs_traj_back_losses = tf.reduce_mean(
                tf.reduce_mean(
                    tf.square(bs_back_mu - bs_normalized_back_delta), axis=-1
                ),
                axis=-1,
            )  # [head_size, ensemble_size, traj_size]
            bs_traj_back_losses = tf.transpose(
                bs_traj_back_losses, [1, 2, 0]
            )  # [ensemble_size, traj_size, head_size]

            self.min_traj_back_idxs = tf.reshape(
                tf.nn.top_k(-1.0 * bs_traj_losses)[1], [ensemble_size, traj_size_tensor]
            )

            flat_min_traj_back_idxs = tf.reshape(
                self.min_traj_back_idxs_ph, [-1]
            )  # [ensemble_size * traj_size]
            flat_bs_traj_back_losses = tf.reshape(
                bs_traj_back_losses, [-1, head_size]
            )  # [ensemble_size * traj_size, head_size]
            nd_back_idxs = tf.transpose(
                tf.stack(
                    [
                        tf.range(tf.shape(flat_min_traj_back_idxs)[0]),
                        flat_min_traj_back_idxs,
                    ],
                    axis=0,
                )
            )  # [ensemble_size * traj_size, 2]
            min_traj_back_losses = tf.gather_nd(
                flat_bs_traj_back_losses, nd_back_idxs
            )  # [ensemble_size * traj_size]
            min_traj_back_losses = tf.reshape(
                min_traj_back_losses, [ensemble_size, traj_size_tensor]
            )

            # Define Back IE Loss
            ie_back_mse_loss = tf.reduce_mean(bs_traj_back_losses, axis=[1, 2])

            back_mse_loss = 0.0
            for head_idx in range(self.head_size):
                head_idx_bool = tf.cast(
                    tf.equal(self.min_traj_back_idxs_ph, head_idx), tf.float32
                )
                back_mse_loss += tf.reduce_sum(
                    min_traj_back_losses * head_idx_bool, axis=1
                ) / tf.maximum(tf.reduce_sum(head_idx_bool, axis=1), 1.0)

            self.back_mse_loss = tf.reduce_sum(back_mse_loss)
            self.ie_back_mse_loss = tf.reduce_sum(ie_back_mse_loss)

            # Weight Decay
            self.l2_reg_loss = tf.reduce_sum(mlp.l2_regs)
            self.back_l2_reg_loss = tf.reduce_sum(back_mlp.l2_regs)

            if self.deterministic:
                self.recon_loss = self.mse_loss
                self.ie_recon_loss = self.ie_mse_loss
                self.back_recon_loss = self.back_mse_loss
                self.ie_back_recon_loss = self.ie_back_mse_loss

                self.loss = self.mse_loss + self.back_coeff * self.back_mse_loss
                self.loss += (
                    self.l2_reg_loss + self.back_l2_reg_loss
                ) * self.weight_decay_coeff
                self.ie_loss = (
                    self.ie_mse_loss + self.back_coeff * self.ie_back_mse_loss
                )
                self.ie_loss += (
                    self.l2_reg_loss + self.back_l2_reg_loss
                ) * self.weight_decay_coeff
            else:
                # Forward
                bs_invvar = tf.exp(-bs_logvar)
                bs_mu_traj_losses = tf.reduce_mean(
                    tf.reduce_mean(
                        tf.square(bs_mu - bs_normalized_delta) * bs_invvar, axis=-1
                    ),
                    axis=-1,
                )
                bs_var_traj_losses = tf.reduce_mean(
                    tf.reduce_mean(bs_logvar, axis=-1), axis=-1
                )
                bs_traj_losses = (
                    bs_mu_traj_losses + bs_var_traj_losses
                )  # [head_size, ensemble_size, traj_size]

                bs_traj_losses = tf.transpose(
                    bs_traj_losses, [1, 2, 0]
                )  # [ensemble_size, traj_size, head_size]

                self.min_traj_idxs = tf.reshape(
                    tf.nn.top_k(-1.0 * bs_traj_losses)[1],
                    [ensemble_size, traj_size_tensor],
                )  # [ensemble_size, traj_size]

                flat_min_traj_idxs = tf.reshape(
                    self.min_traj_idxs_ph, [-1]
                )  # [ensemble_size * traj_size]
                flat_bs_traj_losses = tf.reshape(
                    bs_traj_losses, [-1, head_size]
                )  # [ensemble_size * traj_size, head_size]
                nd_idxs = tf.transpose(
                    tf.stack(
                        [tf.range(tf.shape(flat_min_traj_idxs)[0]), flat_min_traj_idxs],
                        axis=0,
                    )
                )  # [ensemble_size * traj_size, 2]
                min_traj_losses = tf.gather_nd(
                    flat_bs_traj_losses, nd_idxs
                )  # [ensemble_size * traj_size]
                min_traj_losses = tf.reshape(
                    min_traj_losses, [ensemble_size, traj_size_tensor]
                )
                ie_recon_loss = tf.reduce_mean(bs_traj_losses, axis=[1, 2])
                recon_loss = 0.0
                for head_idx in range(self.head_size):
                    head_idx_bool = tf.cast(
                        tf.equal(self.min_traj_idxs_ph, head_idx), tf.float32
                    )
                    recon_loss += tf.reduce_sum(
                        min_traj_losses * head_idx_bool, axis=1
                    ) / tf.maximum(tf.reduce_sum(head_idx_bool, axis=1), 1.0)

                self.recon_loss = tf.reduce_sum(recon_loss)
                self.ie_recon_loss = tf.reduce_sum(ie_recon_loss)
                self.reg_loss = 0.01 * tf.reduce_sum(
                    mlp.max_logvar
                ) - 0.01 * tf.reduce_sum(mlp.min_logvar)

                # Backward
                bs_back_invvar = tf.exp(-bs_back_logvar)
                bs_mu_traj_back_losses = tf.reduce_mean(
                    tf.reduce_mean(
                        tf.square(bs_back_mu - bs_normalized_back_delta)
                        * bs_back_invvar,
                        axis=-1,
                    ),
                    axis=-1,
                )
                bs_var_traj_back_losses = tf.reduce_mean(
                    tf.reduce_mean(bs_back_logvar, axis=-1), axis=-1
                )
                bs_traj_back_losses = (
                    bs_mu_traj_back_losses + bs_var_traj_back_losses
                )  # [head_size, ensemble_size, traj_size]

                bs_traj_back_losses = tf.transpose(
                    bs_traj_back_losses, [1, 2, 0]
                )  # [ensemble_size, traj_size, head_size]

                self.min_traj_back_idxs = tf.reshape(
                    tf.nn.top_k(-1.0 * bs_traj_back_losses)[1],
                    [ensemble_size, traj_size_tensor],
                )  # [ensemble_size, traj_size]

                flat_min_traj_back_idxs = tf.reshape(
                    self.min_traj_back_idxs_ph, [-1]
                )  # [ensemble_size * traj_size]
                flat_bs_traj_back_losses = tf.reshape(
                    bs_traj_back_losses, [-1, head_size]
                )  # [ensemble_size * traj_size, head_size]
                nd_idxs = tf.transpose(
                    tf.stack(
                        [
                            tf.range(tf.shape(flat_min_traj_back_idxs)[0]),
                            flat_min_traj_back_idxs,
                        ],
                        axis=0,
                    )
                )  # [ensemble_size * traj_size, 2]
                min_traj_back_losses = tf.gather_nd(
                    flat_bs_traj_back_losses, nd_idxs
                )  # [ensemble_size * traj_size]
                min_traj_back_losses = tf.reshape(
                    min_traj_back_losses, [ensemble_size, traj_size_tensor]
                )
                ie_back_recon_loss = tf.reduce_mean(bs_traj_back_losses, axis=[1, 2])
                back_recon_loss = 0.0
                for head_idx in range(self.head_size):
                    head_idx_bool = tf.cast(
                        tf.equal(self.min_traj_back_idxs_ph, head_idx), tf.float32
                    )
                    back_recon_loss += tf.reduce_sum(
                        min_traj_back_losses * head_idx_bool, axis=1
                    ) / tf.maximum(tf.reduce_sum(head_idx_bool, axis=1), 1.0)

                self.back_recon_loss = tf.reduce_sum(back_recon_loss)
                self.ie_back_recon_loss = tf.reduce_sum(ie_back_recon_loss)
                self.back_reg_loss = 0.01 * tf.reduce_sum(
                    back_mlp.max_logvar
                ) - 0.01 * tf.reduce_sum(back_mlp.min_logvar)

                self.loss = (
                    self.recon_loss
                    + self.reg_loss
                    + self.back_coeff * (self.back_recon_loss + self.back_reg_loss)
                )
                self.loss += (
                    self.l2_reg_loss + self.back_l2_reg_loss
                ) * self.weight_decay_coeff
                self.ie_loss = (
                    self.ie_recon_loss
                    + self.reg_loss
                    + self.back_coeff * (self.ie_back_recon_loss + self.back_reg_loss)
                )
                self.ie_loss += (
                    self.l2_reg_loss + self.back_l2_reg_loss
                ) * self.weight_decay_coeff

            self.optimizer = optimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
            self.ie_train_op = self.optimizer.minimize(self.ie_loss)

            # tensor_utils
            self._get_cem_action = tensor_utils.compile_function(
                [
                    self.obs_ph,
                    self.cp_obs_ph,
                    self.cp_act_ph,
                    self.history_obs_ph,
                    self.history_act_ph,
                    self.history_delta_ph,
                    self.norm_obs_mean_ph,
                    self.norm_obs_std_ph,
                    self.norm_act_mean_ph,
                    self.norm_act_std_ph,
                    self.norm_delta_mean_ph,
                    self.norm_delta_std_ph,
                    self.norm_cp_obs_mean_ph,
                    self.norm_cp_obs_std_ph,
                    self.norm_cp_act_mean_ph,
                    self.norm_cp_act_std_ph,
                    self.cem_init_mean_ph,
                    self.cem_init_var_ph,
                    self.simulation_param_ph,
                ],
                mlp.optimal_action_var,
            )
            self._get_rs_action = tensor_utils.compile_function(
                [
                    self.obs_ph,
                    self.cp_obs_ph,
                    self.cp_act_ph,
                    self.history_obs_ph,
                    self.history_act_ph,
                    self.history_delta_ph,
                    self.norm_obs_mean_ph,
                    self.norm_obs_std_ph,
                    self.norm_act_mean_ph,
                    self.norm_act_std_ph,
                    self.norm_delta_mean_ph,
                    self.norm_delta_std_ph,
                    self.norm_cp_obs_mean_ph,
                    self.norm_cp_obs_std_ph,
                    self.norm_cp_act_mean_ph,
                    self.norm_cp_act_std_ph,
                    self.simulation_param_ph,
                ],
                mlp.optimal_action_var,
            )

            self._get_context_pred = tensor_utils.compile_function(
                [
                    self.cp_obs_ph,
                    self.cp_act_ph,
                    self.norm_cp_obs_mean_ph,
                    self.norm_cp_obs_std_ph,
                    self.norm_cp_act_mean_ph,
                    self.norm_cp_act_std_ph,
                ],
                mlp.inference_cp_var,
            )  ## inference cp var

            self._get_embedding = tensor_utils.compile_function(
                [
                    self.bs_obs_ph,
                    self.bs_act_ph,
                    self.norm_obs_mean_ph,
                    self.norm_obs_std_ph,
                    self.norm_act_mean_ph,
                    self.norm_act_std_ph,
                ],
                mlp.embedding,
            )  ## inference cp var

    def get_action(
        self,
        obs,
        cp_obs,
        cp_act,
        history_obs,
        history_act,
        history_delta,
        cem_init_mean=None,
        cem_init_var=None,
        sim_params=None,
    ):
        (
            norm_obs_mean,
            norm_obs_std,
            norm_act_mean,
            norm_act_std,
            norm_delta_mean,
            norm_delta_std,
            norm_cp_obs_mean,
            norm_cp_obs_std,
            norm_cp_act_mean,
            norm_cp_act_std,
            _,
            _,
        ) = self.get_normalization_stats()

        if sim_params is None:
            print(sim_params)
            sim_params = np.zeros([obs.shape[0], self.simulation_param_dim])

        if cem_init_mean is not None:
            action = self._get_cem_action(
                obs,
                cp_obs,
                cp_act,
                history_obs,
                history_act,
                history_delta,
                norm_obs_mean,
                norm_obs_std,
                norm_act_mean,
                norm_act_std,
                norm_delta_mean,
                norm_delta_std,
                norm_cp_obs_mean,
                norm_cp_obs_std,
                norm_cp_act_mean,
                norm_cp_act_std,
                cem_init_mean,
                cem_init_var,
                sim_params,
            )
        else:
            action = self._get_rs_action(
                obs,
                cp_obs,
                cp_act,
                history_obs,
                history_act,
                history_delta,
                norm_obs_mean,
                norm_obs_std,
                norm_act_mean,
                norm_act_std,
                norm_delta_mean,
                norm_delta_std,
                norm_cp_obs_mean,
                norm_cp_obs_std,
                norm_cp_act_mean,
                norm_cp_act_std,
                sim_params,
            )
        if not self.discrete:
            action = np.minimum(np.maximum(action, -1.0), 1.0)
        # print(head, pe, sim_params)
        return action

    def get_context_pred(self, cp_obs, cp_act):
        (
            _,
            _,
            _,
            _,
            _,
            _,
            norm_cp_obs_mean,
            norm_cp_obs_std,
            norm_cp_act_mean,
            norm_cp_act_std,
            *_,
        ) = self.get_normalization_stats()
        context = self._get_context_pred(
            cp_obs,
            cp_act,
            norm_cp_obs_mean,
            norm_cp_obs_std,
            norm_cp_act_mean,
            norm_cp_act_std,
        )
        return context

    def get_embedding(self, obs, act):
        (
            norm_obs_mean,
            norm_obs_std,
            norm_act_mean,
            norm_act_std,
            norm_delta_mean,
            norm_delta_std,
            norm_cp_obs_mean,
            norm_cp_obs_std,
            norm_cp_act_mean,
            norm_cp_act_std,
            norm_back_delta_mean,
            norm_back_delta_std,
        ) = self.get_normalization_stats()
        context = self._get_embedding(
            obs, act, norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std
        )
        return context

    def fit(
        self,
        obs,
        act,
        obs_next,
        sim_params,
        cp_obs,
        cp_act,
        future_bool,
        itr=0,
        epochs=1000,
        compute_normalization=True,
        valid_split_ratio=None,
        rolling_average_persitency=None,
        verbose=False,
        log_tabular=False,
        max_logging=20,
        log_only_pred_error=False,
    ):

        assert (
            obs.ndim == 3 and obs.shape[2] == self.obs_space_dims * self.future_length
        )
        assert (
            obs_next.ndim == 3
            and obs_next.shape[2] == self.obs_space_dims * self.future_length
        )
        assert (
            act.ndim == 3
            and act.shape[2] == self.action_space_dims * self.future_length
        )
        assert cp_obs.ndim == 3 and cp_obs.shape[2] == (
            self.obs_space_dims * self.history_length
        )
        assert cp_act.ndim == 3 and cp_act.shape[2] == (
            self.action_space_dims * self.history_length
        )
        assert future_bool.ndim == 3 and future_bool.shape[2] == self.future_length

        if valid_split_ratio is None:
            valid_split_ratio = self.valid_split_ratio
        if rolling_average_persitency is None:
            rolling_average_persitency = self.rolling_average_persitency

        assert 1 > valid_split_ratio >= 0

        sess = tf.get_default_session()

        obs_shape = obs.shape
        obs_next_shape = obs_next.shape

        obs = obs.reshape(-1, self.obs_space_dims)
        obs_next = obs_next.reshape(-1, self.obs_space_dims)
        delta = self.env.targ_proc(obs, obs_next)
        back_delta = self.env.targ_proc(obs_next, obs)

        obs = obs.reshape(obs_shape)
        obs_next = obs_next.reshape(obs_next_shape)
        delta = delta.reshape(obs_shape)
        back_delta = back_delta.reshape(obs_next_shape)

        single_obs = obs[..., : self.obs_space_dims]
        single_obs_next = obs_next[..., : self.obs_space_dims]
        single_act = act[..., : self.action_space_dims]
        single_delta = delta[..., : self.obs_space_dims]
        single_back_delta = back_delta[..., : self.obs_space_dims]

        if self._dataset is None:
            self._dataset = dict(
                obs=obs,
                act=act,
                delta=delta,
                cp_obs=cp_obs,
                cp_act=cp_act,
                future_bool=future_bool,
                obs_next=obs_next,
                back_delta=back_delta,
                sim_params=sim_params,
                single_obs=single_obs,
                single_obs_next=single_obs_next,
                single_act=single_act,
                single_delta=single_delta,
                single_back_delta=single_back_delta,
            )
        else:
            self._dataset["obs"] = np.concatenate([self._dataset["obs"], obs])
            self._dataset["act"] = np.concatenate([self._dataset["act"], act])
            self._dataset["delta"] = np.concatenate([self._dataset["delta"], delta])
            self._dataset["cp_obs"] = np.concatenate([self._dataset["cp_obs"], cp_obs])
            self._dataset["cp_act"] = np.concatenate([self._dataset["cp_act"], cp_act])
            self._dataset["future_bool"] = np.concatenate(
                [self._dataset["future_bool"], future_bool]
            )
            self._dataset["obs_next"] = np.concatenate(
                [self._dataset["obs_next"], obs_next]
            )
            self._dataset["back_delta"] = np.concatenate(
                [self._dataset["back_delta"], back_delta]
            )
            self._dataset["sim_params"] = np.concatenate(
                [self._dataset["sim_params"], sim_params]
            )

            self._dataset["single_obs"] = np.concatenate(
                [self._dataset["single_obs"], single_obs]
            )
            self._dataset["single_obs_next"] = np.concatenate(
                [self._dataset["single_obs_next"], single_obs_next]
            )
            self._dataset["single_act"] = np.concatenate(
                [self._dataset["single_act"], single_act]
            )
            self._dataset["single_delta"] = np.concatenate(
                [self._dataset["single_delta"], single_delta]
            )
            self._dataset["single_back_delta"] = np.concatenate(
                [self._dataset["single_back_delta"], single_back_delta]
            )

        self.compute_normalization(
            self._dataset["single_obs"],
            self._dataset["single_act"],
            self._dataset["single_delta"],
            self._dataset["cp_obs"],
            self._dataset["cp_act"],
            self._dataset["single_back_delta"],
        )

        dataset_size = self._dataset["obs"].shape[0]
        n_valid_split = min(int(dataset_size * valid_split_ratio), max_logging)
        permutation = np.random.permutation(dataset_size)
        train_obs, valid_obs = (
            self._dataset["obs"][permutation[n_valid_split:]],
            self._dataset["obs"][permutation[:n_valid_split]],
        )
        train_act, valid_act = (
            self._dataset["act"][permutation[n_valid_split:]],
            self._dataset["act"][permutation[:n_valid_split]],
        )
        train_delta, valid_delta = (
            self._dataset["delta"][permutation[n_valid_split:]],
            self._dataset["delta"][permutation[:n_valid_split]],
        )
        train_cp_obs, valid_cp_obs = (
            self._dataset["cp_obs"][permutation[n_valid_split:]],
            self._dataset["cp_obs"][permutation[:n_valid_split]],
        )
        train_cp_act, valid_cp_act = (
            self._dataset["cp_act"][permutation[n_valid_split:]],
            self._dataset["cp_act"][permutation[:n_valid_split]],
        )
        train_obs_next, valid_obs_next = (
            self._dataset["obs_next"][permutation[n_valid_split:]],
            self._dataset["obs_next"][permutation[:n_valid_split]],
        )
        train_future_bool, valid_future_bool = (
            self._dataset["future_bool"][permutation[n_valid_split:]],
            self._dataset["future_bool"][permutation[:n_valid_split]],
        )
        train_back_delta, valid_back_delta = (
            self._dataset["back_delta"][permutation[n_valid_split:]],
            self._dataset["back_delta"][permutation[:n_valid_split]],
        )
        train_sim_params, valid_sim_params = (
            self._dataset["sim_params"][permutation[n_valid_split:]],
            self._dataset["sim_params"][permutation[:n_valid_split]],
        )

        # We will calculate head idxs with single obs/act/delta/back_delta
        train_single_obs, valid_single_obs = (
            self._dataset["single_obs"][permutation[n_valid_split:]],
            self._dataset["single_obs"][permutation[:n_valid_split]],
        )
        train_single_obs_next, valid_single_obs_next = (
            self._dataset["single_obs_next"][permutation[n_valid_split:]],
            self._dataset["single_obs_next"][permutation[:n_valid_split]],
        )
        train_single_act, valid_single_act = (
            self._dataset["single_act"][permutation[n_valid_split:]],
            self._dataset["single_act"][permutation[:n_valid_split]],
        )
        train_single_delta, valid_single_delta = (
            self._dataset["single_delta"][permutation[n_valid_split:]],
            self._dataset["single_delta"][permutation[:n_valid_split]],
        )
        train_single_back_delta, valid_single_back_delta = (
            self._dataset["single_back_delta"][permutation[n_valid_split:]],
            self._dataset["single_back_delta"][permutation[:n_valid_split]],
        )

        valid_loss_rolling_average = None
        epoch_times = []

        training_traj_size = train_single_obs.shape[0]
        training_traj_idx = np.arange(training_traj_size, dtype="int32")
        traj_idx = np.tile(training_traj_idx, (self.ensemble_size, 1))

        valid_traj_size = valid_single_obs.shape[0]
        valid_traj_idx = np.tile(
            np.arange(valid_traj_size, dtype="int32"), (self.ensemble_size, 1)
        )

        traj_length = train_single_obs.shape[1]

        # T-MCL assignment data
        ensemble_train_single_obs = train_single_obs[traj_idx]
        ensemble_train_single_obs_next = train_single_obs_next[traj_idx]
        ensemble_train_single_act = train_single_act[traj_idx]
        ensemble_train_single_delta = train_single_delta[traj_idx]
        ensemble_train_single_back_delta = train_single_back_delta[traj_idx]
        ensemble_train_sim_params = train_sim_params[traj_idx]
        ensemble_train_cp_obs = train_cp_obs[traj_idx]
        ensemble_train_cp_act = train_cp_act[traj_idx]

        # Preprocess for training
        (
            train_obs,
            train_act,
            train_delta,
            train_cp_obs,
            train_cp_act,
            train_obs_next,
            train_back_delta,
            train_sim_params,
        ) = self._preprocess_inputs(
            train_obs,
            train_act,
            train_delta,
            train_cp_obs,
            train_cp_act,
            train_obs_next,
            train_back_delta,
            train_sim_params,
            train_future_bool,
        )
        if n_valid_split > 0:
            (
                valid_obs,
                valid_act,
                valid_delta,
                valid_cp_obs,
                valid_cp_act,
                valid_obs_next,
                valid_back_delta,
                validsim_params,
            ) = self._preprocess_inputs(
                valid_obs,
                valid_act,
                valid_delta,
                valid_cp_obs,
                valid_cp_act,
                valid_obs_next,
                valid_back_delta,
                valid_sim_params,
                valid_future_bool,
            )

        # Build dataset for assignments of trajectories
        (
            itr_obses,
            itr_obses_next,
            itr_acts,
            itr_deltas,
            itr_back_deltas,
            itr_sim_params,
            itr_cp_obses,
            itr_cp_acts,
        ) = ([], [], [], [], [], [], [], [])
        for traj_batch_num in range(
            int(np.ceil(traj_idx.shape[-1] / self.traj_batch_size))
        ):
            for sample_batch_num in range(
                int(np.ceil(traj_length / self.segment_size))
            ):
                traj_start = traj_batch_num * self.traj_batch_size
                traj_end = (traj_batch_num + 1) * self.traj_batch_size
                sample_start = sample_batch_num * self.segment_size
                sample_end = (sample_batch_num + 1) * self.segment_size

                obs_segment = ensemble_train_single_obs[
                    :, traj_start:traj_end, sample_start:sample_end
                ]
                obs_next_segment = ensemble_train_single_obs_next[
                    :, traj_start:traj_end, sample_start:sample_end
                ]
                act_segment = ensemble_train_single_act[
                    :, traj_start:traj_end, sample_start:sample_end
                ]
                delta_segment = ensemble_train_single_delta[
                    :, traj_start:traj_end, sample_start:sample_end
                ]
                back_delta_segment = ensemble_train_single_back_delta[
                    :, traj_start:traj_end, sample_start:sample_end
                ]
                sim_params_segment = ensemble_train_sim_params[
                    :, traj_start:traj_end, sample_start:sample_end
                ]
                cp_obs_segment = ensemble_train_cp_obs[
                    :, traj_start:traj_end, sample_start:sample_end
                ]
                cp_act_segment = ensemble_train_cp_act[
                    :, traj_start:traj_end, sample_start:sample_end
                ]

                itr_obses.append(obs_segment)
                itr_obses_next.append(obs_next_segment)
                itr_acts.append(act_segment)
                itr_deltas.append(delta_segment)
                itr_back_deltas.append(back_delta_segment)
                itr_sim_params.append(sim_params_segment)
                itr_cp_obses.append(cp_obs_segment)
                itr_cp_acts.append(cp_act_segment)

        model_assign_dict = {
            model_idx: dict() for model_idx in range(self.ensemble_size)
        }

        train_dataset_size = train_obs.shape[0]
        if self.ensemble_size > 1:
            bootstrap_idx = np.random.randint(
                0, train_dataset_size, size=(self.ensemble_size, train_dataset_size)
            )
        else:
            bootstrap_idx = np.tile(
                np.arange(train_dataset_size, dtype="int32"), (self.ensemble_size, 1)
            )

        valid_dataset_size = valid_obs.shape[0]
        valid_boostrap_idx = np.tile(
            np.arange(valid_dataset_size, dtype="int32"), (self.ensemble_size, 1)
        )

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        """ ------- Looping over training epochs ------- """
        mean_pred_errors = []
        for epoch in range(epochs):
            mse_losses, recon_losses, pred_errors = [], [], []
            t0 = time.time()

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""

            # Head assignment through the dataset
            # and reconstruct train_min_traj_idxs from segmented minibatches
            ensemble_train_min_traj_idxs = []
            ensemble_train_min_traj_back_idxs = []
            segmented_min_traj_idxs = []
            segmented_min_traj_back_idxs = []
            for (
                obs_segment,
                obs_next_segment,
                act_segment,
                delta_segment,
                back_delta_segment,
                sim_params_segment,
                cp_obs_segment,
                cp_act_segment,
            ) in zip(
                itr_obses,
                itr_obses_next,
                itr_acts,
                itr_deltas,
                itr_back_deltas,
                itr_sim_params,
                itr_cp_obses,
                itr_cp_acts,
            ):

                feed_dict = self.get_feed_dict(
                    obs_segment,
                    obs_next_segment,
                    act_segment,
                    delta_segment,
                    back_delta_segment,
                    cp_obs_segment,
                    cp_act_segment,
                    sim_params=sim_params_segment,
                )

                min_traj_idxs, min_traj_back_idxs = sess.run(
                    [self.min_traj_idxs, self.min_traj_back_idxs], feed_dict=feed_dict
                )

                min_traj_idxs = np.tile(
                    min_traj_idxs[:, :, None, None], [1, 1, obs_segment.shape[2], 1]
                )
                min_traj_back_idxs = np.tile(
                    min_traj_back_idxs[:, :, None, None],
                    [1, 1, obs_segment.shape[2], 1],
                )

                segmented_min_traj_idxs.append(min_traj_idxs)
                segmented_min_traj_back_idxs.append(min_traj_back_idxs)
                if len(segmented_min_traj_idxs) == int(
                    np.ceil(traj_length / self.segment_size)
                ):
                    traj_min_traj_idxs = np.concatenate(segmented_min_traj_idxs, axis=2)
                    traj_min_traj_back_idxs = np.concatenate(
                        segmented_min_traj_back_idxs, axis=2
                    )
                    ensemble_train_min_traj_idxs.append(traj_min_traj_idxs)
                    ensemble_train_min_traj_back_idxs.append(traj_min_traj_back_idxs)
                    segmented_min_traj_idxs.clear()
                    segmented_min_traj_back_idxs.clear()

                # save model_assign_dict
                for model_idx in range(self.ensemble_size):
                    for head_idx, sim_param in zip(
                        min_traj_idxs[model_idx, :, 0, 0],
                        sim_params_segment[model_idx, :, 0, 0],
                    ):
                        sim_param = str(sim_param)
                        if sim_param not in model_assign_dict[model_idx]:
                            model_assign_dict[model_idx][sim_param] = []
                        model_assign_dict[model_idx][sim_param].append(head_idx)

            ensemble_train_min_traj_idxs = np.concatenate(
                ensemble_train_min_traj_idxs, axis=1
            )
            ensemble_train_min_traj_back_idxs = np.concatenate(
                ensemble_train_min_traj_back_idxs, axis=1
            )

            assert (
                np.shape(ensemble_train_min_traj_idxs)[0]
                == ensemble_train_single_obs.shape[0]
                and np.shape(ensemble_train_min_traj_idxs)[1]
                == ensemble_train_single_obs.shape[1]
                and np.shape(ensemble_train_min_traj_idxs)[2]
                == ensemble_train_single_obs.shape[2]
            )

            (
                ensemble_train_min_traj_idxs,
                ensemble_train_min_traj_back_idxs,
            ) = self._preprocess_min_idxs(
                ensemble_train_min_traj_idxs,
                ensemble_train_min_traj_back_idxs,
                train_future_bool,
            )

            """ Training dynamics model """
            mse_losses, back_mse_losses, pred_errors, recon_losses = [], [], [], []
            bootstrap_idx = shuffle_rows(bootstrap_idx)

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""
            for batch_num in range(
                int(np.ceil(bootstrap_idx.shape[-1] / self.sample_batch_size))
            ):
                batch_idxs = bootstrap_idx[
                    :,
                    batch_num
                    * self.sample_batch_size : (batch_num + 1)
                    * self.sample_batch_size,
                ]

                effective_batch_size = batch_idxs.shape[1]

                bootstrap_train_obs = train_obs[batch_idxs].reshape(
                    self.ensemble_size, effective_batch_size, 1, self.obs_space_dims
                )
                bootstrap_train_act = train_act[batch_idxs].reshape(
                    self.ensemble_size, effective_batch_size, 1, self.action_space_dims
                )
                bootstrap_train_delta = train_delta[batch_idxs].reshape(
                    self.ensemble_size, effective_batch_size, 1, self.obs_space_dims
                )
                bootstrap_train_obs_next = train_obs_next[batch_idxs].reshape(
                    self.ensemble_size, effective_batch_size, 1, self.obs_space_dims
                )
                bootstrap_train_back_delta = train_back_delta[batch_idxs].reshape(
                    self.ensemble_size, effective_batch_size, 1, self.obs_space_dims
                )
                bootstrap_train_cp_obs = train_cp_obs[batch_idxs].reshape(
                    self.ensemble_size,
                    effective_batch_size,
                    1,
                    self.obs_space_dims * self.history_length,
                )
                bootstrap_train_cp_act = train_cp_act[batch_idxs].reshape(
                    self.ensemble_size,
                    effective_batch_size,
                    1,
                    self.action_space_dims * self.history_length,
                )
                bootstrap_train_sim_params = train_sim_params[batch_idxs].reshape(
                    self.ensemble_size,
                    effective_batch_size,
                    1,
                    self.simulation_param_dim,
                )

                bootstrap_min_traj_idxs = np.stack(
                    [
                        ensemble_train_min_traj_idxs[model_idx][batch_idxs[model_idx]]
                        for model_idx in range(self.ensemble_size)
                    ]
                ).reshape(self.ensemble_size, effective_batch_size)
                bootstrap_min_traj_back_idxs = np.stack(
                    [
                        ensemble_train_min_traj_back_idxs[model_idx][
                            batch_idxs[model_idx]
                        ]
                        for model_idx in range(self.ensemble_size)
                    ]
                ).reshape(self.ensemble_size, effective_batch_size)

                feed_dict = self.get_feed_dict(
                    bootstrap_train_obs,
                    bootstrap_train_obs_next,
                    bootstrap_train_act,
                    bootstrap_train_delta,
                    bootstrap_train_back_delta,
                    bootstrap_train_cp_obs,
                    bootstrap_train_cp_act,
                    sim_params=bootstrap_train_sim_params,
                    min_traj_idxs=bootstrap_min_traj_idxs,
                    min_traj_back_idxs=bootstrap_min_traj_back_idxs,
                )

                if itr < self.ie_itrs and self.use_ie:
                    mse_loss, back_mse_loss, recon_loss, pred_error, _ = sess.run(
                        [
                            self.ie_mse_loss,
                            self.ie_back_mse_loss,
                            self.ie_recon_loss,
                            self.norm_pred_error,
                            self.ie_train_op,
                        ],
                        feed_dict=feed_dict,
                    )
                else:
                    mse_loss, back_mse_loss, recon_loss, pred_error, _ = sess.run(
                        [
                            self.mse_loss,
                            self.back_mse_loss,
                            self.recon_loss,
                            self.norm_pred_error,
                            self.train_op,
                        ],
                        feed_dict=feed_dict,
                    )

                mse_losses.append(mse_loss)
                back_mse_losses.append(back_mse_loss)
                recon_losses.append(recon_loss)
                pred_errors.append(pred_error)

            """ ------- Validation -------"""
            if n_valid_split > 0:
                raise NotImplementedError
            else:
                if verbose:
                    logger.log(
                        "Training DynamicsModel - finished epoch %i --"
                        "[Training] mse loss: %.4f  back mse loss: %.4f  recon loss: %.4f  pred error: %.4f  epoch time: %.2f"
                        % (
                            epoch,
                            np.mean(mse_losses),
                            np.mean(back_mse_losses),
                            np.mean(recon_losses),
                            np.mean(pred_errors),
                            time.time() - t0,
                        )
                    )
            mean_pred_errors.append(np.mean(pred_errors))

        # save model_assign_dict
        checkdir = osp.join(logger.get_dir(), "saved_assign_dicts")
        os.makedirs(checkdir, exist_ok=True)
        save_path = osp.join(checkdir, "assign_epoch_{}".format(itr))
        joblib.dump(model_assign_dict, save_path)

        """ ------- Tabular Logging ------- """
        if log_tabular:
            logger.logkv("AvgModelEpochTime", np.mean(epoch_times))
            logger.logkv("Epochs", epoch)
            logger.logkv("PredictionError", np.mean(mean_pred_errors))
        elif log_only_pred_error:
            logger.logkv("PredictionError", np.mean(mean_pred_errors))

    def save(self, save_path):
        sess = tf.get_default_session()
        ps = sess.run(self.params)
        joblib.dump(ps, save_path)
        if self.normalization is not None:
            norm_save_path = save_path + "_norm_stats"
            joblib.dump(self.normalization, norm_save_path)

    def load(self, load_path):
        sess = tf.get_default_session()
        loaded_params = joblib.load(load_path)
        restores = []
        for p, loaded_p in zip(self.params, loaded_params):
            restores.append(p.assign(loaded_p))
        sess.run(restores)
        if self.normalize_input:
            norm_save_path = load_path + "_norm_stats"
            self.normalization = joblib.load(norm_save_path)

    def compute_normalization(self, obs, act, delta, cp_obs, cp_act, back_delta):
        assert obs.shape[0] == delta.shape[0] == act.shape[0]
        proc_obs = self.env.obs_preproc(obs)

        # store means and std in dict
        self.normalization = OrderedDict()
        self.normalization["obs"] = (
            np.mean(proc_obs, axis=(0, 1)),
            np.std(proc_obs, axis=(0, 1)),
        )
        self.normalization["delta"] = (
            np.mean(delta, axis=(0, 1)),
            np.std(delta, axis=(0, 1)),
        )
        self.normalization["act"] = (
            np.mean(act, axis=(0, 1)),
            np.std(act, axis=(0, 1)),
        )
        self.normalization["cp_obs"] = (
            np.mean(cp_obs, axis=(0, 1)),
            np.std(cp_obs, axis=(0, 1)),
        )
        self.normalization["cp_act"] = (
            np.mean(cp_act, axis=(0, 1)),
            np.std(cp_act, axis=(0, 1)),
        )
        self.normalization["back_delta"] = (
            np.mean(back_delta, axis=(0, 1)),
            np.std(back_delta, axis=(0, 1)),
        )

    def get_normalization_stats(self):
        if self.normalize_input:
            norm_obs_mean = self.normalization["obs"][0]
            norm_obs_std = self.normalization["obs"][1]
            norm_delta_mean = self.normalization["delta"][0]
            norm_delta_std = self.normalization["delta"][1]
            if self.discrete:
                norm_act_mean = np.zeros((self.action_space_dims,))
                norm_act_std = np.ones((self.action_space_dims,))
            else:
                norm_act_mean = self.normalization["act"][0]
                norm_act_std = self.normalization["act"][1]
            if self.state_diff:
                norm_cp_obs_mean = np.zeros(
                    (self.obs_space_dims * self.history_length,)
                )
                norm_cp_obs_std = np.ones((self.obs_space_dims * self.history_length,))
            else:
                norm_cp_obs_mean = self.normalization["cp_obs"][0]
                norm_cp_obs_std = self.normalization["cp_obs"][1]
            if self.discrete:
                norm_cp_act_mean = np.zeros(
                    (self.action_space_dims * self.history_length,)
                )
                norm_cp_act_std = np.ones(
                    (self.action_space_dims * self.history_length,)
                )
            else:
                norm_cp_act_mean = self.normalization["cp_act"][0]
                norm_cp_act_std = self.normalization["cp_act"][1]
            norm_back_delta_mean = self.normalization["back_delta"][0]
            norm_back_delta_std = self.normalization["back_delta"][1]
        else:
            norm_obs_mean = np.zeros((self.proc_obs_space_dims,))
            norm_obs_std = np.ones((self.proc_obs_space_dims,))
            norm_act_mean = np.zeros((self.action_space_dims,))
            norm_act_std = np.ones((self.action_space_dims,))
            norm_delta_mean = np.zeros((self.obs_space_dims,))
            norm_delta_std = np.ones((self.obs_space_dims,))
            norm_cp_obs_mean = np.zeros((self.obs_space_dims * self.history_length,))
            norm_cp_obs_std = np.ones((self.obs_space_dims * self.history_length,))
            norm_cp_act_mean = np.zeros((self.action_space_dims * self.history_length,))
            norm_cp_act_std = np.ones((self.action_space_dims * self.history_length,))
            norm_back_delta_mean = np.zeros((self.obs_space_dims,))
            norm_back_delta_std = np.ones((self.obs_space_dims,))

        return (
            norm_obs_mean,
            norm_obs_std,
            norm_act_mean,
            norm_act_std,
            norm_delta_mean,
            norm_delta_std,
            norm_cp_obs_mean,
            norm_cp_obs_std,
            norm_cp_act_mean,
            norm_cp_act_std,
            norm_back_delta_mean,
            norm_back_delta_std,
        )

    def get_feed_dict(
        self,
        obs,
        obs_next,
        act,
        delta,
        back_delta,
        cp_obs,
        cp_act,
        sim_params=None,
        min_traj_idxs=None,
        min_traj_back_idxs=None,
    ):
        (
            norm_obs_mean,
            norm_obs_std,
            norm_act_mean,
            norm_act_std,
            norm_delta_mean,
            norm_delta_std,
            norm_cp_obs_mean,
            norm_cp_obs_std,
            norm_cp_act_mean,
            norm_cp_act_std,
            norm_back_delta_mean,
            norm_back_delta_std,
        ) = self.get_normalization_stats()

        feed_dict = {
            self.bs_obs_ph: obs,
            self.bs_act_ph: act,
            self.bs_delta_ph: delta,
            self.bs_obs_next_ph: obs_next,
            self.bs_back_delta_ph: back_delta,
            self.bs_cp_obs_ph: cp_obs,
            self.bs_cp_act_ph: cp_act,
            self.norm_obs_mean_ph: norm_obs_mean,
            self.norm_obs_std_ph: norm_obs_std,
            self.norm_act_mean_ph: norm_act_mean,
            self.norm_act_std_ph: norm_act_std,
            self.norm_delta_mean_ph: norm_delta_mean,
            self.norm_delta_std_ph: norm_delta_std,
            self.norm_cp_obs_mean_ph: norm_cp_obs_mean,
            self.norm_cp_obs_std_ph: norm_cp_obs_std,
            self.norm_cp_act_mean_ph: norm_cp_act_mean,
            self.norm_cp_act_std_ph: norm_cp_act_std,
            self.norm_back_delta_mean_ph: norm_back_delta_mean,
            self.norm_back_delta_std_ph: norm_back_delta_std,
        }
        if min_traj_idxs is not None:
            feed_dict[self.min_traj_idxs_ph] = min_traj_idxs
        if min_traj_back_idxs is not None:
            feed_dict[self.min_traj_back_idxs_ph] = min_traj_back_idxs
        if sim_params is not None:
            feed_dict[self.bs_simulation_param_ph] = sim_params
        return feed_dict

    def _preprocess_inputs(
        self,
        obs,
        act,
        delta,
        cp_obs,
        cp_act,
        obs_next,
        back_delta,
        sim_param,
        future_bool,
    ):
        dataset_size = obs.shape[0]

        _future_bool = future_bool.reshape((dataset_size, -1))
        _obs = obs.reshape((dataset_size, -1, self.obs_space_dims))
        _act = act.reshape((dataset_size, -1, self.action_space_dims))
        _delta = delta.reshape((dataset_size, -1, self.obs_space_dims))
        _obs_next = obs_next.reshape((dataset_size, -1, self.obs_space_dims))
        _back_delta = back_delta.reshape((dataset_size, -1, self.obs_space_dims))

        _cp_obs = np.tile(cp_obs, (1, 1, self.future_length))
        _cp_obs = _cp_obs.reshape(
            (dataset_size, -1, self.obs_space_dims * self.history_length)
        )
        _cp_act = np.tile(cp_act, (1, 1, self.future_length))
        _cp_act = _cp_act.reshape(
            (dataset_size, -1, self.action_space_dims * self.history_length)
        )

        sim_param_dim = sim_param.shape[2]
        _sim_param = np.tile(sim_param, (1, 1, self.future_length))
        _sim_param = _sim_param.reshape((dataset_size, -1, sim_param_dim))

        _obs = _obs[_future_bool > 0, :]
        _act = _act[_future_bool > 0, :]
        _delta = _delta[_future_bool > 0, :]
        _obs_next = _obs_next[_future_bool > 0, :]
        _back_delta = _back_delta[_future_bool > 0, :]
        _cp_obs = _cp_obs[_future_bool > 0, :]
        _cp_act = _cp_act[_future_bool > 0, :]
        _sim_param = _sim_param[_future_bool > 0, :]

        return _obs, _act, _delta, _cp_obs, _cp_act, _obs_next, _back_delta, _sim_param

    def _preprocess_min_idxs(self, min_idxs, min_back_idxs, future_bool):
        ensemble_size = min_idxs.shape[0]
        dataset_size = min_idxs.shape[1]

        _future_bool = future_bool.reshape((dataset_size, -1))
        _min_idxs = np.tile(min_idxs, (1, 1, 1, self.future_length))
        _min_idxs = _min_idxs.reshape((ensemble_size, dataset_size, -1, 1))
        _min_back_idxs = np.tile(min_back_idxs, (1, 1, 1, self.future_length))
        _min_back_idxs = _min_back_idxs.reshape((ensemble_size, dataset_size, -1, 1))

        _min_idxs = _min_idxs[:, _future_bool > 0, :]
        _min_back_idxs = _min_back_idxs[:, _future_bool > 0, :]
        return _min_idxs, _min_back_idxs


def normalize(data_array, mean, std):
    return (data_array - mean) / (std + 1e-10)


def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean
