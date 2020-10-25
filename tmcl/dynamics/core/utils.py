import tensorflow as tf
import numpy as np
from baselines.common.distributions import make_pdtype

def create_mcl_cadm_multiheaded_mlp(
    output_dim,
    hidden_sizes,
    hidden_nonlinearity,
    output_nonlinearity,
    input_obs_dim=None,
    input_act_dim=None,
    input_obs_var=None,
    input_act_var=None,
    input_history_obs_var=None,
    input_history_act_var=None,
    input_history_delta_var=None,
    n_forwards=1,
    ensemble_size=5,
    head_size=3,
    weight_decays=None,
    reward_fn=None,
    n_candidates=None,
    norm_obs_mean_var=None,
    norm_obs_std_var=None,
    norm_act_mean_var=None,
    norm_act_std_var=None,
    norm_delta_mean_var=None,
    norm_delta_std_var=None,
    norm_cp_obs_mean_var=None,
    norm_cp_obs_std_var=None,
    norm_cp_act_mean_var=None,
    norm_cp_act_std_var=None,
    norm_back_delta_mean_var=None,
    norm_back_delta_std_var=None,
    n_particles=20,
    bs_input_obs_var=None,
    bs_input_act_var=None,
    discrete=False,
    cem_init_mean_var=None,
    cem_init_var_var=None,
    obs_preproc_fn=None,
    obs_postproc_fn=None,
    deterministic=False,
    reuse=False,
    use_simulation_param=False,
    simulation_param_var=None,
    simulation_param_dim=None,
    bs_input_sim_param_var=None,
    sep_layer_size=0,
    input_cp_obs_var=None,
    input_cp_act_var=None,
    cp_output_dim=None,
    cp_forward=None,
    bs_input_cp_var=None,
    build_policy_graph=False,
    non_adaptive_planning=False,
):

    proc_obs_dim = int(obs_preproc_fn(bs_input_obs_var).shape[-1])
    if bs_input_cp_var is None:
        cp_output_dim = 0
    if use_simulation_param:
        hidden_sizes = [proc_obs_dim + input_act_dim + simulation_param_dim] + list(
            hidden_sizes
        )
    else:
        hidden_sizes = [proc_obs_dim + input_act_dim] + list(hidden_sizes)

    dense_layer = create_dense_layer

    layers = []
    l2_regs = []
    for idx in range(len(hidden_sizes) - 1 - sep_layer_size):
        layer, l2_reg = dense_layer(
            name="hidden_%d" % idx,
            ensemble_size=ensemble_size,
            input_dim=hidden_sizes[idx],
            output_dim=hidden_sizes[idx + 1],
            activation=hidden_nonlinearity,
            weight_decay=weight_decays[idx],
        )
        layers.append(layer)
        l2_regs.append(l2_reg)

    head_linear_layers = []
    head_mu_layers = []
    head_logvar_layers = []
    for head_idx in range(head_size):
        head_linear_layer = []
        for idx in range(len(hidden_sizes) - 1 - sep_layer_size, len(hidden_sizes) - 1):
            if sep_layer_size != 0 and idx == len(hidden_sizes) - 1 - sep_layer_size:
                input_dim = hidden_sizes[idx] + cp_output_dim
            else:
                input_dim = hidden_sizes[idx]
            layer, l2_reg = dense_layer(
                name="hidden_{}_head{}".format(idx, head_idx),
                ensemble_size=ensemble_size,
                input_dim=input_dim,
                output_dim=hidden_sizes[idx + 1],
                activation=hidden_nonlinearity,
                weight_decay=weight_decays[idx],
            )
            head_linear_layer.append(layer)
            l2_regs.append(l2_reg)

        if sep_layer_size == 0:
            input_dim = hidden_sizes[-1] + cp_output_dim
        else:
            input_dim = hidden_sizes[-1]
        mu_layer, mu_l2_reg = dense_layer(
            name="output_head{}_mu".format(head_idx),
            ensemble_size=ensemble_size,
            input_dim=input_dim,
            output_dim=output_dim,
            activation=output_nonlinearity,
            weight_decay=weight_decays[-1],
        )
        logvar_layer, logvar_l2_reg = dense_layer(
            name="output_head{}_logvar".format(head_idx),
            ensemble_size=ensemble_size,
            input_dim=input_dim,
            output_dim=output_dim,
            activation=output_nonlinearity,
            weight_decay=weight_decays[-1],
        )
        head_linear_layers.append(head_linear_layer)
        head_mu_layers.append(mu_layer)
        head_logvar_layers.append(logvar_layer)
        l2_regs += [mu_l2_reg, logvar_l2_reg]

    max_logvar = tf.Variable(
        np.ones([head_size, 1, 1, output_dim]) / 2.0,
        dtype=tf.float32,
        name="max_log_var",
    )
    min_logvar = tf.Variable(
        -np.ones([head_size, 1, 1, output_dim]) * 10,
        dtype=tf.float32,
        name="min_log_var",
    )

    def forward(xx, context=None):
        for layer in layers:
            xx = layer(xx)

        embedding = xx

        head_mu_list = []
        head_logvar_list = []
        for i, (mu_layer, logvar_layer) in enumerate(
            zip(head_mu_layers, head_logvar_layers)
        ):
            if context is not None:
                head_xx = tf.concat([xx, context[i]], axis=2)
            else:
                head_xx = xx
            for h_linear_layer in head_linear_layers[i]:
                head_xx = h_linear_layer(head_xx)

            head_mu = mu_layer(head_xx)
            head_logvar = logvar_layer(head_xx)

            head_mu_list.append(head_mu)
            head_logvar_list.append(head_logvar)

        head_mu = tf.stack(head_mu_list)
        head_logvar = tf.stack(head_logvar_list)

        if norm_delta_mean_var is not None:
            norm_mean = norm_delta_mean_var
            norm_std = norm_delta_std_var
        else:
            norm_mean = norm_back_delta_mean_var
            norm_std = norm_back_delta_std_var

        denormalized_head_mu = denormalize(head_mu, norm_mean, norm_std)

        if deterministic:
            xx = denormalized_head_mu
        else:
            head_logvar = max_logvar - tf.nn.softplus(max_logvar - head_logvar)
            head_logvar = min_logvar + tf.nn.softplus(head_logvar - min_logvar)

            denormalized_head_logvar = head_logvar + 2 * tf.log(norm_std)
            denormalized_head_std = tf.exp(denormalized_head_logvar / 2.0)

            xx = (
                denormalized_head_mu
                + tf.random.normal(tf.shape(denormalized_head_mu))
                * denormalized_head_std
            )

        return xx, head_mu, head_logvar, embedding

    bs_input_proc_obs_var = obs_preproc_fn(bs_input_obs_var)
    bs_normalized_input_obs = normalize(
        bs_input_proc_obs_var, norm_obs_mean_var, norm_obs_std_var
    )
    bs_normalized_input_act = normalize(
        bs_input_act_var, norm_act_mean_var, norm_act_std_var
    )

    if use_simulation_param:
        x = tf.concat(
            [bs_normalized_input_obs, bs_normalized_input_act, bs_input_sim_param_var],
            2,
        )
    else:
        x = tf.concat([bs_normalized_input_obs, bs_normalized_input_act], 2)
    output_var, mu, logvar, embedding = forward(x, bs_input_cp_var)

    """build inference graph for gpu inference"""
    """episodic trajectory sampling(TS-inf) will be used"""

    n = n_candidates
    p = n_particles
    m = tf.shape(input_obs_var)[0]
    h = n_forwards
    obs_dim = input_obs_dim
    act_dim = input_act_dim

    num_elites = 50
    num_cem_iters = 5
    alpha = 0.1

    lower_bound = -1.0
    upper_bound = 1.0

    if build_policy_graph:
        if non_adaptive_planning:
            print("=" * 80)
            print("Non-Adaptive Planning")
            print("=" * 80)
        else:
            print("=" * 80)
            print("Adaptive Planning")
            print("=" * 80)

        if bs_input_cp_var is not None:
            bs_input_cp_obs_var = tf.tile(
                input_cp_obs_var[None, :, :], (ensemble_size, 1, 1)
            )  # (ensemble_size, m, obs_dim*history_length)
            bs_input_cp_act_var = tf.tile(
                input_cp_act_var[None, :, :], (ensemble_size, 1, 1)
            )  # (ensemble_size, m, act_dim*history_length)
            bs_normalized_input_cp_obs = normalize(
                bs_input_cp_obs_var, norm_cp_obs_mean_var, norm_cp_obs_std_var
            )
            bs_normalized_input_cp_act = normalize(
                bs_input_cp_act_var, norm_cp_act_mean_var, norm_cp_act_std_var
            )
            bs_normalized_cp_x = tf.concat(
                [bs_normalized_input_cp_obs, bs_normalized_input_cp_act], axis=-1
            )
            bs_input_cp_var = cp_forward(
                bs_normalized_cp_x, inference=True
            )  # (head_size, ensemble_size, m, cp_output_dim)
            inference_cp_var = bs_input_cp_var
        else:
            bs_input_cp_var = None
            inference_cp_var = None

        history_length = tf.shape(input_history_obs_var)[1]

        def select_best_head():
            pre_obs = tf.tile(
                tf.reshape(input_history_obs_var, [m, history_length, 1, obs_dim]),
                [1, 1, p * ensemble_size, 1],
            )  # (m, history_length, p * ensemble_size, obs_dim)
            proc_pre_obs = obs_preproc_fn(pre_obs)
            normalized_proc_pre_obs = normalize(
                proc_pre_obs, norm_obs_mean_var, norm_obs_std_var
            )
            normalized_proc_pre_obs = tf.reshape(
                tf.transpose(normalized_proc_pre_obs, [2, 0, 1, 3]),
                [ensemble_size, p, m, history_length, proc_obs_dim],
            )

            pre_act = tf.tile(
                tf.reshape(input_history_act_var, [m, history_length, 1, act_dim]),
                [1, 1, p * ensemble_size, 1],
            )  # (m, history_length, p * ensemble_size, act_dim)
            normalized_pre_act = normalize(pre_act, norm_act_mean_var, norm_act_std_var)
            normalized_pre_act = tf.reshape(
                tf.transpose(normalized_pre_act, [2, 0, 1, 3]),
                [ensemble_size, p, m, history_length, act_dim],
            )

            x = tf.concat([normalized_proc_pre_obs, normalized_pre_act], 4)
            x = tf.reshape(
                x, [ensemble_size, p * m * history_length, proc_obs_dim + act_dim]
            )

            if use_simulation_param:
                # simulation param var: [m, simulation_param_dim]
                simulation_param = tf.tile(
                    tf.reshape(simulation_param_var, [m, 1, 1, simulation_param_dim]),
                    [1, history_length, p * ensemble_size, 1],
                )
                simulation_param = tf.transpose(simulation_param, [2, 0, 1, 3])
                simulation_param = tf.reshape(
                    simulation_param,
                    [ensemble_size, p * m * history_length, simulation_param_dim],
                )
                x = tf.concat([x, simulation_param], axis=-1)

            if bs_input_cp_var is not None:
                reshaped_context = tf.transpose(
                    bs_input_cp_var, [0, 2, 1, 3]
                )  # [head_size, m, ensemble_size, cp_output_dim]
                reshaped_context = reshaped_context[
                    :, :, None, :, None, :
                ]  # [head_size, m, 1, ensemble_size, cp_output_dim]
                reshaped_context = tf.tile(
                    reshaped_context, [1, 1, history_length, 1, p, 1]
                )  # [head_size, m, history_length, ensemble_size, p, cp_output_dim]
                reshaped_context = tf.transpose(
                    reshaped_context, [0, 3, 4, 1, 2, 5]
                )  # [head_size, ensemble_size, p, m, history_length, cp_output_dim]
                reshaped_context = tf.reshape(
                    reshaped_context,
                    [head_size, ensemble_size, p * m * history_length, cp_output_dim],
                )
            else:
                reshaped_context = None

            delta_prediction, *_ = forward(
                x, reshaped_context
            )  # [head_size, ensemble_size, p * m * history_length, obs_dim]
            delta_prediction = tf.reshape(
                delta_prediction,
                [head_size, ensemble_size, p, m, history_length, obs_dim],
            )

            prediction_error = tf.reduce_mean(
                tf.square(
                    delta_prediction
                    - tf.reshape(
                        input_history_delta_var, [1, 1, 1, m, history_length, obs_dim]
                    )
                ),
                axis=[2, 4, 5],
            )  # [head_size, ensemble_size, m]

            prediction_error = tf.transpose(
                prediction_error, [1, 2, 0]
            )  # [ensemble_size, m, head_size]
            best_head_idx = tf.nn.top_k(-1.0 * prediction_error)[
                1
            ]  # [ensemble_size, m, 1]
            best_head_idx = tf.reshape(best_head_idx, [ensemble_size, m])
            return best_head_idx

        def select_random_head():
            random_head_idx = tf.random_uniform(
                [ensemble_size, m], maxval=head_size, dtype=tf.int32
            )
            return random_head_idx

        best_head_idx = tf.cond(
            tf.math.not_equal(history_length, 0), select_best_head, select_random_head
        )  # [ensemble_size, m]

        best_head_idx = tf.reshape(best_head_idx, [-1])  # [ensemble_size * m]
        best_head_idx = tf.transpose(
            tf.stack([tf.range(tf.shape(best_head_idx)[0]), best_head_idx])
        )  # [ensemble_size * m, 2]

        if cem_init_mean_var is not None:
            ############################
            ### CROSS ENTROPY METHOD ###
            ############################
            print("=" * 80)
            print("CROSS ENTROPY METHOD")
            print("=" * 80)
            mean = cem_init_mean_var  # (m, h, act_dim)
            var = cem_init_var_var

            # input_obs_var: (m, obs_dim)

            for _ in range(num_cem_iters):
                lb_dist, ub_dist = mean - lower_bound, upper_bound - mean
                constrained_var = tf.minimum(
                    tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var
                )
                repeated_mean = tf.tile(
                    mean[:, None, :, :], [1, n, 1, 1]
                )  # (m, n, h, act_dim)
                repeated_var = tf.tile(
                    constrained_var[:, None, :, :], [1, n, 1, 1]
                )  # (m, n, h, act_dim)
                actions = tf.truncated_normal(
                    [m, n, h, act_dim], repeated_mean, tf.sqrt(repeated_var)
                )

                returns = 0
                observation = tf.tile(
                    tf.reshape(input_obs_var, [m, 1, 1, obs_dim]), [1, n, p, 1]
                )  # (m, n, p, obs_dim)
                if bs_input_cp_var is not None:
                    reshaped_context = tf.transpose(
                        bs_input_cp_var, [0, 2, 1, 3]
                    )  # [head_size, m, ensemble_size, cp_output_dim]
                    reshaped_context = reshaped_context[
                        :, :, None, :, None, :
                    ]  # [head_size, m, 1, ensemble_size, 1, cp_output_dim]
                    reshaped_context = tf.tile(
                        reshaped_context, [1, 1, n, 1, int(p / ensemble_size), 1]
                    )  # [head_size, m, n, ensemble_size, p/ensemble_size, cp_output_dim]
                    reshaped_context = tf.transpose(
                        reshaped_context, [0, 3, 4, 1, 2, 5]
                    )  # [head_size, ensemble_size, p/ensemble_size, m, n, cp_output_dim]
                    reshaped_context = tf.reshape(
                        reshaped_context,
                        [
                            head_size,
                            ensemble_size,
                            int(p / ensemble_size) * m * n,
                            cp_output_dim,
                        ],
                    )
                else:
                    reshaped_context = None

                for t in range(h):
                    action = actions[:, :, t]  # [m, n, act_dim]
                    normalized_act = normalize(
                        action, norm_act_mean_var, norm_act_std_var
                    )  # [m, n, act_dim]
                    normalized_act = tf.tile(
                        normalized_act[:, :, None, :], [1, 1, p, 1]
                    )  # [m, n, p, act_dim]
                    normalized_act = tf.reshape(
                        tf.transpose(normalized_act, [2, 0, 1, 3]),
                        [ensemble_size, int(p / ensemble_size) * m * n, act_dim],
                    )  # [ensemble_size, p/ensemble_size * m * n, act_dim]

                    proc_observation = obs_preproc_fn(observation)
                    normalized_proc_obs = normalize(
                        proc_observation, norm_obs_mean_var, norm_obs_std_var
                    )  # [m, n, p, proc_obs_dim]
                    normalized_proc_obs = tf.reshape(
                        tf.transpose(normalized_proc_obs, [2, 0, 1, 3]),
                        [ensemble_size, int(p / ensemble_size) * m * n, proc_obs_dim],
                    )  # [ensemble_size, p/ensemble_size * m * n, proc_obs_dim ]

                    x = tf.concat(
                        [normalized_proc_obs, normalized_act], 2
                    )  # [ensemble_size, p/ensemble_size * m * n, proc_obs_dim + act_dim]

                    if use_simulation_param:
                        # simulation param var: [m, simulation_param_dim]
                        simulation_param = tf.tile(
                            tf.reshape(
                                simulation_param_var, [m, 1, 1, simulation_param_dim]
                            ),
                            [1, n, p, 1],
                        )  # [m, n, p, sim_dim]
                        simulation_param = tf.transpose(simulation_param, [2, 0, 1, 3])
                        simulation_param = tf.reshape(
                            simulation_param,
                            [
                                ensemble_size,
                                int(p / ensemble_size) * m * n,
                                simulation_param_dim,
                            ],
                        )
                        x = tf.concat([x, simulation_param], axis=2)

                    delta, *_ = forward(
                        x, reshaped_context
                    )  # [head_size, ensemble_size, int(p/ensemble_size) * m * n, obs_dim]
                    delta = tf.transpose(
                        tf.reshape(
                            delta,
                            [
                                head_size,
                                ensemble_size,
                                int(p / ensemble_size),
                                m,
                                n,
                                obs_dim,
                            ],
                        ),
                        [1, 3, 0, 2, 4, 5],
                    )  # [ensemble_size, m, head_size, int(p/ensemble_size), n, obs_dim]
                    delta = tf.reshape(
                        delta,
                        [
                            ensemble_size * m,
                            head_size,
                            int(p / ensemble_size),
                            n,
                            obs_dim,
                        ],
                    )

                    if non_adaptive_planning:
                        delta = tf.reduce_mean(
                            delta, axis=1
                        )  # [ensemble_size * m, int(p/ensemble_size), n, obs_dim]
                    else:
                        delta = tf.gather_nd(
                            delta, best_head_idx
                        )  # [ensemble_size * m, int(p/ensemble_size), n, obs_dim]

                    delta = tf.transpose(
                        tf.reshape(
                            delta,
                            [ensemble_size, m, int(p / ensemble_size), n, obs_dim],
                        ),
                        [1, 3, 0, 2, 4],
                    )  # [m, n, ensemble_size, int(p/ensemble_size), obs_dim]
                    delta = tf.reshape(delta, [m, n, p, obs_dim])

                    next_observation = obs_postproc_fn(
                        observation, delta
                    )  # [m, n, p, obs_dim]
                    repeated_action = tf.tile(action[:, :, None, :], [1, 1, p, 1])
                    reward = reward_fn(observation, repeated_action, next_observation)

                    returns += reward  # [m, n, p]
                    observation = next_observation

                returns = tf.reduce_mean(returns, axis=2)
                _, elites_idx = tf.nn.top_k(
                    returns, k=num_elites, sorted=True
                )  # [m, num_elites]
                elites_idx += tf.range(0, m * n, n)[:, None]
                flat_elites_idx = tf.reshape(
                    elites_idx, [m * num_elites]
                )  # [m * num_elites]
                flat_actions = tf.reshape(actions, [m * n, h, act_dim])
                flat_elites = tf.gather(
                    flat_actions, flat_elites_idx
                )  # [m * num_elites, h, act_dim]
                elites = tf.reshape(flat_elites, [m, num_elites, h, act_dim])

                new_mean = tf.reduce_mean(elites, axis=1)  # [m, h, act_dim]
                new_var = tf.reduce_mean(
                    tf.square(elites - new_mean[:, None, :, :]), axis=1
                )

                mean = mean * alpha + (1 - alpha) * new_mean
                var = var * alpha + (1 - alpha) * new_var

            optimal_action_var = mean

        else:
            #######################
            ### RANDOM SHOOTING ###
            #######################
            print("=" * 80)
            print("RANDOM SHOOTING")
            print("=" * 80)
            raise NotImplementedError
    else:
        optimal_action_var = None
        inference_cp_var = None

    return (
        input_obs_var,
        input_act_var,
        output_var,
        optimal_action_var,
        mu,
        logvar,
        max_logvar,
        min_logvar,
        l2_regs,
        embedding,
        inference_cp_var,
    )


def create_ensemble_multiheaded_context_predictor(
    context_hidden_sizes,
    context_hidden_nonlinearity,
    output_nonlinearity,
    ensemble_size=None,
    cp_input_dim=None,
    context_weight_decays=None,
    bs_input_cp_obs_var=None,
    bs_input_cp_act_var=None,
    norm_cp_obs_mean_var=None,
    norm_cp_obs_std_var=None,
    norm_cp_act_mean_var=None,
    norm_cp_act_std_var=None,
    cp_output_dim=0,
    head_size=3,
    use_global_head=False,
    reuse=False,
):
    #################################
    ### CONTINUOUS CONTEXT VECTOR ###
    #################################
    print("=" * 80)
    print("CONTINUOUS CONTEXT VECTOR")
    print("=" * 80)

    context_hidden_sizes = [cp_input_dim] + list(context_hidden_sizes)

    layers = []
    l2_regs = []
    for idx in range(len(context_hidden_sizes) - 1):
        layer, l2_reg = create_dense_layer(
            name="cp_hidden_%d" % idx,
            ensemble_size=ensemble_size,
            input_dim=context_hidden_sizes[idx],
            output_dim=context_hidden_sizes[idx + 1],
            activation=context_hidden_nonlinearity,
            weight_decay=context_weight_decays[idx],
        )
        layers.append(layer)
        l2_regs.append(l2_reg)

    if not use_global_head:
        print("=" * 80)
        print("MULTI-HEADED CONTEXT ENCODER")
        print("=" * 80)
        head_layers = []
        for head_idx in range(head_size):
            head_layer, head_l2_reg = create_dense_layer(
                name="cp_output_head_{}".format(head_idx),
                ensemble_size=ensemble_size,
                input_dim=context_hidden_sizes[-1],
                output_dim=cp_output_dim,
                activation=output_nonlinearity,
                weight_decay=context_weight_decays[-1],
            )
            head_layers += [head_layer]
            l2_regs += [l2_reg]

        def forward(xx, inference=False):
            for layer in layers:
                xx = layer(xx)

            output_heads = []
            for head_layer in head_layers:
                head_xx = head_layer(xx)
                output_heads.append(head_xx)

            output = tf.stack(output_heads)
            return output

    else:
        print("=" * 80)
        print("GLOBAL CONTEXT ENCODER")
        print("=" * 80)
        head_layer, head_l2_reg = create_dense_layer(
            name="cp_output_head",
            ensemble_size=ensemble_size,
            input_dim=context_hidden_sizes[-1],
            output_dim=cp_output_dim,
            activation=output_nonlinearity,
            weight_decay=context_weight_decays[-1],
        )
        l2_regs += [head_l2_reg]

        def forward(xx, inference=False):
            for layer in layers:
                xx = layer(xx)
            output_head = head_layer(xx)
            output = tf.tile(
                tf.reshape(output_head, [1, ensemble_size, -1, cp_output_dim]),
                [head_size, 1, 1, 1],
            )
            return output

    bs_normalized_input_cp_obs = normalize(
        bs_input_cp_obs_var, norm_cp_obs_mean_var, norm_cp_obs_std_var
    )
    bs_normalized_input_cp_act = normalize(
        bs_input_cp_act_var, norm_cp_act_mean_var, norm_cp_act_std_var
    )
    bs_normalized_cp_x = tf.concat(
        [bs_normalized_input_cp_obs, bs_normalized_input_cp_act], axis=-1
    )
    bs_cp_output_var = forward(bs_normalized_cp_x)  # [head_size, ensemble_size, ...]

    return bs_cp_output_var, l2_regs, forward


def normalize(data_array, mean, std):
    return (data_array - mean) / (std + 1e-10)


def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean


def create_dense_layer(
    name, ensemble_size, input_dim, output_dim, activation, weight_decay=0.0
):
    weights = tf.get_variable(
        "{}_weight".format(name),
        shape=[ensemble_size, input_dim, output_dim],
        initializer=tf.truncated_normal_initializer(
            stddev=1 / (2 * np.sqrt(input_dim))
        ),
    )
    biases = tf.get_variable(
        "{}_bias".format(name),
        shape=[ensemble_size, 1, output_dim],
        initializer=tf.constant_initializer(0.0),
    )

    l2_reg = tf.multiply(
        weight_decay, tf.nn.l2_loss(weights), name="{}_l2_reg".format(name)
    )

    def _thunk(input_tensor):
        out = tf.matmul(input_tensor, weights) + biases
        out = activation(out)
        return out

    return _thunk, l2_reg
