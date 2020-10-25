import numpy as np
import itertools
from tmcl.samplers.vectorized_env_executor import ParallelEnvExecutor


def rollout_multi(
    vec_env,
    policy,
    discrete,
    animated=False,
    ignore_done=False,
    num_rollouts=10,
    test_total=20,
    state_diff=False,
    act_dim=None,
    use_cem=False,
    horizon=None,
    context=None,
    history_length=None,
    env=None,
    mcl_cadm=False,
):

    num_envs = vec_env.num_envs
    obses = np.asarray(vec_env.reset())

    if use_cem:
        prev_sol = np.tile(0.0, [num_rollouts, horizon, act_dim])
        init_var = np.tile(np.square(2) / 16, [num_rollouts, horizon, act_dim])

    n_test = 0
    total_reward_list = []
    test_reward_list = np.zeros(num_envs)
    policy.reset(dones=[True] * num_envs)
    sim_params = vec_env.get_sim_params()

    while n_test < test_total:
        if use_cem:
            cem_solutions, agent_infos = policy.get_actions(
                obses, init_mean=prev_sol, init_var=init_var, sim_params=sim_params
            )
            prev_sol[:, :-1] = cem_solutions[:, 1:]
            prev_sol[:, -1:] = 0.0
            actions = cem_solutions[:, 0]
        else:
            actions, agent_info = policy.get_actions(obses, sim_params=sim_params)
        if discrete:
            actions = actions.reshape(-1)

        next_obses, rewards, dones, env_infos = vec_env.step(actions)

        reset_flag = 0
        for idx, reward, done in zip(itertools.count(), rewards, dones):
            test_reward_list[idx] += reward
            if done:
                reset_flag += 1
                n_test += 1
                total_reward_list.append(test_reward_list[idx])
                test_reward_list[idx] = 0
                if use_cem:
                    prev_sol[idx] = 0.0

        if reset_flag > 0:
            policy.reset(dones=dones)
        obses = np.asarray(next_obses)
        sim_params = vec_env.get_sim_params()

    return np.average(total_reward_list)


def context_rollout_multi(
    vec_env,
    policy,
    discrete,
    animated=False,
    ignore_done=False,
    num_rollouts=10,
    test_total=20,
    state_diff=False,
    act_dim=None,
    use_cem=False,
    horizon=None,
    context=None,
    history_length=None,
    env=None,
    mcl_cadm=False,
):

    num_envs = vec_env.num_envs
    obses = np.asarray(vec_env.reset())

    if use_cem:
        prev_sol = np.tile(0.0, [num_rollouts, horizon, act_dim])
        init_var = np.tile(np.square(2) / 16, [num_rollouts, horizon, act_dim])

    obs_dim = obses.shape[1]

    n_test = 0
    total_reward_list = []
    test_reward_list = np.zeros(num_envs)
    policy.reset(dones=[True] * num_envs)
    history_state = np.zeros((obses.shape[0], obs_dim * history_length))
    history_act = np.zeros((obses.shape[0], act_dim * history_length))
    history_state_whole = np.zeros((obses.shape[0], obs_dim * history_length))
    sim_params = vec_env.get_sim_params()
    state_counts = [0] * num_envs

    while n_test < test_total:
        if mcl_cadm:
            if state_counts[0] > 1:
                _history_state = np.reshape(
                    history_state_whole, (obses.shape[0], history_length, obs_dim)
                )
                _history_act = np.reshape(
                    history_act, (obses.shape[0], history_length, act_dim)
                )

                _history_next_obs = _history_state[:, 1:]
                _history_obs = _history_state[:, :-1]
                _history_act = _history_act[:, :-1]
                _history_delta = env.targ_proc(_history_obs, _history_next_obs)

                hist_length = state_counts[0] - 1
                _history_next_obs = _history_next_obs[:, :hist_length]
                _history_obs = _history_obs[:, :hist_length]
                _history_act = _history_act[:, :hist_length]
                _history_delta = _history_delta[:, :hist_length]
            else:
                _history_obs = np.zeros((obses.shape[0], 0, obs_dim))
                _history_act = np.zeros((obses.shape[0], 0, act_dim))
                _history_delta = np.zeros((obses.shape[0], 0, obs_dim))

            if use_cem:
                cem_solutions, agent_infos = policy.get_actions(
                    obses,
                    cp_obs=history_state,
                    cp_act=history_act,
                    history_obs=_history_obs,
                    history_act=_history_act,
                    history_delta=_history_delta,
                    init_mean=prev_sol,
                    init_var=init_var,
                    sim_params=sim_params,
                )
                prev_sol[:, :-1] = cem_solutions[:, 1:]
                prev_sol[:, -1:] = 0.0
                actions = cem_solutions[:, 0]
            else:
                actions, agent_info = policy.get_actions(
                    obses,
                    cp_obs=history_state,
                    cp_act=history_act,
                    history_obs=_history_obs,
                    history_act=_history_act,
                    history_delta=_history_delta,
                    sim_params=sim_params,
                )

        else:
            if use_cem:
                cem_solutions, agent_infos = policy.get_actions(
                    obses,
                    cp_obs=history_state,
                    cp_act=history_act,
                    init_mean=prev_sol,
                    init_var=init_var,
                )
                prev_sol[:, :-1] = cem_solutions[:, 1:]
                prev_sol[:, -1:] = 0.0
                actions = cem_solutions[:, 0]
            else:
                actions, agent_info = policy.get_actions(
                    obses, cp_obs=history_state, cp_act=history_act
                )
        if discrete:
            actions = actions.reshape(-1)

        next_obses, rewards, dones, env_infos = vec_env.step(actions)

        reset_flag = 0
        for idx, obs, action, reward, done in zip(
            itertools.count(), obses, actions, rewards, dones
        ):
            if discrete:
                action = np.eye(act_dim)[action]
            else:
                if action.ndim == 0:
                    action = np.expand_dims(action, 0)
            test_reward_list[idx] += reward

            if state_counts[idx] < history_length:
                if state_diff == 0:
                    history_state[idx][
                        state_counts[idx] * obs_dim : (state_counts[idx] + 1) * obs_dim
                    ] = obs
                else:
                    history_state[idx][
                        state_counts[idx] * obs_dim : (state_counts[idx] + 1) * obs_dim
                    ] = (next_obses[idx] - obs)
                history_state_whole[idx][
                    state_counts[idx] * obs_dim : (state_counts[idx] + 1) * obs_dim
                ] = obs
                history_act[idx][
                    state_counts[idx] * act_dim : (state_counts[idx] + 1) * act_dim
                ] = action
            else:
                history_state[idx][:-obs_dim] = history_state[idx][obs_dim:]
                if state_diff == 0:
                    history_state[idx][-obs_dim:] = obs
                else:
                    history_state[idx][-obs_dim:] = next_obses[idx] - obs
                history_state_whole[idx][:-obs_dim] = history_state_whole[idx][obs_dim:]
                history_state_whole[idx][-obs_dim:] = obs
                history_act[idx][:-act_dim] = history_act[idx][act_dim:]
                history_act[idx][-act_dim:] = action

            if done:
                reset_flag += 1
                n_test += 1
                total_reward_list.append(test_reward_list[idx])
                test_reward_list[idx] = 0
                history_state[idx] = np.zeros((obs_dim * history_length))
                history_act[idx] = np.zeros((act_dim * history_length))
                history_state_whole[idx] = np.zeros((obs_dim * history_length))
                state_counts[idx] = 0
                if use_cem:
                    prev_sol[idx] = 0.0
            else:
                state_counts[idx] += 1
        if reset_flag > 0:
            policy.reset(dones=dones)
        obses = np.asarray(next_obses)
        sim_params = vec_env.get_sim_params()

    return np.average(total_reward_list)
