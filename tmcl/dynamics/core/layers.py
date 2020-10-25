from tmcl.utils.serializable import Serializable
from tmcl.utils.utils import remove_scope_from_name
from tmcl.dynamics.core.utils import *
import tensorflow as tf
import copy
from collections import OrderedDict


class Layer(Serializable):
    """
    A container for storing the current pre and post update policies
    Also provides functions for executing and updating policy parameters

    Note:
        the preupdate policy is stored as tf.Variables, while the postupdate
        policy is stored in numpy arrays and executed through tf.placeholders

    Args:
        obs_dim (int): dimensionality of the observation space -> specifies the input size of the policy
        action_dim (int): dimensionality of the action space -> specifies the output size of the policy
        name (str) : Name used for scoping variables in policy
        hidden_sizes (tuple) : size of hidden layers of network
        learn_std (bool) : whether to learn variance of network output
        hidden_nonlinearity (Operation) : nonlinearity used between hidden layers of network
        output_nonlinearity (Operation) : nonlinearity used after the final layer of network
    """

    def __init__(
        self,
        name,
        input_dim,
        output_dim,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=tf.nn.relu,
        context_hidden_sizes=(32, 32),
        context_hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=None,
        input_var=None,
        params=None,
        context_var=None,
        context_dim=0,
        context_out_dim=10,
        context_act_var=None,
        context_obs_var=None,
        context_act_dim=0,
        context_obs_dim=0,
        stochastic_flag=0,
        n_samples=1,
        sim_dim=0,
        input_act_var=None,
        input_obs_var=None,
        input_act_dim=None,
        input_obs_dim=None,
        use_gaussian_product=False,
        cp_output_var=None,
        action_space=None,
        policy_hidden_sizes=(32, 32),
        policy_hidden_nonlinearity=tf.nn.tanh,
        dynamics_hidden_sizes=(32, 32),
        sigma_flag=1,
        separate_flag=1,
        n_forwards=1,
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
        discrete=False,
        bootstrap_idx_var=None,
        ensemble_size=5,
        head_size=3,
        n_particles=20,
        cem_init_mean_var=None,
        cem_init_var_var=None,
        obs_preproc_fn=None,
        obs_postproc_fn=None,
        use_cem=False,
        bs_input_obs_var=None,
        bs_input_act_var=None,
        bs_input_cp_obs_var=None,
        bs_input_cp_act_var=None,
        bs_input_cp_var=None,
        deterministic=False,
        weight_decays=None,
        context_weight_decays=None,
        history_length=None,
        build_policy_graph=False,
        cp_forward=None,
        input_history_obs_var=None,
        input_history_act_var=None,
        input_history_delta_var=None,
        use_simulation_param=False,
        simulation_param_dim=None,
        simulation_param_var=None,
        bs_input_sim_param_var=None,
        sep_layer_size=0,
        use_global_head=False,
        non_adaptive_planning=False,
        **kwargs
    ):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.input_var = input_var

        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        self.context_hidden_sizes = context_hidden_sizes
        self.context_hidden_nonlinearity = context_hidden_nonlinearity
        self.context_dim = context_dim
        self.context_var = context_var
        self.context_out_dim = context_out_dim
        self.context_act_dim = context_act_dim
        self.context_act_var = context_act_var
        self.context_obs_dim = context_obs_dim
        self.context_obs_var = context_obs_var
        self.stochastic_flag = stochastic_flag
        self.n_samples = n_samples
        self.sim_dim = sim_dim
        self.input_act_var = input_act_var
        self.input_obs_var = input_obs_var
        self.input_act_dim = input_act_dim
        self.input_obs_dim = input_obs_dim
        self.use_gaussian_product = use_gaussian_product
        self.cp_output_var = cp_output_var
        self.policy_hidden_sizes = policy_hidden_sizes
        self.policy_hidden_nonlinearity = policy_hidden_nonlinearity
        self.dynamics_hidden_sizes = dynamics_hidden_sizes
        self.sigma_flag = sigma_flag
        self.separate_flag = separate_flag
        self.n_forwards = n_forwards
        self.reward_fn = reward_fn
        self.n_candidates = n_candidates
        self.discrete = discrete
        self.ensemble_size = ensemble_size
        self.n_particles = n_particles
        self.bootstrap_idx_var = bootstrap_idx_var
        self.cem_init_mean_var = cem_init_mean_var
        self.cem_init_var_var = cem_init_var_var
        self.obs_preproc_fn = obs_preproc_fn
        self.obs_postproc_fn = obs_postproc_fn
        self.use_cem = use_cem

        self.deterministic = deterministic
        self.weight_decays = weight_decays
        self.context_weight_decays = context_weight_decays
        self.build_policy_graph = build_policy_graph
        self.cp_forward = cp_forward

        # Input Placeholders
        self.bs_input_obs_var = bs_input_obs_var
        self.bs_input_act_var = bs_input_act_var
        self.bs_input_cp_obs_var = bs_input_cp_obs_var
        self.bs_input_cp_act_var = bs_input_cp_act_var
        self.bs_input_cp_var = bs_input_cp_var

        # Normalization Stat Placeholders
        self.norm_obs_mean_var = norm_obs_mean_var
        self.norm_obs_std_var = norm_obs_std_var
        self.norm_act_mean_var = norm_act_mean_var
        self.norm_act_std_var = norm_act_std_var
        self.norm_delta_mean_var = norm_delta_mean_var
        self.norm_delta_std_var = norm_delta_std_var
        self.norm_cp_obs_mean_var = norm_cp_obs_mean_var
        self.norm_cp_obs_std_var = norm_cp_obs_std_var
        self.norm_cp_act_mean_var = norm_cp_act_mean_var
        self.norm_cp_act_std_var = norm_cp_act_std_var
        self.norm_back_delta_mean_var = norm_back_delta_mean_var
        self.norm_back_delta_std_var = norm_back_delta_std_var

        self.history_length = history_length

        # history variables for MCL
        self.input_history_obs_var = input_history_obs_var
        self.input_history_act_var = input_history_act_var
        self.input_history_delta_var = input_history_delta_var

        # MCL
        self.use_simulation_param = use_simulation_param
        self.simulation_param_dim = simulation_param_dim
        self.simulation_param_var = simulation_param_var
        self.bs_input_sim_param_var = bs_input_sim_param_var
        self.head_size = head_size
        self.sep_layer_size = sep_layer_size

        # CaDM + MCL
        self.use_global_head = use_global_head
        self.non_adaptive_planning = non_adaptive_planning

        self._params = params
        self._assign_ops = None
        self._assign_phs = None

    def build_graph(self):
        """
        Builds computational graph for policy
        """
        raise NotImplementedError

    """ --- methods for serialization --- """

    def get_params(self):
        """
        Get the tf.Variables representing the trainable weights of the network (symbolic)

        Returns:
            (dict) : a dict of all trainable Variables
        """
        return self._params

    def get_param_values(self):
        """
        Gets a list of all the current weights in the network (in original code it is flattened, why?)

        Returns:
            (list) : list of values for parameters
        """
        param_values = tf.get_default_session().run(self._params)
        return param_values

    def set_params(self, policy_params):
        """
        Sets the parameters for the graph

        Args:
            policy_params (dict): of variable names and corresponding parameter values
        """
        assert all(
            [k1 == k2 for k1, k2 in zip(self.get_params().keys(), policy_params.keys())]
        ), "parameter keys must match with variable"

        if self._assign_ops is None:
            assign_ops, assign_phs = [], []
            for var in self.get_params().values():
                assign_placeholder = tf.placeholder(dtype=var.dtype)
                assign_op = tf.assign(var, assign_placeholder)
                assign_ops.append(assign_op)
                assign_phs.append(assign_placeholder)
            self._assign_ops = assign_ops
            self._assign_phs = assign_phs
        feed_dict = dict(zip(self._assign_phs, policy_params.values()))
        tf.get_default_session().run(self._assign_ops, feed_dict=feed_dict)

    def __getstate__(self):
        state = {
            # 'init_args': Serializable.__getstate__(self),
            "network_params": self.get_param_values()
        }
        return state

    def __setstate__(self, state):
        # Serializable.__setstate__(self, state['init_args'])
        tf.get_default_session().run(
            tf.variables_initializer(self.get_params().values())
        )
        self.set_params(state["network_params"])


# CHECK
class MCLMultiHeadedCaDMEnsembleMLP(Layer):
    def __init__(self, *args, **kwargs):
        # store the init args for serialization and call the super constructors
        Serializable.quick_init(self, locals())
        Layer.__init__(self, *args, **kwargs)

        self.build_graph()

    def build_graph(self):
        """
        Builds computational graph for policy
        """

        if not self.use_cem:
            self.cem_init_mean_var = None
            self.cem_init_std_var = None

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            if self._params is None:
                # build the actual policy network
                (
                    self.input_obs_var,
                    self.input_act_var,
                    self.output_var,
                    self.optimal_action_var,
                    self.mu,
                    self.logvar,
                    self.max_logvar,
                    self.min_logvar,
                    self.l2_regs,
                    self.embedding,
                    self.inference_cp_var,
                ) = create_mcl_cadm_multiheaded_mlp(
                    output_dim=self.output_dim,
                    hidden_sizes=self.hidden_sizes,
                    hidden_nonlinearity=self.hidden_nonlinearity,
                    output_nonlinearity=self.output_nonlinearity,
                    input_obs_dim=self.input_obs_dim,
                    input_act_dim=self.input_act_dim,
                    input_obs_var=self.input_obs_var,
                    input_act_var=self.input_act_var,
                    input_history_obs_var=self.input_history_obs_var,
                    input_history_act_var=self.input_history_act_var,
                    input_history_delta_var=self.input_history_delta_var,
                    n_forwards=self.n_forwards,
                    reward_fn=self.reward_fn,
                    n_candidates=self.n_candidates,
                    norm_obs_mean_var=self.norm_obs_mean_var,
                    norm_obs_std_var=self.norm_obs_std_var,
                    norm_act_mean_var=self.norm_act_mean_var,
                    norm_act_std_var=self.norm_act_std_var,
                    norm_delta_mean_var=self.norm_delta_mean_var,
                    norm_delta_std_var=self.norm_delta_std_var,
                    norm_back_delta_mean_var=self.norm_back_delta_mean_var,
                    norm_back_delta_std_var=self.norm_back_delta_std_var,
                    norm_cp_obs_mean_var=self.norm_cp_obs_mean_var,
                    norm_cp_obs_std_var=self.norm_cp_obs_std_var,
                    norm_cp_act_mean_var=self.norm_cp_act_mean_var,
                    norm_cp_act_std_var=self.norm_cp_act_std_var,
                    discrete=self.discrete,
                    ensemble_size=self.ensemble_size,
                    head_size=self.head_size,
                    bs_input_obs_var=self.bs_input_obs_var,
                    bs_input_act_var=self.bs_input_act_var,
                    n_particles=self.n_particles,
                    cem_init_mean_var=self.cem_init_mean_var,
                    cem_init_var_var=self.cem_init_var_var,
                    obs_preproc_fn=self.obs_preproc_fn,
                    obs_postproc_fn=self.obs_postproc_fn,
                    deterministic=self.deterministic,
                    weight_decays=self.weight_decays,
                    use_simulation_param=self.use_simulation_param,
                    simulation_param_dim=self.simulation_param_dim,
                    simulation_param_var=self.simulation_param_var,
                    bs_input_sim_param_var=self.bs_input_sim_param_var,
                    sep_layer_size=self.sep_layer_size,
                    # CaDM
                    input_cp_obs_var=self.context_obs_var,
                    input_cp_act_var=self.context_act_var,
                    cp_output_dim=self.context_out_dim,
                    cp_forward=self.cp_forward,
                    bs_input_cp_var=self.bs_input_cp_var,
                    build_policy_graph=self.build_policy_graph,
                    non_adaptive_planning=self.non_adaptive_planning,
                )

                # save the policy's trainable variables in dicts
                current_scope = tf.get_default_graph().get_name_scope()
                trainable_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope
                )
                self._params = OrderedDict(
                    [
                        (remove_scope_from_name(var.name, current_scope), var)
                        for var in trainable_vars
                    ]
                )


# CHECK
class MultiHeadedEnsembleContextPredictor(Layer):
    def __init__(self, *args, **kwargs):
        # store the init args for serialization and call the super constructors
        Serializable.quick_init(self, locals())
        Layer.__init__(self, *args, **kwargs)
        self.build_graph()

    def build_graph(self):
        """
        Builds computational graph for policy
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # build the actual policy network
            (
                self.context_output_var,
                self.l2_regs,
                self.forward,
            ) = create_ensemble_multiheaded_context_predictor(
                context_hidden_sizes=self.context_hidden_sizes,
                context_hidden_nonlinearity=self.context_hidden_nonlinearity,
                output_nonlinearity=self.output_nonlinearity,
                ensemble_size=self.ensemble_size,
                cp_input_dim=self.context_dim,
                context_weight_decays=self.context_weight_decays,
                bs_input_cp_obs_var=self.bs_input_cp_obs_var,
                bs_input_cp_act_var=self.bs_input_cp_act_var,
                norm_cp_obs_mean_var=self.norm_cp_obs_mean_var,
                norm_cp_obs_std_var=self.norm_cp_obs_std_var,
                norm_cp_act_mean_var=self.norm_cp_act_mean_var,
                norm_cp_act_std_var=self.norm_cp_act_std_var,
                cp_output_dim=self.context_out_dim,
                head_size=self.head_size,
                use_global_head=self.use_global_head,
            )

            # save the policy's trainable variables in dicts
            current_scope = tf.get_default_graph().get_name_scope()
            trainable_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope
            )
            self._params = OrderedDict(
                [
                    (remove_scope_from_name(var.name, current_scope), var)
                    for var in trainable_vars
                ]
            )

