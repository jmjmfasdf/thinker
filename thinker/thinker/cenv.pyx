# distutils: language = c++
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn.functional as F
import thinker.util as util

import cython
from libcpp cimport bool
from libcpp.vector cimport vector
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from libc.math cimport sqrt, log
from libc.stdlib cimport malloc, free

# util function

@cython.cdivision(True)
cdef float average(vector[float]& arr):
    cdef int n = arr.size()    
    if n == 0: return 0.
    cdef float sum = 0
    cdef int i    
    for i in range(n): sum += arr[i]
    return sum / n    

cdef float maximum(vector[float]& arr):
    cdef int n = arr.size()    
    if n == 0: return 0.
    cdef float max_val = arr[0]
    cdef int i
    for i in range(1, n): 
        if arr[i] > max_val: 
            max_val = arr[i]
    return max_val       

# Node-related function (we use structure instead of class to minimize Python code)

cdef struct Node:
    int action # action
    float r # reward
    float v # value
    int t # time step when last expanded
    bool done # whether done or not
    float logit # logit    
    vector[Node*]* ppchildren # children node list
    Node* pparent # parent node
    float trail_r # trailing reward
    float trail_discount # trailing discount
    float rollout_q # trailing rollout q    
    bool visited # visited?
    vector[float]* prollout_qs # all rollout return
    vector[vector[Node*]*]* ppaths # node path corresponding to the rollout return in prollout_qs
    int rollout_n # number of rollout
    float max_q # maximum of all v    
    PyObject* encoded # all python object    
    int rec_t # number of planning step
    int num_actions # number of actions
    float discounting # discount rate
    bool remember_path # whethere to remember path for the rollout

cdef Node* node_new(Node* pparent, int action, float logit, int num_actions, float discounting, int rec_t, bool remember_path):
    cdef Node* pnode = <Node*> malloc(sizeof(Node))
    cdef vector[Node*]* ppchildren =  new vector[Node*]()
    cdef vector[float]* prollout_qs = new vector[float]()
    cdef vector[vector[Node*]*]* ppaths
    if remember_path:
        ppaths = new vector[vector[Node*]*]()
    else:
        ppaths = NULL
    pnode[0] = Node(action=action, r=0., v=0., t=0, done=False, logit=logit, ppchildren=ppchildren, pparent=pparent, trail_r=0., trail_discount=1., rollout_q=0,
        visited=False, prollout_qs=prollout_qs, ppaths=ppaths, rollout_n=0, max_q=0., encoded=NULL, rec_t=rec_t, num_actions=num_actions, discounting=discounting, remember_path=remember_path)
    return pnode

cdef bool node_expanded(Node* pnode, int t):
    """
    Whether the node is expanded after time step t
    """
    return pnode[0].ppchildren[0].size() > 0 and t <= pnode[0].t

cdef node_expand(Node* pnode, float r, float v, int t, bool done, float[:] logits, PyObject* encoded, bool override):
    """
    First time arriving a node and so we expand it
    """
    cdef int a    
    cdef Node* pnode_
    if override and not node_expanded(pnode, -1):
        override = False # no override if not yet expanded

    if not override: 
        assert not node_expanded(pnode, -1), "node should not be expanded"
    else:        
        pnode[0].prollout_qs[0][0] = r + v * pnode[0].discounting
        for a in range(1, int(pnode[0].prollout_qs[0].size())):
            pnode[0].prollout_qs[0][a] = pnode[0].prollout_qs[0][a] - pnode[0].r + r
        if pnode[0].pparent != NULL and pnode[0].remember_path:
            node_refresh(pnode[0].pparent, pnode, r - pnode[0].r, v - pnode[0].v, pnode[0].discounting, 1)
    pnode[0].r = r
    pnode[0].v = v
    pnode[0].t = t
    if pnode[0].encoded != NULL: 
        Py_DECREF(<object>pnode[0].encoded)    
    pnode[0].encoded = encoded
    pnode[0].done = done
    Py_INCREF(<object>encoded)
    for a in range(pnode[0].num_actions):
        if not override:
            pnode[0].ppchildren[0].push_back(node_new(pparent=pnode, action=a, logit=logits[a], 
                num_actions = pnode[0].num_actions, discounting = pnode[0].discounting, rec_t = pnode[0].rec_t,
                remember_path = pnode[0].remember_path))
        else:
            pnode[0].ppchildren[0][a][0].logit = logits[a]    

cdef node_refresh(Node* pnode, Node* pnode_to_refresh, float r_diff, float v_diff, float discounting, int depth):
    """
    Refresh the r and v in the rollout_qs that contains pnode[0]; only available when remember_path is enabled
    """
    cdef int i, j, k
    cdef Node* pnode_check
    for i in range(int(pnode[0].ppaths[0].size())):
        j = int(pnode[0].ppaths[0][i][0].size())
        k = j - 1 - depth
        if k < 0: continue
        pnode_check = pnode[0].ppaths[0][i][0][k]
        if pnode_check == pnode_to_refresh:
            pnode[0].prollout_qs[0][i] += discounting * r_diff
            if k == 0:
                pnode[0].prollout_qs[0][i] += discounting * pnode[0].discounting * v_diff
    if pnode[0].pparent != NULL: node_refresh(pnode[0].pparent, pnode_to_refresh, r_diff, v_diff, discounting * pnode[0].discounting, depth+1)

cdef node_visit(Node* pnode):
    cdef vector[Node*]* ppath    
    pnode[0].trail_r = 0.
    pnode[0].trail_discount = 1.    
    if not pnode[0].visited and pnode[0].remember_path:
        ppath = new vector[Node*]()
    else:
        ppath = NULL
    node_propagate(pnode=pnode, r=pnode[0].r, v=pnode[0].v, new_rollout=not pnode[0].visited, ppath=ppath)
    # node_propagate stores an owned copy at every node on the path.  The
    # initially empty seed vector is only scratch space and must be released by
    # its creator.
    if ppath != NULL:
        del ppath
    pnode[0].visited = True
    pnode[0].rollout_n = pnode[0].rollout_n + 1        

cdef void node_propagate(Node* pnode, float r, float v, bool new_rollout, vector[Node*]* ppath):
    cdef int i
    cdef vector[Node*]* ppath_
    pnode[0].trail_r = pnode[0].trail_r + pnode[0].trail_discount * r
    pnode[0].trail_discount = pnode[0].trail_discount * pnode[0].discounting
    pnode[0].rollout_q = pnode[0].trail_r + pnode[0].trail_discount * v    
    if new_rollout:
        if pnode[0].remember_path:
            ppath_ = new vector[Node*]()
            for i in range(int(ppath[0].size())):                
                ppath_.push_back(ppath[0][i])
            ppath_.push_back(pnode)
            pnode[0].ppaths[0].push_back(ppath_)
        else:
            ppath_ = NULL
        pnode[0].prollout_qs[0].push_back(pnode[0].rollout_q)        
        #pnode[0].rollout_n = pnode[0].rollout_n + 1        
    if pnode[0].pparent != NULL: 
        node_propagate(pnode[0].pparent, r, v, new_rollout, ppath=ppath_)

#@cython.cdivision(True)
cdef float[:] node_stat(Node* pnode, bool detailed, int enc_type, int enc_f_type, int mask_type, int raw_num_actions=-1, bool dynamic=False):
    cdef int i, j, dim_actions, base_idx
    cdef float[:] result
    cdef int obs_n
    obs_n = pnode[0].num_actions*5+3    
    if detailed: obs_n += 3

    result = np.zeros(obs_n, dtype=np.float32)    
    pnode[0].max_q = (maximum(pnode[0].prollout_qs[0]) - pnode[0].r) / pnode[0].discounting
    if mask_type == 3: return result

    result[pnode[0].action] = 1. # action
    if enc_type == 0:
        f = lambda x : x  
    else:
        if enc_f_type == 0:
            f = enc_0
        elif enc_f_type == 1:
            f = enc_1
    result[pnode[0].num_actions] = f(pnode[0].r) # reward
    result[pnode[0].num_actions+1] = <float>pnode[0].done # done
    if not mask_type in [2]:         
        result[pnode[0].num_actions+2] = f(pnode[0].v) # value
    for i in range(int(pnode[0].ppchildren[0].size())):
        child = pnode[0].ppchildren[0][i][0]
        if not mask_type in [2]: 
            result[pnode[0].num_actions+3+i] = child.logit # child_logits
        if not mask_type in [1, 2] and not (mask_type == 5 and not detailed): 
            result[pnode[0].num_actions*2+3+i] = f(average(child.prollout_qs[0])) # child_rollout_qs_mean
            if not mask_type in [3, 4]:
                result[pnode[0].num_actions*3+3+i] = f(maximum(child.prollout_qs[0])) # child_rollout_qs_max
            if dynamic:
                # Dynamic search has no fixed recurrent budget with which to
                # normalize visit counts.  A log transform keeps the feature
                # bounded enough for the actor while retaining ordering.
                result[pnode[0].num_actions*4+3+i] = log(1. + child.rollout_n)
            else:
                result[pnode[0].num_actions*4+3+i] = child.rollout_n / <float>pnode[0].rec_t # child_rollout_ns_enc
    base_idx = pnode[0].num_actions*5+3
    
    if detailed and not mask_type in [1, 2, 4]:        
        result[pnode[0].num_actions*5+3] = f((pnode[0].trail_r - pnode[0].r) / pnode[0].discounting)
        result[pnode[0].num_actions*5+4] = f((pnode[0].rollout_q - pnode[0].r) / pnode[0].discounting)
        result[pnode[0].num_actions*5+5] = f(pnode[0].max_q)
        base_idx += 3

    return result

cdef void node_del(Node* pnode, int except_idx) noexcept:
    cdef int i
    del pnode[0].prollout_qs

    if pnode[0].ppaths != NULL:
        for i in range(int(pnode[0].ppaths[0].size())):
            del pnode[0].ppaths[0][i]
        del pnode[0].ppaths

    for i in range(int(pnode[0].ppchildren[0].size())):
        if i != except_idx:
            node_del(pnode[0].ppchildren[0][i], -1)
        else:
            pnode[0].ppchildren[0][i][0].pparent = NULL
    del pnode[0].ppchildren
    if pnode[0].encoded != NULL:
        Py_DECREF(<object>pnode[0].encoded)
    free(pnode)

cdef void node_collect_snapshot_slots(Node* pnode, object slots):
    """Collect perfect-model snapshot ids reachable from a carried subtree."""
    cdef int i
    cdef dict encoded
    if pnode[0].encoded != NULL:
        encoded = <dict>pnode[0].encoded
        if "snapshot_slot" in encoded:
            slots.add(int(encoded["snapshot_slot"]))
    for i in range(int(pnode[0].ppchildren[0].size())):
        node_collect_snapshot_slots(pnode[0].ppchildren[0][i], slots)

cdef float enc_0(float x):
    return sign(x)*(sqrt(abs(x)+1)-1)+(0.001)*x

cdef float enc_1(float x):
    return sign(x)*log(abs(x)+1)

cdef float sign(float x):
    if x > 0.: return 1.
    if x < 0.: return -1.
    return 0.

cdef float abs(float x):
    if x > 0.: return x
    if x < 0.: return -x
    return 0.

cdef class cWrapper():
    """Wrap the gym environment with a model; output for each 
    step is (out, reward, done, info), where out is a tuple 
    of (gym_env_out, model_out, model_encodes) that corresponds to underlying 
    environment frame, output from the model wrapper, and encoding from the model
    Assume a learned dynamic model.
    """
    # setting
    cdef int rec_t
    cdef int rep_rec_t
    cdef float discounting
    cdef int max_depth    
    cdef bool tree_carry    
    cdef int reset_mode
    cdef bool dynamic_search
    cdef int max_search_steps

    cdef int enc_type
    cdef int enc_f_type
    cdef bool pred_done
    cdef int num_actions    
    cdef int obs_n    
    cdef int env_n
    
    cdef int raw_num_actions
    cdef int raw_dim_actions
    cdef int discrete_k
    cdef int state_dtype
    cdef bool cont_raw_actions

    cdef bool return_h
    cdef bool return_x
    cdef bool return_double
    cdef bool has_action_seq

    cdef bool im_enable    
    cdef int stat_mask_type
    cdef bool time 

    # python object
    cdef object device
    cdef object env
    cdef object timings
    cdef object real_states
    cdef object real_states_np
    cdef object baseline_mean_q    
    cdef object initial_per_state   
    cdef object per_state    
    cdef list model_states_keys
    cdef object default_info    

    cdef readonly object observation_space
    cdef readonly object action_space
    cdef readonly object reward_range
    cdef readonly object metadata

    # tree statistic
    cdef vector[Node*] cur_nodes
    cdef vector[Node*] root_nodes    
    cdef float[:] root_nodes_qmax
    cdef float[:] root_nodes_qmax_
    cdef int[:] rollout_depth
    cdef int[:] max_rollout_depth
    cdef int[:] cur_t

    # Dynamic-search persistent state.  Control actions are PROCEED=0,
    # RESET=1, STOP=2; phases are SEARCH=0, NEED_REAL=1, WAIT=2.
    cdef int[:] phase
    cdef int[:] search_steps
    cdef int[:] pending_real_action
    cdef bool[:] pending_real_valid
    cdef int[:] next_snapshot_slot

    # Dynamic-search events describe only the transition returned by the
    # current wrapper call.  Keeping them separate from persistent phase makes
    # STOP and the synchronous real-action barrier unambiguous to the actor.
    cdef bool[:] tree_reset_event
    cdef bool[:] tree_token_valid_event
    cdef bool[:] search_state_reset_event
    cdef bool[:] real_transition_event
    cdef bool[:] stage_end_event
    cdef bool[:] forced_stop_event
    cdef bool[:] stored_action_mask_event
    cdef bool[:] barrier_released_event
    cdef int[:] executed_primary_action_event
    cdef int[:] accepted_primary_action_event
    cdef int[:] accepted_control_event
    cdef float[:] think_reward_event

    # internal variables only used in step function    
    cdef int[:] max_rollout_depth_
    cdef float[:] mean_q
    cdef float[:] max_q
    cdef int[:] status
    cdef vector[Node*] cur_nodes_
    cdef float[:] par_logits
    cdef float[:] full_reward
    cdef float[:] full_im_reward
    cdef bool[:] full_done
    cdef bool[:] full_truncated_done
    cdef bool[:] full_im_done
    cdef int[:] step_status
    cdef int[:] total_step  
    cdef int[:] step_from_done  
    cdef int internal_counter
    cdef bool resources_closed
    

    def __init__(self, env, env_n, flags, model_net, device=None, timing=False):        
           
        if flags.test_rec_t > 0:
            self.rec_t = flags.test_rec_t
            self.rep_rec_t = flags.rec_t         
        else:
            self.rec_t = flags.rec_t           
            self.rep_rec_t = flags.rec_t         
        self.discounting = flags.discounting
        self.max_depth = flags.max_depth
        self.tree_carry = flags.tree_carry
        self.reset_mode = flags.reset_mode
        self.dynamic_search = getattr(flags, "dynamic_search", False)
        self.max_search_steps = int(getattr(flags, "max_search_steps", -1))
        self.has_action_seq = flags.has_action_seq

        self.has_action_seq = flags.has_action_seq
        if self.max_depth <= 0: self.has_action_seq = False

        self.enc_type = flags.model_enc_type
        self.enc_f_type = flags.model_enc_f_type
        self.pred_done = flags.model_done_loss_cost > 0.     

        self.discrete_k = flags.discrete_k
        self.cont_raw_actions = self.discrete_k > 0

        action_space =  env.action_space[0]
        if type(action_space) == spaces.discrete.Discrete:                    
            self.raw_num_actions = action_space.n    
            self.raw_dim_actions = 1
        elif type(action_space) == spaces.tuple.Tuple:
            self.raw_num_actions = action_space[0].n    
            self.raw_dim_actions = len(action_space)  
        else:
            raise Exception(f"action type {action_space} not supported by cWrapper")          

        self.num_actions = self.raw_num_actions
        if env.observation_space.dtype == 'uint8':
            self.state_dtype = 0
        elif env.observation_space.dtype == 'float32':
            self.state_dtype = 1
        else:
            raise Exception(f"Unupported observation sapce", env.observation_space)

        if self.dynamic_search:
            # detailed root (5A+6), compact current node (5A+3), five
            # budget-independent metadata scalars.
            self.obs_n = 10 * self.num_actions + 14
            self.has_action_seq = False
        else:
            self.obs_n = 11 + self.num_actions * 10 + self.rep_rec_t
            if self.has_action_seq:
                self.obs_n += self.max_depth * self.num_actions
                if self.reset_mode == 0:
                    self.obs_n += self.num_actions

        self.env_n = env_n
        self.return_h = flags.return_h  
        self.return_x = flags.return_x
        self.return_double = False

        self.im_enable = flags.im_enable
        self.stat_mask_type = flags.stat_mask_type
        self.time = timing        
        
        self.device = torch.device("cpu") if device is None else device        
        self.env = env  
        self.timings = util.Timings()        

        self.observation_space = {
            "tree_reps": spaces.Box(low=-np.inf, high=np.inf, shape=(self.env_n, self.obs_n), 
                dtype=np.float32),
            "real_states": self.env.observation_space,
        }

        if self.return_x:
            xs_shape = list(self.env.observation_space.shape)
            if self.return_double:
                xs_shape[1] *= 2
            xs_space = spaces.Box(low=0., high=1., shape=xs_shape, dtype=np.float32)
            self.observation_space["xs"] = xs_space

        if self.return_h:
            hs_shape = [env_n,] + list(model_net.hidden_shape)
            if self.return_double:
                hs_shape[1] *= 2
            hs_space = spaces.Box(low=-np.inf, high=np.inf, shape=hs_shape, dtype=np.float32)
            self.observation_space["hs"] = hs_space

        self.observation_space = spaces.Dict(self.observation_space)

        aug_action_space = spaces.Tuple((spaces.Discrete(self.num_actions),)*self.env_n)
        reset_space = spaces.Tuple((spaces.Discrete(3 if self.dynamic_search else 2),)*self.env_n)
        self.action_space = spaces.Tuple((aug_action_space, reset_space))
        self.reward_range = env.reward_range
        self.metadata = env.metadata

        # internal variable init.
        self.max_rollout_depth_ = np.zeros(self.env_n, dtype=np.intc)
        self.mean_q =  np.zeros(self.env_n, dtype=np.float32)
        self.max_q = np.zeros(self.env_n, dtype=np.float32)
        self.status = np.zeros(self.env_n, dtype=np.intc)
        self.par_logits = np.zeros(self.num_actions, dtype=np.float32)
        self.full_reward = np.zeros(self.env_n, dtype=np.float32)
        self.full_im_reward = np.zeros(self.env_n, dtype=np.float32)
        self.full_done = np.zeros(self.env_n, dtype=np.bool_)
        self.full_truncated_done = np.zeros(self.env_n, dtype=np.bool_)
        self.full_im_done = np.zeros(self.env_n, dtype=np.bool_)
        self.total_step = np.zeros(self.env_n, dtype=np.intc)
        self.step_status = np.zeros(self.env_n, dtype=np.intc)  
        self.step_from_done = np.zeros(self.env_n, dtype=np.intc)  
        self.internal_counter = 0
        self.resources_closed = False

        self.phase = np.zeros(self.env_n, dtype=np.intc)
        self.search_steps = np.zeros(self.env_n, dtype=np.intc)
        self.pending_real_action = np.zeros(self.env_n, dtype=np.intc)
        self.pending_real_valid = np.zeros(self.env_n, dtype=np.bool_)
        self.next_snapshot_slot = np.ones(self.env_n, dtype=np.intc)
        self.tree_reset_event = np.zeros(self.env_n, dtype=np.bool_)
        self.tree_token_valid_event = np.zeros(self.env_n, dtype=np.bool_)
        self.search_state_reset_event = np.zeros(self.env_n, dtype=np.bool_)
        self.real_transition_event = np.zeros(self.env_n, dtype=np.bool_)
        self.stage_end_event = np.zeros(self.env_n, dtype=np.bool_)
        self.forced_stop_event = np.zeros(self.env_n, dtype=np.bool_)
        self.stored_action_mask_event = np.zeros(self.env_n, dtype=np.bool_)
        self.barrier_released_event = np.zeros(self.env_n, dtype=np.bool_)
        self.executed_primary_action_event = np.full(self.env_n, -1, dtype=np.intc)
        self.accepted_primary_action_event = np.full(self.env_n, -1, dtype=np.intc)
        self.accepted_control_event = np.full(self.env_n, -1, dtype=np.intc)
        self.think_reward_event = np.zeros(self.env_n, dtype=np.float32)
    
    cdef float[:, :] compute_tree_reps(self, int[:]& reset, int[:]& status):
        cdef int i, j
        cdef int idx1, idx2, idx3, idx4, idx5
        cdef float[:, :] result
        cdef Node* node

        idx1 = self.num_actions * 5 + 6
        idx2 = idx1
        idx3 = idx2 + self.num_actions * 5 + 3
        idx4 = idx3
        idx5 = idx4 + 2 + self.rep_rec_t        

        result = np.zeros((self.env_n, self.obs_n), dtype=np.float32)
        for i in range(self.env_n):
            result[i, :idx1] = node_stat(self.root_nodes[i], detailed=True, enc_type=self.enc_type, enc_f_type=self.enc_f_type, mask_type=self.stat_mask_type, raw_num_actions=self.raw_num_actions)
            result[i, idx2:idx3] = node_stat(self.cur_nodes[i], detailed=False, enc_type=self.enc_type, enc_f_type=self.enc_f_type, mask_type=self.stat_mask_type, raw_num_actions=self.raw_num_actions)
            # reset
            if reset is None or status[i] == 1:
                result[i, idx4] = 1.
            else:
                result[i, idx4] = reset[i]
            # time
            if self.cur_t[i] < self.rep_rec_t:
                result[i, idx4+1+self.cur_t[i]] = 1.
            # deprec
            result[i, idx4+self.rep_rec_t+1] = (self.discounting ** (self.rollout_depth[i]))     
            # action sequence
            if self.has_action_seq:
                node = self.cur_nodes[i]            
                for j in range(self.rollout_depth[i] + 1):               
                    result[i, idx5+(self.rollout_depth[i] - j)*self.num_actions+node[0].action] = 1.
                    node = node[0].pparent

        return result

    cdef void clear_dynamic_events(self):
        cdef int i
        for i in range(self.env_n):
            self.tree_reset_event[i] = False
            self.tree_token_valid_event[i] = False
            self.search_state_reset_event[i] = False
            self.real_transition_event[i] = False
            self.stage_end_event[i] = False
            self.forced_stop_event[i] = False
            self.stored_action_mask_event[i] = False
            self.barrier_released_event[i] = False
            self.executed_primary_action_event[i] = -1
            self.accepted_primary_action_event[i] = -1
            self.accepted_control_event[i] = -1
            self.think_reward_event[i] = 0.

    cdef void reset_dynamic_state(self):
        """Initialize a fresh search stage after env.reset()."""
        cdef int i
        self.clear_dynamic_events()
        for i in range(self.env_n):
            self.phase[i] = 0
            self.search_steps[i] = 0
            self.pending_real_action[i] = 0
            self.pending_real_valid[i] = False
            self.next_snapshot_slot[i] = 1
            self.tree_reset_event[i] = True
            self.tree_token_valid_event[i] = True
            self.search_state_reset_event[i] = True
            self.step_status[i] = 0

    cdef bool all_real_actions_stored(self):
        cdef int i
        for i in range(self.env_n):
            if not self.pending_real_valid[i]:
                return False
        return True

    cdef float[:, :] compute_tree_reps_dynamic(self):
        cdef int i
        cdef int root_end, cur_end
        cdef float[:, :] result

        root_end = self.num_actions * 5 + 6
        cur_end = root_end + self.num_actions * 5 + 3
        result = np.zeros((self.env_n, self.obs_n), dtype=np.float32)
        for i in range(self.env_n):
            result[i, :root_end] = node_stat(
                self.root_nodes[i], detailed=True, enc_type=self.enc_type,
                enc_f_type=self.enc_f_type, mask_type=self.stat_mask_type,
                raw_num_actions=self.raw_num_actions, dynamic=True)
            result[i, root_end:cur_end] = node_stat(
                self.cur_nodes[i], detailed=False, enc_type=self.enc_type,
                enc_f_type=self.enc_f_type, mask_type=self.stat_mask_type,
                raw_num_actions=self.raw_num_actions, dynamic=True)
            # [tree_reset, depth_discount, log1p(search_steps),
            #  log1p(rollout_depth), search_start]
            result[i, cur_end] = <float>self.tree_reset_event[i]
            result[i, cur_end + 1] = self.discounting ** self.rollout_depth[i]
            result[i, cur_end + 2] = log(1. + self.search_steps[i])
            result[i, cur_end + 3] = log(1. + self.rollout_depth[i])
            result[i, cur_end + 4] = <float>(self.phase[i] == 0 and self.search_steps[i] == 0)
        return result

    def reset(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this!")

    def step(self, *args, **kwargs):  
        raise NotImplementedError("Subclasses must implement this!")

    def render(self, *args, **kwargs):  
        return self.env.render(*args, **kwargs)
    
    cdef compute_model_out(self, vector[Node*] nodes, key):
        cdef int i
        outs = []
        for i in range(self.env_n):
            encoded = <dict>nodes[i][0].encoded
            if key not in encoded or encoded[key] is None: 
                return None
            outs.append((encoded[key] if not nodes[i][0].done 
                else torch.zeros_like(encoded[key])))
        outs = torch.stack(outs, dim=0)
        return outs

    cdef prepare_state(self, int[:]& reset, int[:]& status):
        cdef int i

        if status is None or np.any(np.array(status)==1):
            self.real_states = self.compute_model_out(self.root_nodes, "real_states")
       
        tree_reps = self.compute_tree_reps(reset, status)
        tree_reps = torch.tensor(tree_reps, dtype=torch.float, device=self.device)

        states = {
            "tree_reps": tree_reps,
            "real_states": self.real_states,
        }

        if self.return_x:
            xs = self.compute_model_out(self.cur_nodes, "xs")
            if self.return_double and xs is not None:
                root_xs = self.compute_model_out(self.root_nodes, "xs")
                xs = torch.concat([root_xs, xs], dim=1)        
            assert xs is not None, "xs cannot be None"
            states["xs"] = xs

        if self.return_h:
            hs = self.compute_model_out(self.cur_nodes, "hs")
            if self.return_double and hs is not None:
                root_hs = self.compute_model_out(self.root_nodes, "hs")
                hs = torch.concat([root_hs, hs], dim=1)
            assert hs is not None, "hs cannot be None"
            states["hs"] = hs

        return states    

    cdef prepare_state_dynamic(self):
        cdef int i

        for i in range(self.env_n):
            if self.search_state_reset_event[i]:
                self.real_states = self.compute_model_out(self.root_nodes, "real_states")
                break

        tree_reps = torch.tensor(
            self.compute_tree_reps_dynamic(), dtype=torch.float, device=self.device)
        states = {
            "tree_reps": tree_reps,
            "real_states": self.real_states,
        }

        if self.return_x:
            xs = self.compute_model_out(self.cur_nodes, "xs")
            if self.return_double and xs is not None:
                root_xs = self.compute_model_out(self.root_nodes, "xs")
                xs = torch.concat([root_xs, xs], dim=1)
            assert xs is not None, "xs cannot be None"
            states["xs"] = xs

        if self.return_h:
            hs = self.compute_model_out(self.cur_nodes, "hs")
            if self.return_double and hs is not None:
                root_hs = self.compute_model_out(self.root_nodes, "hs")
                hs = torch.concat([root_hs, hs], dim=1)
            assert hs is not None, "hs cannot be None"
            states["hs"] = hs
        return states

    cdef update_per_state(self, model_net_out, idx):
        if idx is None or len(idx) == self.env_n:
            self.per_state = model_net_out.state
        else:
            for k in model_net_out.state.keys(): 
                self.per_state[k][idx] = model_net_out.state[k]

    def prepare_info(self, info, status, perfect):        
        status = np.array(status)
        if np.all(status == 1): 
            return_info = util.dict_map(info, lambda x: torch.tensor(x, device=self.device))
        elif np.all(status != 1): 
            return_info = util.dict_map(self.default_info, lambda x: x.clone())
        else:
            return_info = util.dict_map(self.default_info, lambda x: x.clone())
            j = 0
            for i in range(self.env_n):
                if status[i] == 1:
                    for k in info: 
                        return_info[k][i] = torch.tensor(info[k][j], device=self.device)
                    j += 1
                if status[i] == 4 and perfect: j += 1        
        
        for i in range(self.env_n):
            # compute step status
            if self.cur_t[i] == 0:
                self.step_status[i] = 0 # real action just taken
            elif self.rec_t <= 1:
                self.step_status[i] = 3 # real action just taken and next action is real action
            elif self.cur_t[i] < self.rec_t - 1:
                self.step_status[i] = 1 # im action just taken
            elif self.cur_t[i] >= self.rec_t - 1:
                self.step_status[i] = 2 # im action just taken; next action is real action

        return_info.update(
            {
                "step_status": torch.tensor(self.step_status, device=self.device),
                "max_rollout_depth":  torch.tensor(self.max_rollout_depth_, dtype=torch.long, device=self.device),
                "baseline": self.baseline_mean_q.unsqueeze(-1),                    
                "real_states_np": self.real_states_np,          
            }
        )
        if not perfect:
            return_info["initial_per_state"] = self.initial_per_state
        if self.im_enable:
            # placeholder only
            return_info["im_reward"] = torch.zeros((self.env_n, 1), dtype=torch.float, device=self.device)
        return return_info

    def prepare_info_dynamic(self, info, perfect):
        """Build the explicit Dynamic Thinker transition contract."""
        cdef int i
        legal_control_mask = np.zeros((self.env_n, 3), dtype=np.bool_)
        for i in range(self.env_n):
            if self.phase[i] == 0:
                # PROCEED may not cross max_depth implicitly: at the boundary
                # the actor must explicitly RESET the rollout or STOP search.
                if not (self.max_depth > 0 and
                        self.rollout_depth[i] >= self.max_depth - 1):
                    legal_control_mask[i, 0] = True
                legal_control_mask[i, 1] = True
                legal_control_mask[i, 2] = True

        if info is None:
            return_info = util.dict_map(self.default_info, lambda x: x.clone())
        else:
            # Dynamic real transitions are released only as a full batch, so
            # unlike the fixed path no subset-to-batch scatter is required.
            return_info = util.dict_map(info, lambda x: torch.tensor(x, device=self.device))

        return_info.update({
            "step_status": torch.tensor(self.step_status, dtype=torch.long, device=self.device),
            "max_rollout_depth": torch.tensor(self.max_rollout_depth_, dtype=torch.long, device=self.device),
            "baseline": self.baseline_mean_q.unsqueeze(-1),
            "real_states_np": self.real_states_np,
            "phase": torch.tensor(self.phase, dtype=torch.long, device=self.device),
            "legal_control_mask": torch.tensor(legal_control_mask, dtype=torch.bool, device=self.device),
            "tree_token_valid": torch.tensor(self.tree_token_valid_event, dtype=torch.bool, device=self.device),
            "search_state_reset": torch.tensor(self.search_state_reset_event, dtype=torch.bool, device=self.device),
            "real_transition": torch.tensor(self.real_transition_event, dtype=torch.bool, device=self.device),
            "stage_end": torch.tensor(self.stage_end_event, dtype=torch.bool, device=self.device),
            "forced_stop": torch.tensor(self.forced_stop_event, dtype=torch.bool, device=self.device),
            "stored_action_mask": torch.tensor(self.stored_action_mask_event, dtype=torch.bool, device=self.device),
            "executed_primary_action": torch.tensor(self.executed_primary_action_event, dtype=torch.long, device=self.device),
            "accepted_primary_action": torch.tensor(self.accepted_primary_action_event, dtype=torch.long, device=self.device),
            "accepted_control": torch.tensor(self.accepted_control_event, dtype=torch.long, device=self.device),
            "search_steps": torch.tensor(self.search_steps, dtype=torch.long, device=self.device),
            "barrier_released": torch.tensor(self.barrier_released_event, dtype=torch.bool, device=self.device),
            "think_reward": torch.tensor(self.think_reward_event, dtype=torch.float, device=self.device),
        })
        if not perfect:
            return_info["initial_per_state"] = self.initial_per_state
        if self.im_enable:
            return_info["im_reward"] = torch.zeros(
                (self.env_n, 1), dtype=torch.float, device=self.device)
        return return_info


    cdef void dispose_trees(self) noexcept:
        """Release every owned root tree and invalidate all pointer vectors."""
        cdef int i
        cdef int root_n = <int>self.root_nodes.size()
        for i in range(root_n):
            if self.root_nodes[i] != NULL:
                node_del(self.root_nodes[i], except_idx=-1)
        self.root_nodes.clear()
        # Current nodes alias nodes owned by a root tree; never delete them a
        # second time, but do not leave dangling pointers behind either.
        self.cur_nodes.clear()

    def close(self):
        self.dispose_trees()
        if not self.resources_closed:
            self.resources_closed = True
            self.env.close()

    def __dealloc__(self):
        # Python calls are unsafe here, so environment shutdown remains in
        # close(); all malloc/new/Py_INCREF resources owned by Node are local
        # and can be released deterministically.
        self.dispose_trees()

    def seed(self, x):
        self.env.seed(x)

    def print_time(self):
        print(self.timings.summary())

    def quick_save(self, env_id=None):
        return self.env.quick_save(env_id)

    def quick_load(self, env_id=None):
        self.env.quick_load(env_id)

    def unwrapped_step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)

    def get_action_meanings(self):
        return self.env.get_action_meanings()  

    def __getattr__(self, name):
        return getattr(self.env, name)   

    def most_visited_path(self, n):
        cdef vector[Node*] nodes, new_nodes
        cdef Node node, child
        cdef Node* ptop_child
        cdef int i, j, m, most_visited, visit_count

        actions = np.zeros((n, self.env_n), dtype=np.intc)
        nodes = self.root_nodes

        for m in range(n):            
            for i in range(self.env_n):
                most_visited, visit_count = 0, 0
                node = nodes[i][0]
                ptop_child = nodes[i]
                for j in range(int(node.ppchildren[0].size())):
                    child = node.ppchildren[0][j][0]
                    if child.rollout_n > visit_count:
                        most_visited = j
                        visit_count = child.rollout_n
                        ptop_child = node.ppchildren[0][j]
                new_nodes.push_back(ptop_child)
                actions[m, i] = most_visited
            nodes = new_nodes
            new_nodes.clear()

        return actions

    def load_ckp(self, data):
        return self.env.load_ckp(data)
    
    def save_ckp(self):
        return self.env.save_ckp()

cdef class cModelWrapper(cWrapper):

    def reset(self, model_net, seed=None):
        """Reset the environment and replace any previously active tree."""
        cdef int i
        cdef Node* root_node
        cdef Node* cur_node
        cdef float[:,:] model_out       

        # reset() may be called again by evaluators or tests.  Replace the
        # active forest instead of appending roots behind stale index-zero
        # entries in the C++ vectors.
        self.dispose_trees()

        with torch.no_grad():
            # some init.
            self.root_nodes_qmax = np.zeros(self.env_n, dtype=np.float32)
            self.root_nodes_qmax_ = np.zeros(self.env_n, dtype=np.float32)
            self.rollout_depth = np.zeros(self.env_n, dtype=np.intc)
            self.max_rollout_depth = np.zeros(self.env_n, dtype=np.intc)
            self.cur_t = np.zeros(self.env_n, dtype=np.intc)
            self.baseline_mean_q = torch.zeros(self.env_n, dtype=torch.float, device=self.device)        

            # reset obs
            obs, info = self.env.reset(reset_stat=True, seed=seed)
            self.default_info = util.dict_map(info, lambda x: torch.tensor(x, device=self.device))
            self.real_states_np = np.copy(obs)

            # obtain output from model
            obs_py = torch.tensor(obs, dtype=torch.uint8 if self.state_dtype==0 else torch.float32, device=self.device)
            pass_action = torch.zeros(self.env_n, self.raw_dim_actions, dtype=torch.long)
            self.initial_per_state = model_net.initial_state(batch_size=self.env_n, device=self.device)
            model_net_out = model_net(env_state=obs_py, 
                                      done=None,
                                      actions=pass_action.unsqueeze(0).to(self.device), 
                                      state=self.initial_per_state,)  
            self.update_per_state(model_net_out, idx=None)
            vs = model_net_out.vs[-1, :, 0].cpu()
            logits = model_net_out.policy[-1]    
            logits = logits.squeeze(-2)
            logits = logits.cpu().numpy()

            # compute and update root node and current node
            for i in range(self.env_n):
                root_node = node_new(pparent=NULL, action=pass_action[i, 0].item(), logit=0., num_actions=self.num_actions, 
                    discounting=self.discounting, rec_t=self.rec_t, remember_path=True)   
                if i == 0: 
                    self.model_states_keys = [md for md in model_net_out.state.keys() if not md.startswith("per_sr")]

                encoded = {"real_states": obs_py[i],
                           "xs": model_net_out.xs[-1, i] if model_net_out.xs is not None else None,
                           "hs": model_net_out.hs[-1, i] if model_net_out.hs is not None else None,
                           "model_states": tuple(model_net_out.state[md][i] for md in self.model_states_keys),    
                          }  
                node_expand(pnode=root_node, r=0., v=vs[i].item(), t=self.total_step[i], done=False,
                    logits=logits[i], encoded=<PyObject*>encoded, override=False)
                node_visit(pnode=root_node)
                self.root_nodes.push_back(root_node)
                self.cur_nodes.push_back(root_node)
            
            # record initial root_nodes_qmax 
            for i in range(self.env_n):
                self.root_nodes_qmax[i] = self.root_nodes[i][0].max_q
            
            if self.dynamic_search:
                self.reset_dynamic_state()
                states = self.prepare_state_dynamic()
                info = self.prepare_info_dynamic(info, False)
            else:
                states = self.prepare_state(None, None)
                info = self.prepare_info(info, self.status, False)
            return states, info

    def step(self, action, model_net):  
        cdef int i, j, k, l
        cdef int[:] re_action
        cdef int[:] im_action
        cdef int[:] reset

        cdef Node* root_node
        cdef Node* cur_node
        cdef Node* next_node
        cdef vector[Node*] cur_nodes_
        cdef vector[Node*] root_nodes_    
        cdef float[:,:] model_out        

        cdef vector[int] pass_idx_restore
        cdef vector[int] pass_idx_step
        cdef vector[int] pass_idx_reset
        cdef vector[int] pass_idx_reset_
        cdef vector[int] pass_action
        cdef vector[int] pass_model_action

        cdef float[:] vs_1
        cdef float[:,:] logits_1

        cdef float[:] rs_4
        cdef float[:] vs_4
        cdef float[:,:] logits_4

        cdef str md        

        if self.dynamic_search:
            return self.step_dynamic(action, model_net)

        if self.time: self.timings.reset()

        assert type(action) is tuple and len(action) == 2, \
            "action should be a tuple of size 2, containing augmented action and reset"        
        for i in range(len(action)):
            a = action[i]            
            assert torch.is_tensor(a) or \
                isinstance(a, np.ndarray) or \
                isinstance(a, list), \
                f"action[{i}] should be either torch.tensor or np.ndarray or list"            
            if torch.is_tensor(a):
                a = a.detach().cpu().numpy()
            if isinstance(a, list):
                a = np.array(a, dtype=np.int32)
            assert a.shape == (self.env_n,), \
                f"action[{i}] shape should be {(self.env_n,)}, not {a.shape}"
            if a.dtype != np.int32:
                a = a.astype(np.int32)
            if i == 0:
                re_action = a
                im_action = a
                assert (a >= 0).all() and (a < self.num_actions).all(), \
                    f"primary action should be in [0, {self.num_actions-1}], not {a}"
            else:
                reset = a       
                assert (a >= 0).all() and (a < 2).all(), \
                    f"reset action should be in [0, 1], not {a}"  

        pass_model_states = []
        pass_raw_action, pass_model_raw_action = [], []
        for i in range(self.env_n):            
            # compute the mask of real / imagination step                             
            self.max_rollout_depth_[i] = self.max_rollout_depth[i]            
            if self.cur_t[i] < self.rec_t - 1: # imagaination step
                self.cur_t[i] += 1
                self.rollout_depth[i] += 1
                self.max_rollout_depth[i] = max(self.max_rollout_depth[i], self.rollout_depth[i])
                if (
                    (self.max_depth > 0 and self.rollout_depth[i] >= self.max_depth) or 
                    (self.reset_mode == 1 and self.cur_t[i] == self.rec_t - 1)
                ):
                    # force reset if serach depth exceeds max depth or next step is real step (reset_mode 1)
                    reset[i] = 1
                next_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                if self.reset_mode == 1 and reset[i]:
                    self.status[i] = 5 # reset
                elif node_expanded(next_node, self.total_step[i]):
                    self.status[i] = 2 # expanded status
                elif self.cur_nodes[i][0].done:
                    self.status[i] = 3 # done status
                else:
                    encoded = <dict> self.cur_nodes[i][0].encoded
                    pass_model_states.append(tuple(encoded["model_states"]))
                    pass_model_action.push_back(im_action[i])
                    self.status[i] = 4  
            else: # real step
                self.cur_t[i] = 0
                self.rollout_depth[i] = 0          
                self.max_rollout_depth[i] = 0
                self.total_step[i] = self.total_step[i] + 1
                # record baseline before moving on
                self.baseline_mean_q[i] = (average(self.root_nodes[i][0].prollout_qs[0]) -
                    self.root_nodes[i][0].r) / self.discounting
                encoded = <dict> self.root_nodes[i][0].encoded
                pass_idx_restore.push_back(i)
                pass_action.push_back(re_action[i])                                  
                pass_idx_step.push_back(i)
                self.status[i] = 1        
        if self.time: self.timings.time("misc_1")
        # one step of env
        if pass_idx_step.size() > 0:
            pass_action_in = np.array(pass_action)
            obs, reward, done, truncated_done, info = self.env.step(pass_action_in, env_id=pass_idx_step) 
        else:
            info = None

        if self.time: self.timings.time("step_state")
        # env reset needed?
        for i, j in enumerate(pass_idx_step):
            if done[i]:
                pass_idx_reset.push_back(j)                
                pass_idx_reset_.push_back(i) # index within pass_idx_step

        # env reset
        if pass_idx_reset.size() > 0:
            for i, j in enumerate(pass_idx_reset_):                
                pass_action[j] = 0        

        # update real_states_np
        for i, j in enumerate(pass_idx_step):
            self.real_states_np[j] = obs[i]

        # use model for status 1 transition (real transition)
        if pass_idx_step.size() > 0:
            with torch.no_grad():
                obs_py = torch.tensor(obs, dtype=torch.uint8 if self.state_dtype==0 else torch.float32, device=self.device)
                pass_action_py = torch.tensor(pass_action, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(-1)
                done_py = torch.tensor(done, dtype=torch.bool, device=self.device)
                if pass_idx_step.size() == self.env_n:
                    self.initial_per_state = self.per_state
                else:
                    self.initial_per_state = {sk: sv[pass_idx_step] for sk, sv in self.per_state.items()}
                model_net_out_1 = model_net(
                    env_state=obs_py, 
                    done=done_py,
                    actions=pass_action_py, 
                    state=self.initial_per_state,
                )  
                self.update_per_state(model_net_out_1, idx=pass_idx_step)
            vs_1 = model_net_out_1.vs[-1, :, 0].float().cpu().numpy()
            logits_1_ = model_net_out_1.policy[-1].float()
            logits_1_ = logits_1_.squeeze(-2)
            logits_1 = logits_1_.cpu().numpy()

        if self.time: self.timings.time("misc_2")
        # use model for status 4 transition (imagination transition)
        if pass_model_action.size() > 0:
            with torch.no_grad():
                pass_model_states = dict({md: torch.stack([ms[i] for ms in pass_model_states], dim=0)
                        for i, md in enumerate(self.model_states_keys)})
                pass_model_action_py = torch.tensor(pass_model_action, dtype=torch.long, device=self.device).unsqueeze(-1)                
                model_net_out_4 = model_net.forward_single(
                    state=pass_model_states,
                    action=pass_model_action_py)  
            rs_4 = model_net_out_4.rs[-1, :, 0].float().cpu().numpy()
            vs_4 = model_net_out_4.vs[-1, :, 0].float().cpu().numpy()
            logits_4_ = model_net_out_4.policy[-1].float()
            logits_4_ = logits_4_.squeeze(-2)

            logits_4 = logits_4_.cpu().numpy()
            if self.pred_done:
                done_4 = model_net_out_4.dones[-1].bool().cpu().numpy()

        if self.time: self.timings.time("model_unroll_4")

        # compute the current and root nodes
        j = 0 # counter for status 1 transition
        l = 0 # counter for status 4 transition

        for i in range(self.env_n):
            if self.status[i] == 1:
                # real transition
                new_root = (not self.tree_carry or 
                    not node_expanded(self.root_nodes[i][0].ppchildren[0][re_action[i]], -1) or done[j])
                encoded = {"real_states": obs_py[j], 
                           "xs": model_net_out_1.xs[-1, j] if self.return_x else None,
                           "hs": model_net_out_1.hs[-1, j] if self.return_h else None,
                           "model_states": tuple(model_net_out_1.state[md][j] for md in self.model_states_keys),                           
                          }      
                if new_root:
                    root_node = node_new(pparent=NULL, action=pass_action[j], logit=0., num_actions=self.num_actions, 
                        discounting=self.discounting, rec_t=self.rec_t, remember_path=True)                    
                    node_expand(pnode=root_node, r=reward[j], v=vs_1[j], t=self.total_step[i], done=False,
                        logits=logits_1[j], encoded=<PyObject*>encoded, override=False)
                    node_del(self.root_nodes[i], except_idx=-1)
                    node_visit(root_node)
                else:
                    root_node = self.root_nodes[i][0].ppchildren[0][re_action[i]]
                    node_expand(pnode=root_node, r=reward[j], v=vs_1[j], t=self.total_step[i], done=False,
                        logits=logits_1[j], encoded=<PyObject*>encoded, override=True)                        
                    node_del(self.root_nodes[i], except_idx=re_action[i])
                    node_visit(root_node)
                    
                j += 1
                root_nodes_.push_back(root_node)
                cur_nodes_.push_back(root_node) 
            elif self.status[i] == 2:
                # expanded already
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)      

            elif self.status[i] == 3:
                # done already
                for k in range(self.num_actions):
                    self.par_logits[k] = self.cur_nodes[i].ppchildren[0][k][0].logit
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                node_expand(pnode=cur_node, r=0., v=0., t=self.total_step[i], done=True,
                        logits=self.par_logits, encoded=self.cur_nodes[i][0].encoded, override=True)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)        
            
            elif self.status[i] == 4:
                # need expand
                encoded = {"real_states": None, 
                           "xs": model_net_out_4.xs[-1, l] if self.return_x else None,
                           "hs": model_net_out_4.hs[-1, l] if self.return_h else None,
                           "model_states": tuple(model_net_out_4.state[md][l] for md in self.model_states_keys)
                           }
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                if self.pred_done:
                    v_in = vs_4[l] if not done_4[l] else 0.
                    done_in = done_4[l]
                else:
                    v_in = vs_4[l]
                    done_in = False

                node_expand(pnode=cur_node, r=rs_4[l], 
                        v=v_in, t=self.total_step[i], done=done_in,
                        logits=logits_4[l], encoded=<PyObject*>encoded, override=True)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)   
                l += 1                   
            elif self.status[i] == 5:
                # reset
                self.rollout_depth[i] = 0
                cur_node = self.root_nodes[i]
                node_visit(cur_node) 
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node) 

        self.root_nodes = root_nodes_
        self.cur_nodes = cur_nodes_
        if self.time: self.timings.time("compute_root_cur_nodes")      
        # compute states        
        states = self.prepare_state(reset, self.status)
        if self.time: self.timings.time("compute_state")
        # compute reward
        j = 0
        for i in range(self.env_n):
            # real reward
            if self.status[i] == 1:
                self.full_reward[i] = reward[j]
            else:
                self.full_reward[i] = 0.
            # planning reward
            if self.im_enable:                        
                self.root_nodes_qmax_[i] = self.root_nodes[i][0].max_q
                if self.status[i] != 1:                
                    self.full_im_reward[i] = (self.root_nodes_qmax_[i] - self.root_nodes_qmax[i])
                    if self.full_im_reward[i] < 0 or reset[i]: self.full_im_reward[i] = 0
                else:
                    self.full_im_reward[i] = 0.
                self.root_nodes_qmax[i] = self.root_nodes_qmax_[i]
            if self.status[i] == 1:
                j += 1
        if self.time: self.timings.time("compute_reward")        
        
        # some extra info
        info = self.prepare_info(info=info, status=self.status, perfect=False)

        j = 0
        for i in range(self.env_n):
            # compute done
            if self.status[i] == 1:
                self.full_done[i] = done[j] 
                self.full_truncated_done[i] = truncated_done[j]         
                self.full_im_done[i] = False                
                if done[j]:
                    self.step_from_done[i] = 0
                else:
                    self.step_from_done[i] += 1
            else:
                self.full_done[i] = False              
                self.full_truncated_done[i] = False
                self.full_im_done[i] = self.cur_nodes[i][0].done
            if self.status[i] == 1:
                j += 1        
            if self.reset_mode == 0 and reset[i] and self.status[i] != 1:
                # reset after computing the tree rep for reset_mode 0
                self.rollout_depth[i] = 0
                self.cur_nodes[i] = self.root_nodes[i]
                node_visit(self.cur_nodes[i])
                self.status[i] = 5
        
        if self.im_enable:
            info["im_reward"] = torch.tensor(self.full_im_reward, dtype=torch.float, device=self.device).unsqueeze(-1)
        if self.time: self.timings.time("end")

        self.internal_counter += 1
        if self.internal_counter % 200 == 0 and self.time: print(self.timings.summary())

        return (states, 
                torch.tensor(self.full_reward, dtype=torch.float, device=self.device), 
                torch.tensor(self.full_done, dtype=torch.float, device=self.device).bool(), 
                torch.tensor(self.full_truncated_done, dtype=torch.float, device=self.device).bool(), 
                info)

    def step_dynamic(self, action, model_net):
        """Dynamic learned-model search with a synchronous real-step barrier.

        SEARCH consumes the three-way control head.  STOP changes only the
        policy phase; the next call supplies a real action and then waits.  The
        final stored action releases one full-batch real environment step in
        the same call.
        """
        cdef int i, j, k, l, control_i
        cdef int[:] primary
        cdef int[:] control
        cdef int[:] reset
        cdef Node* root_node
        cdef Node* cur_node
        cdef Node* next_node
        cdef vector[Node*] cur_nodes_
        cdef vector[Node*] root_nodes_
        cdef vector[int] pass_model_action
        cdef float[:] vs_1
        cdef float[:,:] logits_1
        cdef float[:] rs_4
        cdef float[:] vs_4
        cdef float[:,:] logits_4
        cdef str md

        if self.time:
            self.timings.reset()
        assert type(action) is tuple and len(action) == 2, \
            "action should be (primary_action, search_control)"

        parsed_action = []
        for i in range(2):
            a = action[i]
            assert torch.is_tensor(a) or isinstance(a, np.ndarray) or isinstance(a, list), \
                f"action[{i}] should be either torch.tensor, np.ndarray, or list"
            if torch.is_tensor(a):
                a = a.detach().cpu().numpy()
            if isinstance(a, list):
                a = np.array(a, dtype=np.int32)
            assert a.shape == (self.env_n,), \
                f"action[{i}] shape should be {(self.env_n,)}, not {a.shape}"
            if a.dtype != np.int32:
                a = a.astype(np.int32)
            parsed_action.append(a)
        primary = parsed_action[0]
        control = parsed_action[1]
        assert (parsed_action[0] >= 0).all() and (parsed_action[0] < self.num_actions).all(), \
            f"primary action should be in [0, {self.num_actions-1}]"
        assert (parsed_action[1] >= 0).all() and (parsed_action[1] < 3).all(), \
            "search control should be PROCEED=0, RESET=1, or STOP=2"

        reset_np = np.zeros(self.env_n, dtype=np.intc)
        reset = reset_np
        self.clear_dynamic_events()
        pass_model_states = []

        # Phase-local policy dispatch.  WAIT inputs are deliberately ignored.
        for i in range(self.env_n):
            self.max_rollout_depth_[i] = self.max_rollout_depth[i]
            if self.phase[i] == 0:  # SEARCH
                control_i = control[i]
                assert not (
                    control_i == 0 and self.max_depth > 0 and
                    self.rollout_depth[i] >= self.max_depth - 1), \
                    "PROCEED is illegal at max_depth; choose RESET or STOP"
                self.accepted_control_event[i] = control_i

                if control_i == 2:  # STOP; no tree/model/environment mutation
                    self.phase[i] = 1
                    self.stage_end_event[i] = True
                    self.status[i] = 6
                    self.step_status[i] = 2
                    continue

                self.accepted_primary_action_event[i] = primary[i]
                self.think_reward_event[i] = -1.
                self.search_steps[i] += 1
                self.cur_t[i] += 1
                self.rollout_depth[i] += 1
                self.max_rollout_depth[i] = max(
                    self.max_rollout_depth[i], self.rollout_depth[i])
                reset[i] = <int>(control_i == 1)
                if reset[i]:
                    self.tree_reset_event[i] = True
                self.tree_token_valid_event[i] = True
                self.max_rollout_depth_[i] = self.max_rollout_depth[i]

                next_node = self.cur_nodes[i][0].ppchildren[0][primary[i]]
                if self.reset_mode == 1 and reset[i]:
                    self.status[i] = 5
                elif node_expanded(next_node, self.total_step[i]):
                    self.status[i] = 2
                elif self.cur_nodes[i][0].done:
                    self.status[i] = 3
                else:
                    encoded = <dict>self.cur_nodes[i][0].encoded
                    pass_model_states.append(tuple(encoded["model_states"]))
                    pass_model_action.push_back(primary[i])
                    self.status[i] = 4
                if (self.max_search_steps > 0 and
                        self.search_steps[i] >= self.max_search_steps):
                    # The Nth computation ends the stage immediately.  There
                    # is no extra STOP policy sample/call at the safety cap.
                    self.phase[i] = 1
                    self.stage_end_event[i] = True
                    self.forced_stop_event[i] = True
                    self.step_status[i] = 2
                else:
                    self.step_status[i] = 1

            elif self.phase[i] == 1:  # NEED_REAL_ACTION
                self.pending_real_action[i] = primary[i]
                self.pending_real_valid[i] = True
                self.phase[i] = 2
                self.stored_action_mask_event[i] = True
                self.accepted_primary_action_event[i] = primary[i]
                self.status[i] = 7
                self.step_status[i] = 4

            else:  # WAIT
                self.status[i] = 8
                self.step_status[i] = 4

        # The call that stores the last missing action also performs the real
        # transition.  Because all phases are now WAIT, this call cannot also
        # contain an imagination transition.
        if self.all_real_actions_stored():
            executed_np = np.array(self.pending_real_action, dtype=np.int32, copy=True)
            model_action_np = executed_np.copy()
            all_env_id = list(range(self.env_n))
            for i in range(self.env_n):
                self.executed_primary_action_event[i] = executed_np[i]
                self.max_rollout_depth_[i] = self.max_rollout_depth[i]
                self.baseline_mean_q[i] = (
                    average(self.root_nodes[i][0].prollout_qs[0]) -
                    self.root_nodes[i][0].r) / self.discounting
                self.total_step[i] += 1

            obs, reward, done, truncated_done, info = self.env.step(
                executed_np, env_id=all_env_id)
            terminal_np = np.logical_or(done, truncated_done)
            model_action_np[terminal_np] = 0
            self.real_states_np[:] = obs

            with torch.no_grad():
                obs_py = torch.tensor(
                    obs,
                    dtype=torch.uint8 if self.state_dtype == 0 else torch.float32,
                    device=self.device)
                model_action_py = torch.tensor(
                    model_action_np, dtype=torch.long, device=self.device
                ).unsqueeze(0).unsqueeze(-1)
                done_py = torch.tensor(terminal_np, dtype=torch.bool, device=self.device)
                self.initial_per_state = self.per_state
                model_net_out_1 = model_net(
                    env_state=obs_py,
                    done=done_py,
                    actions=model_action_py,
                    state=self.initial_per_state)
                self.update_per_state(model_net_out_1, idx=None)
            vs_1 = model_net_out_1.vs[-1, :, 0].float().cpu().numpy()
            logits_1_ = model_net_out_1.policy[-1].float().squeeze(-2)
            logits_1 = logits_1_.cpu().numpy()

            for i in range(self.env_n):
                new_root = (
                    not self.tree_carry or
                    not node_expanded(
                        self.root_nodes[i][0].ppchildren[0][executed_np[i]], -1) or
                    terminal_np[i])
                encoded = {
                    "real_states": obs_py[i],
                    "xs": model_net_out_1.xs[-1, i] if self.return_x else None,
                    "hs": model_net_out_1.hs[-1, i] if self.return_h else None,
                    "model_states": tuple(
                        model_net_out_1.state[md][i] for md in self.model_states_keys),
                }
                if new_root:
                    root_node = node_new(
                        pparent=NULL, action=model_action_np[i], logit=0.,
                        num_actions=self.num_actions,
                        discounting=self.discounting, rec_t=self.rec_t,
                        remember_path=True)
                    node_expand(
                        pnode=root_node, r=reward[i], v=vs_1[i],
                        t=self.total_step[i], done=False, logits=logits_1[i],
                        encoded=<PyObject*>encoded, override=False)
                    node_del(self.root_nodes[i], except_idx=-1)
                else:
                    root_node = self.root_nodes[i][0].ppchildren[0][executed_np[i]]
                    node_expand(
                        pnode=root_node, r=reward[i], v=vs_1[i],
                        t=self.total_step[i], done=False, logits=logits_1[i],
                        encoded=<PyObject*>encoded, override=True)
                    node_del(self.root_nodes[i], except_idx=executed_np[i])
                node_visit(root_node)
                root_nodes_.push_back(root_node)
                cur_nodes_.push_back(root_node)

                self.status[i] = 1
                self.phase[i] = 0
                self.search_steps[i] = 0
                self.cur_t[i] = 0
                self.rollout_depth[i] = 0
                self.max_rollout_depth[i] = 0
                self.pending_real_valid[i] = False
                self.pending_real_action[i] = 0
                self.tree_reset_event[i] = True
                self.tree_token_valid_event[i] = True
                self.search_state_reset_event[i] = True
                self.real_transition_event[i] = True
                self.barrier_released_event[i] = True
                self.step_status[i] = 0
                self.full_reward[i] = reward[i]
                self.full_done[i] = done[i]
                self.full_truncated_done[i] = truncated_done[i]
                self.full_im_done[i] = False
                if terminal_np[i]:
                    self.step_from_done[i] = 0
                else:
                    self.step_from_done[i] += 1

            self.root_nodes = root_nodes_
            self.cur_nodes = cur_nodes_
            for i in range(self.env_n):
                self.root_nodes_qmax[i] = self.root_nodes[i][0].max_q
                self.root_nodes_qmax_[i] = self.root_nodes_qmax[i]
                self.full_im_reward[i] = 0.

            states = self.prepare_state_dynamic()
            info = self.prepare_info_dynamic(info, False)
            if self.im_enable:
                info["im_reward"] = torch.zeros(
                    (self.env_n, 1), dtype=torch.float, device=self.device)
            return (
                states,
                torch.tensor(self.full_reward, dtype=torch.float, device=self.device),
                torch.tensor(self.full_done, dtype=torch.bool, device=self.device),
                torch.tensor(self.full_truncated_done, dtype=torch.bool, device=self.device),
                info)

        # No barrier: only unexpanded SEARCH nodes require a learned-model call.
        if pass_model_action.size() > 0:
            with torch.no_grad():
                pass_model_states = dict({
                    md: torch.stack([ms[i] for ms in pass_model_states], dim=0)
                    for i, md in enumerate(self.model_states_keys)})
                pass_model_action_py = torch.tensor(
                    pass_model_action, dtype=torch.long, device=self.device
                ).unsqueeze(-1)
                model_net_out_4 = model_net.forward_single(
                    state=pass_model_states, action=pass_model_action_py)
            rs_4 = model_net_out_4.rs[-1, :, 0].float().cpu().numpy()
            vs_4 = model_net_out_4.vs[-1, :, 0].float().cpu().numpy()
            logits_4_ = model_net_out_4.policy[-1].float().squeeze(-2)
            logits_4 = logits_4_.cpu().numpy()
            if self.pred_done:
                done_4 = model_net_out_4.dones[-1].bool().cpu().numpy()

        l = 0
        for i in range(self.env_n):
            if self.status[i] == 2:
                cur_node = self.cur_nodes[i][0].ppchildren[0][primary[i]]
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)
            elif self.status[i] == 3:
                for k in range(self.num_actions):
                    self.par_logits[k] = self.cur_nodes[i][0].ppchildren[0][k][0].logit
                cur_node = self.cur_nodes[i][0].ppchildren[0][primary[i]]
                node_expand(
                    pnode=cur_node, r=0., v=0., t=self.total_step[i],
                    done=True, logits=self.par_logits,
                    encoded=self.cur_nodes[i][0].encoded, override=True)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)
            elif self.status[i] == 4:
                encoded = {
                    "real_states": None,
                    "xs": model_net_out_4.xs[-1, l] if self.return_x else None,
                    "hs": model_net_out_4.hs[-1, l] if self.return_h else None,
                    "model_states": tuple(
                        model_net_out_4.state[md][l] for md in self.model_states_keys),
                }
                cur_node = self.cur_nodes[i][0].ppchildren[0][primary[i]]
                if self.pred_done:
                    v_in = vs_4[l] if not done_4[l] else 0.
                    done_in = done_4[l]
                else:
                    v_in = vs_4[l]
                    done_in = False
                node_expand(
                    pnode=cur_node, r=rs_4[l], v=v_in,
                    t=self.total_step[i], done=done_in, logits=logits_4[l],
                    encoded=<PyObject*>encoded, override=True)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)
                l += 1
            elif self.status[i] == 5:
                self.rollout_depth[i] = 0
                cur_node = self.root_nodes[i]
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)
            else:  # STOP, stored real action, or WAIT: freeze tree/current node
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(self.cur_nodes[i])

        self.root_nodes = root_nodes_
        self.cur_nodes = cur_nodes_
        states = self.prepare_state_dynamic()

        for i in range(self.env_n):
            self.full_reward[i] = 0.
            self.full_done[i] = False
            self.full_truncated_done[i] = False
            self.full_im_done[i] = self.cur_nodes[i][0].done
            if self.im_enable:
                self.root_nodes_qmax_[i] = self.root_nodes[i][0].max_q
                if self.status[i] in [2, 3, 4, 5]:
                    self.full_im_reward[i] = (
                        self.root_nodes_qmax_[i] - self.root_nodes_qmax[i])
                    if self.full_im_reward[i] < 0 or reset[i]:
                        self.full_im_reward[i] = 0.
                else:
                    self.full_im_reward[i] = 0.
                self.root_nodes_qmax[i] = self.root_nodes_qmax_[i]

        # reset_mode=0 exposes the just-completed rollout token, then makes the
        # root current for the next call (matching the original Thinker order).
        for i in range(self.env_n):
            if (self.reset_mode == 0 and reset[i] and
                    self.status[i] in [2, 3, 4]):
                self.rollout_depth[i] = 0
                self.cur_nodes[i] = self.root_nodes[i]
                node_visit(self.cur_nodes[i])
                self.status[i] = 5

        # Phase/legal masks describe the state accepted by the *next* call,
        # whereas the tree token above intentionally describes the completed
        # rollout before its reset.
        info = self.prepare_info_dynamic(None, False)
        if self.im_enable:
            info["im_reward"] = torch.tensor(
                self.full_im_reward, dtype=torch.float, device=self.device
            ).unsqueeze(-1)

        self.internal_counter += 1
        return (
            states,
            torch.tensor(self.full_reward, dtype=torch.float, device=self.device),
            torch.tensor(self.full_done, dtype=torch.bool, device=self.device),
            torch.tensor(self.full_truncated_done, dtype=torch.bool, device=self.device),
            info)

cdef class cPerfectWrapper(cWrapper):
    """Wrap the gym environment with a perfect model (i.e. env that supports quick_save
    and quick_load); output for each step is (out, reward, done, truncated_done, info), where out is a dict 
    that consists tree rep, environment frame, output from the model wrapper, and encoding from the model
    Assume a perfect dynamic model.
    """
        
    def reset(self, model_net, seed=None):
        """Reset the environment and replace any previously active tree."""
        cdef int i, slot_i, old_slot_limit
        cdef Node* root_node
        cdef Node* cur_node      

        # See cModelWrapper.reset: a new reset owns a new forest and must not
        # retain either roots or current-node aliases from the prior episode.
        # No carried subtree survives, so its perfect-model snapshots must not
        # survive either (slot zero is overwritten with the new real root).
        old_slot_limit = 1
        if self.dynamic_search:
            for i in range(self.env_n):
                old_slot_limit = max(old_slot_limit, self.next_snapshot_slot[i])
            if hasattr(self.env, "quick_delete"):
                for slot_i in range(1, old_slot_limit):
                    self.env.quick_delete(slot_id=slot_i)
        self.dispose_trees()

        with torch.no_grad():
            # some init.
            self.root_nodes_qmax = np.zeros(self.env_n, dtype=np.float32)
            self.root_nodes_qmax_ = np.zeros(self.env_n, dtype=np.float32)
            self.rollout_depth = np.zeros(self.env_n, dtype=np.intc)
            self.max_rollout_depth = np.zeros(self.env_n, dtype=np.intc)
            self.cur_t = np.zeros(self.env_n, dtype=np.intc)
            self.baseline_mean_q = torch.zeros(self.env_n, dtype=torch.float, device=self.device)        

            # reset obs
            obs, info = self.env.reset(reset_stat=True, seed=seed)
            self.default_info = util.dict_map(info, lambda x: torch.tensor(x, device=self.device))
            self.real_states_np = np.copy(obs)

            # obtain output from model
            obs_py = torch.tensor(obs, dtype=torch.uint8 if self.state_dtype==0 else torch.float32, device=self.device)
            pass_action = torch.zeros(self.env_n, self.raw_dim_actions, dtype=torch.long)
            model_net_out = model_net(env_state=obs_py, 
                                      done=None,
                                      actions=pass_action.unsqueeze(0).to(self.device), 
                                      state=None,)  
            vs = model_net_out.vs[:, :, 0].cpu()
            logits = model_net_out.policy[-1]    
            logits = logits.squeeze(-2)
            logits = logits.cpu().numpy()
            self.env.quick_save()

            # compute and update root node and current node
            for i in range(self.env_n):
                root_node = node_new(pparent=NULL, action=pass_action[i, 0].item(), logit=0., num_actions=self.num_actions, 
                    discounting=self.discounting, rec_t=self.rec_t, remember_path=False)                
                encoded = {"real_states": obs_py[i],
                           "xs": model_net_out.xs[-1, i] if model_net_out.xs is not None else None,
                           "hs": model_net_out.hs[-1, i] if model_net_out.hs is not None else None,
                           }
                if self.dynamic_search:
                    encoded["snapshot_slot"] = 0
                node_expand(pnode=root_node, r=0., v=vs[-1, i].item(), t=self.total_step[i], done=False,
                    logits=logits[i], encoded=<PyObject*>encoded, override=False)
                node_visit(pnode=root_node)
                self.root_nodes.push_back(root_node)
                self.cur_nodes.push_back(root_node)
            
            # record initial root_nodes_qmax 
            for i in range(self.env_n):
                self.root_nodes_qmax[i] = self.root_nodes[i][0].max_q

            if self.dynamic_search:
                self.reset_dynamic_state()
                states = self.prepare_state_dynamic()
                info = self.prepare_info_dynamic(info, True)
            else:
                states = self.prepare_state(None, None)
                info = self.prepare_info(info, self.status, True)
            return states, info

    def step(self, action, model_net):  
        cdef int i, j, k
        cdef int[:] re_action
        cdef int[:] im_action
        cdef int[:] reset

        cdef Node* root_node
        cdef Node* cur_node
        cdef Node* next_node
        cdef vector[Node*] cur_nodes_
        cdef vector[Node*] root_nodes_          

        cdef vector[int] pass_idx_restore
        cdef vector[int] pass_idx_step
        cdef vector[int] pass_idx_reset
        cdef vector[int] pass_idx_reset_
        cdef vector[int] pass_action

        cdef float[:] vs
        cdef float[:,:] logits

        if self.dynamic_search:
            return self.step_dynamic(action, model_net)

        if self.time: self.timings.reset()        

        assert type(action) is tuple and len(action) == 2, \
            "action should be a tuple of size 2, containing augmented action and reset"        
        for i in range(len(action)):
            a = action[i]            
            assert torch.is_tensor(a) or \
                isinstance(a, np.ndarray) or \
                isinstance(a, list), \
                f"action[{i}] should be either torch.tensor or np.ndarray or list"            
            if torch.is_tensor(a):
                a = a.detach().cpu().numpy()
            if isinstance(a, list):
                a = np.array(a, dtype=np.int32)
            assert a.shape == (self.env_n,), \
                f"action[{i}] shape should be {(self.env_n,)}, not {a.shape}"
            if a.dtype != np.int32:
                a = a.astype(np.int32)
            if i == 0:
                re_action = a
                im_action = a
                assert (a >= 0).all() and (a < self.num_actions).all(), \
                    f"primiary action should be in [0, {self.num_actions-1}], not {a}"
            else:
                reset = a       
                assert (a >= 0).all() and (a < 2).all(), \
                    f"reset action should be in [0, 1], not {a}"  

        for i in range(self.env_n):            
            # compute the mask of real / imagination step                             
            self.max_rollout_depth_[i] = self.max_rollout_depth[i]
            if self.cur_t[i] < self.rec_t - 1: # imagaination step
                self.cur_t[i] += 1
                self.rollout_depth[i] += 1
                self.max_rollout_depth[i] = max(self.max_rollout_depth[i], self.rollout_depth[i])                
                if (
                    (self.max_depth > 0 and self.rollout_depth[i] >= self.max_depth) or 
                    (self.reset_mode == 1 and self.cur_t[i] == self.rec_t - 1)
                ):
                    # force reset if serach depth exceeds max depth or next step is real step (reset_mode 1)
                    reset[i] = 1
                next_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                if self.reset_mode == 1 and reset[i]:
                    self.status[i] = 5 # reset
                elif node_expanded(next_node, -1):
                    self.status[i] = 2 # expanded status
                elif self.cur_nodes[i][0].done:
                    self.status[i] = 3 # done status
                else:
                    if self.status[i] != 0 or self.status[i] != 4: # no need restore if last step is real or just expanded
                        encoded = <dict> self.cur_nodes[i][0].encoded
                        pass_idx_restore.push_back(i)
                        pass_action.push_back(im_action[i])
                        pass_idx_step.push_back(i)
                    self.status[i] = 4  
            else: # real step
                self.cur_t[i] = 0
                self.rollout_depth[i] = 0          
                self.max_rollout_depth[i] = 0
                self.total_step[i] = self.total_step[i] + 1
                # record baseline before moving on
                self.baseline_mean_q[i] = (average(self.root_nodes[i][0].prollout_qs[0]) -
                    self.root_nodes[i][0].r) / self.discounting                
                encoded = <dict> self.root_nodes[i][0].encoded
                pass_idx_restore.push_back(i)
                pass_action.push_back(re_action[i])
                pass_idx_step.push_back(i)
                self.status[i] = 1                            
        if self.time: self.timings.time("misc_1")

        # restore env      
        if pass_idx_restore.size() > 0:
            self.env.quick_load(env_id=pass_idx_restore)

        # one step of env
        if pass_idx_step.size() > 0:
            obs, reward, done, truncated_done, info = self.env.step(np.array(pass_action), env_id=pass_idx_step) 
        else:
            info = None
        if self.time: self.timings.time("step_state")

        # env reset needed?
        for i, j in enumerate(pass_idx_step):
            if self.status[j] == 1 and done[i]:
                pass_idx_reset.push_back(j)
                pass_idx_reset_.push_back(i) # index within pass_idx_step

        # reset
        if pass_idx_reset.size() > 0:
            for i, j in enumerate(pass_idx_reset_):
                pass_action[j] = 0            
        if self.time: self.timings.time("misc_2")

        # update real_states_np
        for i, j in enumerate(pass_idx_step):
            if self.status[j] == 1:
                self.real_states_np[j] = obs[i]

        # use model
        if pass_idx_step.size() > 0:
            with torch.no_grad():
                obs_py = torch.tensor(obs, dtype=torch.uint8 if self.state_dtype==0 else torch.float32, device=self.device)
                pass_action_py = torch.tensor(pass_action, dtype=torch.long, device=self.device).unsqueeze(-1).unsqueeze(0)
                done_py = torch.tensor(done, dtype=torch.bool, device=self.device)
                model_net_out = model_net(env_state=obs_py, 
                                          done=done_py,
                                          actions=pass_action_py, 
                                          state=None,)  
            vs = model_net_out.vs[-1, :, 0].float().cpu().numpy()
            logits_ = model_net_out.policy[-1].float()
            logits_ = logits_.squeeze(-2)
            logits = logits_.cpu().numpy()
            if self.time: self.timings.time("model")
            self.env.quick_save(env_id=pass_idx_step)   
            if self.time: self.timings.time("quick_save")        

        # compute the current and root nodes
        j = 0
        for i in range(self.env_n):
            if self.status[i] == 1:
                # real transition
                new_root = (not self.tree_carry or 
                    not node_expanded(self.root_nodes[i][0].ppchildren[0][re_action[i]], -1) or done[j])
                if new_root:
                    root_node = node_new(pparent=NULL, action=pass_action[j], logit=0., num_actions=self.num_actions, 
                        discounting=self.discounting, rec_t=self.rec_t, remember_path=True)
                else:
                    root_node = self.root_nodes[i][0].ppchildren[0][re_action[i]]
                encoded = {"real_states": obs_py[j],
                            "xs": model_net_out.xs[-1, j] if model_net_out.xs is not None else None,
                            "hs": model_net_out.hs[-1, j] if model_net_out.hs is not None else None,
                            }
                node_expand(pnode=root_node, r=reward[j], v=vs[j], t=self.total_step[i], done=False,
                    logits=logits[j], encoded=<PyObject*>encoded, override=not new_root)
                except_idx = -1 if new_root else re_action[i]
                node_del(self.root_nodes[i], except_idx=except_idx)
                node_visit(root_node)                    
                j += 1
                root_nodes_.push_back(root_node)
                cur_nodes_.push_back(root_node)

            elif self.status[i] == 2:
                # expanded already
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)    

            elif self.status[i] == 3:
                # done already
                for k in range(self.num_actions):
                    self.par_logits[k] = self.cur_nodes[i].ppchildren[0][k][0].logit
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                node_expand(pnode=cur_node, r=0., v=0., t=self.total_step[i], done=True,
                        logits=self.par_logits, encoded=self.cur_nodes[i][0].encoded, override=False)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)              
            
            elif self.status[i] == 4:
                # need expand
                encoded = {"real_states": None,
                            "xs": model_net_out.xs[-1, j] if model_net_out.xs is not None else None,
                            "hs": model_net_out.hs[-1, j] if model_net_out.hs is not None else None,
                            }
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                r = reward[j]
                node_expand(pnode=cur_node, r=r, v=vs[j] if not done[j] else 0., t=self.total_step[i], done=done[j],
                        logits=logits[j], encoded=<PyObject*>encoded, override=False)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)   
                j += 1                            

            elif self.status[i] == 5:
                # reset
                self.rollout_depth[i] = 0
                cur_node = self.root_nodes[i]
                node_visit(cur_node) 
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node) 

        self.root_nodes = root_nodes_
        self.cur_nodes = cur_nodes_
        if self.time: self.timings.time("compute_root_cur_nodes")

        # compute states        
        states = self.prepare_state(reset, self.status)
        if self.time: self.timings.time("compute_state")
        
        # compute reward
        j = 0
        for i in range(self.env_n):
            if self.status[i] == 1:
                self.full_reward[i] = reward[j]
            else:
                self.full_reward[i] = 0.
            if self.im_enable:                        
                self.root_nodes_qmax_[i] = self.root_nodes[i][0].max_q
                if self.status[i] != 1:                                
                    self.full_im_reward[i] = (self.root_nodes_qmax_[i] - self.root_nodes_qmax[i])
                    if self.full_im_reward[i] < 0 or reset[i]: self.full_im_reward[i] = 0
                else:
                    self.full_im_reward[i] = 0.
                self.root_nodes_qmax[i] = self.root_nodes_qmax_[i]
            if self.status[i] == 1 or self.status[i] == 4:
                j += 1
        if self.time: self.timings.time("compute_reward")
        
        # some extra info
        info = self.prepare_info(info=info, status=self.status, perfect=True)
        
        j = 0
        for i in range(self.env_n):
            # compute done
            if self.status[i] == 1:
                self.full_done[i] = done[j]
                self.full_truncated_done[i] = truncated_done[j]
                self.full_im_done[i] = False
            else:
                self.full_done[i] = False
                self.full_truncated_done[i] = False
                self.full_im_done[i] = self.cur_nodes[i][0].done
            if self.status[i] == 1 or self.status[i] == 4:
                j += 1
            if self.reset_mode == 0 and reset[i] and self.status[i] != 1:
                # reset after computing the tree rep for reset_mode 0
                self.rollout_depth[i] = 0
                self.cur_nodes[i] = self.root_nodes[i]
                node_visit(self.cur_nodes[i])
                self.status[i] = 5
            if self.cur_t[i] == 0:
                self.step_status[i] = 0 # real action just taken
            elif self.rec_t <= 1:
                self.step_status[i] = 3 # real action just taken and next action is real action
            elif self.cur_t[i] < self.rec_t - 1:
                self.step_status[i] = 1 # im action just taken
            elif self.cur_t[i] >= self.rec_t - 1:
                self.step_status[i] = 2 # im action just taken; next action is real action
        
        if self.im_enable:
            info["im_reward"] = torch.tensor(self.full_im_reward, dtype=torch.float, device=self.device).unsqueeze(-1)
        if self.time: self.timings.time("end")

        self.internal_counter += 1
        if self.internal_counter % 500 == 0 and self.time: print(self.timings.summary())
 
        return (states, 
                torch.tensor(self.full_reward, dtype=torch.float, device=self.device), 
                torch.tensor(self.full_done, dtype=torch.float, device=self.device).bool(), 
                torch.tensor(self.full_truncated_done, dtype=torch.float, device=self.device).bool(), 
                info)

    def step_dynamic(self, action, model_net):
        """Dynamic perfect-model search using per-node snapshot slots."""
        cdef int i, j, k, control_i, slot_i, old_slot_limit
        cdef int[:] primary
        cdef int[:] control
        cdef int[:] reset
        cdef Node* root_node
        cdef Node* cur_node
        cdef Node* next_node
        cdef vector[Node*] cur_nodes_
        cdef vector[Node*] root_nodes_
        cdef float[:] vs
        cdef float[:,:] logits

        if self.time:
            self.timings.reset()
        assert type(action) is tuple and len(action) == 2, \
            "action should be (primary_action, search_control)"

        parsed_action = []
        for i in range(2):
            a = action[i]
            assert torch.is_tensor(a) or isinstance(a, np.ndarray) or isinstance(a, list), \
                f"action[{i}] should be either torch.tensor, np.ndarray, or list"
            if torch.is_tensor(a):
                a = a.detach().cpu().numpy()
            if isinstance(a, list):
                a = np.array(a, dtype=np.int32)
            assert a.shape == (self.env_n,), \
                f"action[{i}] shape should be {(self.env_n,)}, not {a.shape}"
            if a.dtype != np.int32:
                a = a.astype(np.int32)
            parsed_action.append(a)
        primary = parsed_action[0]
        control = parsed_action[1]
        assert (parsed_action[0] >= 0).all() and (parsed_action[0] < self.num_actions).all(), \
            f"primary action should be in [0, {self.num_actions-1}]"
        assert (parsed_action[1] >= 0).all() and (parsed_action[1] < 3).all(), \
            "search control should be PROCEED=0, RESET=1, or STOP=2"

        reset_np = np.zeros(self.env_n, dtype=np.intc)
        reset = reset_np
        self.clear_dynamic_events()
        pass_idx_step = []
        pass_action = []
        pass_restore_slots = []

        for i in range(self.env_n):
            self.max_rollout_depth_[i] = self.max_rollout_depth[i]
            if self.phase[i] == 0:  # SEARCH
                control_i = control[i]
                assert not (
                    control_i == 0 and self.max_depth > 0 and
                    self.rollout_depth[i] >= self.max_depth - 1), \
                    "PROCEED is illegal at max_depth; choose RESET or STOP"
                self.accepted_control_event[i] = control_i

                if control_i == 2:
                    self.phase[i] = 1
                    self.stage_end_event[i] = True
                    self.status[i] = 6
                    self.step_status[i] = 2
                    continue

                self.accepted_primary_action_event[i] = primary[i]
                self.think_reward_event[i] = -1.
                self.search_steps[i] += 1
                self.cur_t[i] += 1
                self.rollout_depth[i] += 1
                self.max_rollout_depth[i] = max(
                    self.max_rollout_depth[i], self.rollout_depth[i])
                reset[i] = <int>(control_i == 1)
                if reset[i]:
                    self.tree_reset_event[i] = True
                self.tree_token_valid_event[i] = True
                self.max_rollout_depth_[i] = self.max_rollout_depth[i]

                next_node = self.cur_nodes[i][0].ppchildren[0][primary[i]]
                if self.reset_mode == 1 and reset[i]:
                    self.status[i] = 5
                elif node_expanded(next_node, -1):
                    self.status[i] = 2
                elif self.cur_nodes[i][0].done:
                    self.status[i] = 3
                else:
                    encoded = <dict>self.cur_nodes[i][0].encoded
                    pass_idx_step.append(i)
                    pass_action.append(primary[i])
                    pass_restore_slots.append(int(encoded["snapshot_slot"]))
                    self.status[i] = 4
                if (self.max_search_steps > 0 and
                        self.search_steps[i] >= self.max_search_steps):
                    self.phase[i] = 1
                    self.stage_end_event[i] = True
                    self.forced_stop_event[i] = True
                    self.step_status[i] = 2
                else:
                    self.step_status[i] = 1

            elif self.phase[i] == 1:  # NEED_REAL_ACTION
                self.pending_real_action[i] = primary[i]
                self.pending_real_valid[i] = True
                self.phase[i] = 2
                self.stored_action_mask_event[i] = True
                self.accepted_primary_action_event[i] = primary[i]
                self.status[i] = 7
                self.step_status[i] = 4
            else:
                self.status[i] = 8
                self.step_status[i] = 4

        if self.all_real_actions_stored():
            executed_np = np.array(self.pending_real_action, dtype=np.int32, copy=True)
            model_action_np = executed_np.copy()
            all_env_id = list(range(self.env_n))
            preserved_snapshot_slots = []
            preserved_snapshot_slots_global = {0}
            old_slot_limit = 1
            for i in range(self.env_n):
                old_slot_limit = max(old_slot_limit, self.next_snapshot_slot[i])
                self.executed_primary_action_event[i] = executed_np[i]
                self.max_rollout_depth_[i] = self.max_rollout_depth[i]
                self.baseline_mean_q[i] = (
                    average(self.root_nodes[i][0].prollout_qs[0]) -
                    self.root_nodes[i][0].r) / self.discounting
                self.total_step[i] += 1

            # Slot zero is the immutable real root throughout a search stage.
            self.env.quick_load(slot_id=0)
            obs, reward, done, truncated_done, info = self.env.step(
                executed_np, env_id=all_env_id)
            terminal_np = np.logical_or(done, truncated_done)
            model_action_np[terminal_np] = 0
            self.real_states_np[:] = obs

            with torch.no_grad():
                obs_py = torch.tensor(
                    obs,
                    dtype=torch.uint8 if self.state_dtype == 0 else torch.float32,
                    device=self.device)
                model_action_py = torch.tensor(
                    model_action_np, dtype=torch.long, device=self.device
                ).unsqueeze(0).unsqueeze(-1)
                done_py = torch.tensor(terminal_np, dtype=torch.bool, device=self.device)
                model_net_out = model_net(
                    env_state=obs_py, done=done_py,
                    actions=model_action_py, state=None)
            vs = model_net_out.vs[-1, :, 0].float().cpu().numpy()
            logits_ = model_net_out.policy[-1].float().squeeze(-2)
            logits = logits_.cpu().numpy()

            # The post-real state becomes next stage's immutable root.
            self.env.quick_save(slot_id=0)

            for i in range(self.env_n):
                new_root = (
                    not self.tree_carry or
                    not node_expanded(
                        self.root_nodes[i][0].ppchildren[0][executed_np[i]], -1) or
                    terminal_np[i])
                if new_root:
                    root_node = node_new(
                        pparent=NULL, action=model_action_np[i], logit=0.,
                        num_actions=self.num_actions,
                        discounting=self.discounting, rec_t=self.rec_t,
                        remember_path=False)
                else:
                    root_node = self.root_nodes[i][0].ppchildren[0][executed_np[i]]
                encoded = {
                    "real_states": obs_py[i],
                    "xs": model_net_out.xs[-1, i] if model_net_out.xs is not None else None,
                    "hs": model_net_out.hs[-1, i] if model_net_out.hs is not None else None,
                    "snapshot_slot": 0,
                }
                node_expand(
                    pnode=root_node, r=reward[i], v=vs[i],
                    t=self.total_step[i], done=False, logits=logits[i],
                    encoded=<PyObject*>encoded, override=not new_root)
                node_del(
                    self.root_nodes[i],
                    except_idx=-1 if new_root else executed_np[i])
                env_snapshot_slots = set()
                node_collect_snapshot_slots(root_node, env_snapshot_slots)
                env_snapshot_slots.add(0)
                preserved_snapshot_slots.append(env_snapshot_slots)
                preserved_snapshot_slots_global.update(env_snapshot_slots)
                node_visit(root_node)
                root_nodes_.push_back(root_node)
                cur_nodes_.push_back(root_node)

                self.status[i] = 1
                self.phase[i] = 0
                self.search_steps[i] = 0
                self.cur_t[i] = 0
                self.rollout_depth[i] = 0
                self.max_rollout_depth[i] = 0
                self.pending_real_valid[i] = False
                self.pending_real_action[i] = 0
                self.tree_reset_event[i] = True
                self.tree_token_valid_event[i] = True
                self.search_state_reset_event[i] = True
                self.real_transition_event[i] = True
                self.barrier_released_event[i] = True
                self.step_status[i] = 0
                self.full_reward[i] = reward[i]
                self.full_done[i] = done[i]
                self.full_truncated_done[i] = truncated_done[i]
                self.full_im_done[i] = False

            self.root_nodes = root_nodes_
            self.cur_nodes = cur_nodes_
            for i in range(self.env_n):
                # Allocate above every slot still reachable by this env's
                # carried subtree.  A numeric slot may be live in a different
                # env, because VectorWrap stores per-slot arrays for the batch.
                self.next_snapshot_slot[i] = max(preserved_snapshot_slots[i]) + 1
                self.root_nodes_qmax[i] = self.root_nodes[i][0].max_q
                self.root_nodes_qmax_[i] = self.root_nodes_qmax[i]
                self.full_im_reward[i] = 0.

            # Delete only slots unused by *all* carried subtrees; targeted
            # deletion would otherwise pop a shared VectorWrap slot needed by
            # another environment.
            if hasattr(self.env, "quick_delete"):
                for slot_i in range(1, old_slot_limit):
                    if slot_i not in preserved_snapshot_slots_global:
                        self.env.quick_delete(slot_id=slot_i)

            states = self.prepare_state_dynamic()
            info = self.prepare_info_dynamic(info, True)
            if self.im_enable:
                info["im_reward"] = torch.zeros(
                    (self.env_n, 1), dtype=torch.float, device=self.device)
            return (
                states,
                torch.tensor(self.full_reward, dtype=torch.float, device=self.device),
                torch.tensor(self.full_done, dtype=torch.bool, device=self.device),
                torch.tensor(self.full_truncated_done, dtype=torch.bool, device=self.device),
                info)

        # Restore the exact parent snapshot for every unexpanded node, step the
        # selected subset, then give each resulting child a stable slot.
        if len(pass_idx_step) > 0:
            self.env.quick_load_slots(
                env_id=pass_idx_step, slot_ids=pass_restore_slots)
            obs, reward, done, truncated_done, _ = self.env.step(
                np.array(pass_action, dtype=np.int32), env_id=pass_idx_step)
            terminal_np = np.logical_or(done, truncated_done)
            model_action_np = np.array(pass_action, dtype=np.int32)
            model_action_np[terminal_np] = 0
            new_snapshot_slots = []
            for i in pass_idx_step:
                new_snapshot_slots.append(self.next_snapshot_slot[i])
                self.next_snapshot_slot[i] += 1
            self.env.quick_save_slots(
                env_id=pass_idx_step, slot_ids=new_snapshot_slots)

            with torch.no_grad():
                obs_py = torch.tensor(
                    obs,
                    dtype=torch.uint8 if self.state_dtype == 0 else torch.float32,
                    device=self.device)
                model_action_py = torch.tensor(
                    model_action_np, dtype=torch.long, device=self.device
                ).unsqueeze(-1).unsqueeze(0)
                done_py = torch.tensor(terminal_np, dtype=torch.bool, device=self.device)
                model_net_out = model_net(
                    env_state=obs_py, done=done_py,
                    actions=model_action_py, state=None)
            vs = model_net_out.vs[-1, :, 0].float().cpu().numpy()
            logits_ = model_net_out.policy[-1].float().squeeze(-2)
            logits = logits_.cpu().numpy()

        j = 0
        for i in range(self.env_n):
            if self.status[i] == 2:
                cur_node = self.cur_nodes[i][0].ppchildren[0][primary[i]]
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)
            elif self.status[i] == 3:
                for k in range(self.num_actions):
                    self.par_logits[k] = self.cur_nodes[i][0].ppchildren[0][k][0].logit
                cur_node = self.cur_nodes[i][0].ppchildren[0][primary[i]]
                node_expand(
                    pnode=cur_node, r=0., v=0., t=self.total_step[i],
                    done=True, logits=self.par_logits,
                    encoded=self.cur_nodes[i][0].encoded, override=False)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)
            elif self.status[i] == 4:
                encoded = {
                    "real_states": None,
                    "xs": model_net_out.xs[-1, j] if model_net_out.xs is not None else None,
                    "hs": model_net_out.hs[-1, j] if model_net_out.hs is not None else None,
                    "snapshot_slot": new_snapshot_slots[j],
                }
                cur_node = self.cur_nodes[i][0].ppchildren[0][primary[i]]
                node_expand(
                    pnode=cur_node, r=reward[j],
                    v=vs[j] if not terminal_np[j] else 0.,
                    t=self.total_step[i], done=terminal_np[j], logits=logits[j],
                    encoded=<PyObject*>encoded, override=False)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)
                j += 1
            elif self.status[i] == 5:
                self.rollout_depth[i] = 0
                cur_node = self.root_nodes[i]
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)
            else:
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(self.cur_nodes[i])

        self.root_nodes = root_nodes_
        self.cur_nodes = cur_nodes_
        states = self.prepare_state_dynamic()

        for i in range(self.env_n):
            self.full_reward[i] = 0.
            self.full_done[i] = False
            self.full_truncated_done[i] = False
            self.full_im_done[i] = self.cur_nodes[i][0].done
            if self.im_enable:
                self.root_nodes_qmax_[i] = self.root_nodes[i][0].max_q
                if self.status[i] in [2, 3, 4, 5]:
                    self.full_im_reward[i] = (
                        self.root_nodes_qmax_[i] - self.root_nodes_qmax[i])
                    if self.full_im_reward[i] < 0 or reset[i]:
                        self.full_im_reward[i] = 0.
                else:
                    self.full_im_reward[i] = 0.
                self.root_nodes_qmax[i] = self.root_nodes_qmax_[i]

        for i in range(self.env_n):
            if (self.reset_mode == 0 and reset[i] and
                    self.status[i] in [2, 3, 4]):
                self.rollout_depth[i] = 0
                self.cur_nodes[i] = self.root_nodes[i]
                node_visit(self.cur_nodes[i])
                self.status[i] = 5


        info = self.prepare_info_dynamic(None, True)
        if self.im_enable:
            info["im_reward"] = torch.tensor(
                self.full_im_reward, dtype=torch.float, device=self.device
            ).unsqueeze(-1)

        self.internal_counter += 1
        return (
            states,
            torch.tensor(self.full_reward, dtype=torch.float, device=self.device),
            torch.tensor(self.full_done, dtype=torch.bool, device=self.device),
            torch.tensor(self.full_truncated_done, dtype=torch.bool, device=self.device),
            info)
