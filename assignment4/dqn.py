import uuid
import time
import pickle
import os
import sys
import gym.spaces
import itertools
import numpy as np
import random
import logz
import tensorflow as tf
from collections import namedtuple
from dqn_utils import *
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


class QLearner(object):

    def __init__(self,
                 env,
                 q_func,
                 optimizer_spec,
                 session,
                 exploration,
                 replay_buffer_size,
                 batch_size,
                 gamma,
                 learning_starts,
                 learning_freq,
                 frame_history_len,
                 target_update_freq,
                 grad_norm_clipping,
                 double_q=True,
                 logdir=None,
                 max_steps=2e8,
                 cartpole=False):
        """Run Deep Q-learning algorithm.

        You can specify your own convnet using `q_func`.
        All schedules are w.r.t. total number of steps taken in the environment.

        Parameters
        ----------
        env: gym.Env
            gym environment to train on.
        q_func: function
            Model to use for computing the q function. It should accept the
            following named arguments:
                img_in: tf.Tensor
                    tensorflow tensor representing the input image
                num_actions: int
                    number of actions
                scope: str
                    scope in which all the model related variables
                    should be created
                reuse: bool
                    whether previously created variables should be reused.
        optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
        session: tf.Session
            tensorflow session to use.
        exploration: Schedule
            schedule for probability of chosing random action.
        replay_buffer_size: int
            How many memories to store in the replay buffer.
        batch_size: int
            How many transitions to sample each time experience is replayed.
        gamma: float
            Discount Factor
        learning_starts: int
            After how many environment steps to start replaying experiences
        learning_freq: int
            How many steps of environment to take between every experience replay
        frame_history_len: int
            How many past frames to include as input to the model.
        target_update_freq: int
            How many experience replay rounds (not steps!) to perform between
            each update to the target Q network
        grad_norm_clipping: float or None
            If not None gradients' norms are clipped to this value.
        double_q: bool
            If True, use double Q-learning to compute target values. Otherwise, vanilla DQN.
            https://papers.nips.cc/paper/3964-double-q-learning.pdf
        logdir: str
            Where we save the results for plotting later.
        max_steps: int
            Maximum number of training steps. The number of *frames* is 4x this
            quantity (modulo the initial random no-op steps).
        cartpole: bool
            If True, CartPole-v0. Else, PongNoFrameskip-v4
        """
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space)      == gym.spaces.Discrete
        self.max_steps = int(max_steps)
        self.target_update_freq = target_update_freq
        self.optimizer_spec = optimizer_spec
        self.batch_size = batch_size
        self.learning_freq = learning_freq
        self.learning_starts = learning_starts
        self.session = session
        self.exploration = exploration
        self.double_q = double_q
        self.cartpole = cartpole
        self.env = env

        if cartpole:
            input_shape = self.env.observation_space.shape # should be (4,)
        else:
            img_h, img_w, img_c = self.env.observation_space.shape
            input_shape = (img_h, img_w, frame_history_len * img_c)
        self.num_actions = self.env.action_space.n

        # ----------------------------------------------------------------------
        # Set up TensorFlow placeholders for:
        #
        #   - current observation (or state)
        #   - current action
        #   - current reward
        #   - next observation (or state)
        #   - end of episode mask
        #
        # For the end of episode mask: value is 1 if the next state corresponds
        # to the end of an episode, in which case there is no Q-value at the
        # next state; at the end of an episode, only the current state reward
        # contributes to the target, not the next state Q-value (i.e. target is
        # just rew_t_ph, not rew_t_ph + gamma * q_tp1).
        #
        # (You should not need to modify this placeholder code.)
        # ----------------------------------------------------------------------
        if cartpole:
            self.obs_t_ph   = tf.placeholder(tf.float32, [None]+list(input_shape))
            self.obs_tp1_ph = tf.placeholder(tf.float32, [None]+list(input_shape))
        else:
            self.obs_t_ph   = tf.placeholder(tf.uint8, [None]+list(input_shape))
            self.obs_tp1_ph = tf.placeholder(tf.uint8, [None]+list(input_shape))
        self.act_t_ph     = tf.placeholder(tf.int32,   [None])
        self.rew_t_ph     = tf.placeholder(tf.float32, [None])
        self.done_mask_ph = tf.placeholder(tf.float32, [None])

        # Casting to float on GPU ensures lower data transfer times.
        if cartpole:
            obs_t_float   = self.obs_t_ph
            obs_tp1_float = self.obs_tp1_ph
        else:
            obs_t_float   = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
            obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

        # ----------------------------------------------------------------------
        # You should fill in your own code to compute the Bellman error.  This
        # requires evaluating the current and next Q-values and constructing the
        # corresponding error.  TensorFlow will differentiate this error for
        # you; you just need to pass it to the optimizer.
        #
        # Your code should produce one scalar-valued tensor: `self.total_error`.
        # This will be passed to the optimizer in the provided code below.
        #
        # Your code should also produce two collections of variables:
        #
        #   q_func_vars
        #   target_q_func_vars
        #
        # These should hold all of the variables of the Q-function network and
        # target network, respectively. A convenient way to get these is to make
        # use of TF's "scope" feature.  For example, you can create your
        # Q-function network with the scope "q_func" like this:
        #
        #   <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
        #
        # And then you can obtain the variables like this:
        #
        #   q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
        #
        # Tips: use huber_loss (from dqn_utils) instead of squared error when
        # defining `self.total_error`. If you are using double DQN, modify your
        # code here to support that and normal (i.e., non-double) DQN.
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # START OF YOUR CODE
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # END OF YOUR CODE
        # ----------------------------------------------------------------------

        # Construct optimization op (with gradient clipping).
        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
        optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate,
                    **self.optimizer_spec.kwargs)
        self.train_fn = minimize_and_clip(optimizer, self.total_error,
                     var_list=q_func_vars, clip_val=grad_norm_clipping)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

        # Construct the replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len,
                                          cartpole=cartpole)
        self.replay_buffer_idx = None

        # Bells and whistles. Note the `self.env.reset()` call, though!
        self.model_initialized = False
        self.num_param_updates = 0
        self.mean_episode_reward      = -float('nan')
        self.std_episode_reward       = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        if cartpole: 
            self.log_every_n_steps = 1000
        else:
            self.log_every_n_steps = 10000
        self.start_time = time.time()
        self.last_obs = self.env.reset()
        self.t = 0


    def step_env(self):
        """Step the env and store the transition.

        At this point, `self.last_obs` contains the latest observation that was
        recorded from the simulator; see the end of `__init__`. Here, your code
        needs to store this observation and outcome (reward, next observation,
        etc.) into the replay buffer while doing one step in the env simulator.

        At the end of this block of code, the simulator should have been
        advanced one step, the replay buffer should contain one more transition,
        and, `self.last_obs` must point to the new latest observation.

        Useful functions you'll need to call:

            obs, reward, done, info = self.env.step(action)

        This steps the environment forward one step. And:

            obs = self.env.reset()

        This resets the environment if you reached an episode boundary.  Call
        `self.env.reset()` to get a new observation if `done=True`. For Pong and
        CartPole, this is guaranteed to start a new episode as they don't have a
        notion of 'ale.lives' in them.

        You cannot use `self.last_obs` directly as input into your network,
        since it needs to be processed to include context from previous frames.
        You should check out the replay buffer implementation in dqn_utils.py to
        see what functionality the replay buffer exposes. The replay buffer has
        a function `encode_recent_observation` that will take the latest
        observation that you pushed into the buffer and compute the
        corresponding input that should be given to a Q network by appending
        some previous frames. (The reason for this is to be memory-efficient
        and avoid having to save copies of each 84x84 frame.)

        Don't forget to include epsilon greedy exploration!  And remember that
        the first time you enter this loop, the model may not yet have been
        initialized; but of course, the first step might as well be random,
        since you haven't trained your net...
        """
        # ----------------------------------------------------------------------
        # START OF YOUR CODE
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # END OF YOUR CODE
        # ----------------------------------------------------------------------


    def update_model(self):
        """Perform experience replay and train the network.

        This is only done if the replay buffer contains enough samples for us to
        learn something useful -- until then, the model will not be initialized
        and random actions should be taken.  Training consists of four steps:
        
        3.a: Use the replay buffer to sample a batch of transitions. See the
        replay buffer code for function definitions.
        
        3.b: The boolean variable `model_initialized` indicates whether or not
        the model has been initialized. If the model is not initialized, then
        initialize it via standard TensorFlow initialization (you might find
        `tf.global_variables()` useful). Then, update the target network.
        
        3.c: Train the model. You will need to use `self.train_fn` and
        `self.total_error` ops that were created earlier: `self.total_error` is
        what you created to compute the total Bellman error in a batch, and
        `self.train_fn` will actually perform a gradient step and update the
        network parameters to reduce total_error. You might need to populate the
        following in your code:

          self.obs_t_ph
          self.act_t_ph
          self.rew_t_ph
          self.obs_tp1_ph
          self.done_mask_ph
          self.learning_rate  (get from `self.optimizer_spec`)
        
        3.d: Periodically update the target network by calling

          self.session.run(self.update_target_fn)

        you should update every `target_update_freq` steps, and you may find the
        variable `self.num_param_updates` usefull; it was initialized to 0.
        """
        if (self.t > self.learning_starts and \
            self.t % self.learning_freq == 0 and \
            self.replay_buffer.can_sample(self.batch_size)):
            # ------------------------------------------------------------------
            # START OF YOUR CODE
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # END OF YOUR CODE
            # ------------------------------------------------------------------
            self.num_param_updates += 1
        self.t += 1


    def log_progress(self):
        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
            self.std_episode_reward  = np.std(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = \
                max(self.best_mean_episode_reward, self.mean_episode_reward)

        # See the `log.txt` file for where these statistics are stored.
        if self.t % self.log_every_n_steps == 0:
            lr = self.optimizer_spec.lr_schedule.value(self.t)
            hours = (time.time() - self.start_time) / (60.*60.)
            logz.log_tabular("Steps",                 self.t)
            logz.log_tabular("Avg_Last_100_Episodes", self.mean_episode_reward)
            logz.log_tabular("Std_Last_100_Episodes", self.std_episode_reward)
            logz.log_tabular("Best_Avg_100_Episodes", self.best_mean_episode_reward)
            logz.log_tabular("Num_Episodes",          len(episode_rewards))
            logz.log_tabular("Exploration_Epsilon",   self.exploration.value(self.t))
            logz.log_tabular("Adam_Learning_Rate",    lr)
            logz.log_tabular("Elapsed_Time_Hours",    hours)
            logz.dump_tabular()


def learn(*args, **kwargs):
    alg = QLearner(*args, **kwargs)
    while True:
        alg.step_env()
        # The environment should have been advanced one step (and reset if done
        # was true), and `self.last_obs` should point to new latest observation
        alg.update_model()
        alg.log_progress()
        if alg.t > alg.max_steps:
            print("\nt = {} exceeds max_steps = {}".format(alg.t, alg.max_steps))
            sys.exit()
