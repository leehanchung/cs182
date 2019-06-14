import os, random, time, argparse, gym, sys
import logz
from gym import wrappers
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import dqn
from dqn_utils import *
from atari_wrappers import *


def atari_model(img_in, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            out = layers.convolution2d(out, num_outputs=32,
                    kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64,
                    kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64,
                    kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,
                    activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions,
                    activation_fn=None)
        return out


def cartpole_model(x_input, num_actions, scope, reuse=False):
    """For CartPole we'll use a smaller network.
    """
    with tf.variable_scope(scope, reuse=reuse):
        out = x_input
        out = layers.fully_connected(out, num_outputs=32,
                activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=32,
                activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions,
                activation_fn=None)
        return out


def learn(env, session, args):
    if args.env == 'PongNoFrameskip-v4':
        lr_schedule = ConstantSchedule(1e-4)
        optimizer = dqn.OptimizerSpec(
            constructor=tf.train.AdamOptimizer,
            kwargs=dict(epsilon=1e-4),
            lr_schedule=lr_schedule
        )
        limit = max(int(args.num_steps/2), 2e6)
        exploration_schedule = PiecewiseSchedule([
                (0,     1.00),
                (1e6,   0.10),
                (limit, 0.01),
            ], outside_value=0.01
        )
        dqn.learn(
            env=env,
            q_func=atari_model,
            optimizer_spec=optimizer,
            session=session,
            exploration=exploration_schedule,
            replay_buffer_size=1000000,
            batch_size=32,
            gamma=0.99,
            learning_starts=50000,
            learning_freq=4,
            frame_history_len=4,
            target_update_freq=10000,
            grad_norm_clipping=10,
            double_q=args.double_q,
            logdir=args.logdir,
            max_steps=args.num_steps
        )
    elif args.env == 'CartPole-v0':
        lr_schedule = ConstantSchedule(5e-4)
        optimizer = dqn.OptimizerSpec(
            constructor=tf.train.AdamOptimizer,
            kwargs=dict(epsilon=1e-4),
            lr_schedule=lr_schedule
        )
        exploration_schedule = PiecewiseSchedule([
                (0,   1.00),
                (5e4, 0.10),
                (1e5, 0.02),
            ], outside_value=0.02
        )
        dqn.learn(
            env=env,
            q_func=cartpole_model,
            optimizer_spec=optimizer,
            session=session,
            exploration=exploration_schedule,
            replay_buffer_size=10000,
            batch_size=100,
            gamma=0.99,
            learning_starts=1000,
            learning_freq=4,
            frame_history_len=1,
            target_update_freq=500,
            grad_norm_clipping=10,
            double_q=args.double_q,
            logdir=args.logdir,
            max_steps=args.num_steps,
            cartpole=True
        )
    else:
        raise ValueError(args.env)
    env.close()


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def get_session():
    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.physical_device_desc for x in local_device_protos \
                if x.device_type == 'GPU']

    tf.reset_default_graph()
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                               intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def get_env(args):
    if args.env == 'CartPole-v0':
        env = gym.make(args.env)
        set_global_seeds(args.seed)
        env.seed(args.seed)
        expt_dir = os.path.join(args.logdir, "gym")
        env = wrappers.Monitor(env, expt_dir, force=True, video_callable=False)
    else:
        # Atari requires some environment wrapping; `print(env)` will show:
        #
        # <ClippedRewardsWrapper<ProcessFrame84<FireResetEnv ...
        #     <MaxAndSkipEnv<NoopResetEnv<EpisodicLifeEnv<Monitor ...
        #         <TimeLimit<AtariEnv<PongNoFrameskip-v4>>>>>>>>>>
        #
        # These are chained so that (for example) calling `step` on the outer
        # most wrapper moves back up the hierarchy to the AtariEnv, which
        # returns the output that moves in reverse and goes to the outer env.
        #
        # We also wrap around a Monitor, and information about episodes and
        # videos can be found in `expt_dir`, which you may find useful. See:
        # https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py
        env = gym.make(args.env)
        set_global_seeds(args.seed)
        env.seed(args.seed)
        expt_dir = os.path.join(args.logdir, "gym")
        env = wrappers.Monitor(env, expt_dir, force=True)
        env = wrap_deepmind(env)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_steps', type=int, default=4e6)
    parser.add_argument('--double_q', action='store_true', default=False)
    args = parser.parse_args()

    assert args.env in ['PongNoFrameskip-v4', 'CartPole-v0']
    if args.seed is None:
        args.seed = random.randint(0, 9999)
    print('random seed = {}'.format(args.seed))
    exp_name = 'dqn'
    if args.double_q:
        exp_name = 'double-dqn'

    if not(os.path.exists('data_dqn')):
        os.makedirs('data_dqn')
    logdir = exp_name+ '_' +args.env+ '_' +time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data_dqn', logdir)
    logz.configure_output_dir(logdir)
    args.logdir = logdir

    env = get_env(args)
    session = get_session()
    learn(env, session, args)
