
import argparse
import gym
import numpy as np
import os
import torch

import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter


offline_hyperparameters = {
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'alpha': 1.0, 'eta': 2.0,   'num_epochs': 2000, 'gn': 9.0,  'expectile': 0.7},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'alpha': 1.0, 'eta': 2.0,   'num_epochs': 2000, 'gn': 2.0,  'expectile': 0.7},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'alpha': 1.0, 'eta': 1.0,   'num_epochs': 2000, 'gn': 8.0,  'expectile': 0.7},
    'hopper-medium-v2':              {'lr': 3e-4, 'alpha': 1.0, 'eta': 2.0,   'num_epochs': 2000, 'gn': 9.0,  'expectile': 0.6},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'alpha': 1.0, 'eta': 1.0,   'num_epochs': 2000, 'gn': 4.0,  'expectile': 0.6},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'alpha': 1.0, 'eta': 0.5,   'num_epochs': 2000, 'gn': 5.0,  'expectile': 0.6},
    'walker2d-medium-v2':            {'lr': 3e-4, 'alpha': 1.0, 'eta': 1.0,   'num_epochs': 2000, 'gn': 1.0,  'expectile': 0.6},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'alpha': 1.0, 'eta': 1.0,   'num_epochs': 2000, 'gn': 4.0,  'expectile': 0.6},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'alpha': 1.0, 'eta': 1.0,   'num_epochs': 2000, 'gn': 5.0,  'expectile': 0.6},
    'antmaze-umaze-v0':              {'lr': 3e-4, 'alpha': 1.0, 'eta': 1.0,   'num_epochs': 1000, 'gn': 2.0,  'expectile': 0.9},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-4, 'alpha': 1.0, 'eta': 2.0,   'num_epochs': 1000, 'gn': 3.0,  'expectile': 0.9},
    'antmaze-medium-play-v0':        {'lr': 1e-3, 'alpha': 1.0, 'eta': 4.0,   'num_epochs': 1000, 'gn': 2.0,  'expectile': 0.9},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'alpha': 1.0, 'eta': 3.0,   'num_epochs': 1000, 'gn': 1.0,  'expectile': 0.9},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'alpha': 1.0, 'eta': 4.5,   'num_epochs': 1000, 'gn': 10.0, 'expectile': 0.9},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'alpha': 1.0, 'eta': 3.5,   'num_epochs': 1000, 'gn': 7.0,  'expectile': 0.9},
    'pen-human-v1':                  {'lr': 3e-5, 'alpha': 1.0, 'eta': 0.15,  'num_epochs': 1000, 'gn': 7.0,  'expectile': 0.7},
    'pen-cloned-v1':                 {'lr': 3e-5, 'alpha': 1.0, 'eta': 0.1,   'num_epochs': 1000, 'gn': 8.0,  'expectile': 0.7},                 
}

online_hyperparameters = {
    'mujoco': {'lr': 3e-4, 'alpha': 0.05, 'eta': 1.0, 'num_epochs': 1000, 'gn': 2.0},
    'dmc':    {'lr': 3e-4, 'alpha': 0.05, 'eta': 1.0, 'num_epochs': 500 , 'gn': 2.0},
}

def train_agent(env, state_dim, action_dim, device, output_dir, args):
    if args.rl_type == 'offline':
        # Load buffer
        dataset = d4rl.qlearning_dataset(env)
        data_sampler = Data_Sampler(dataset, device, args.reward_tune)
        utils.print_banner('Loaded buffer')

    from agents.ql_cm import CPQL as Agent
    agent = Agent(state_dim=state_dim,
                  action_dim=action_dim,
                  action_space = env.action_space,
                  device=device,
                  discount=args.discount,
                  max_q_backup=args.max_q_backup,
                  lr=args.lr,
                  eta=args.eta,
                  alpha=args.alpha,
                  lr_decay=args.lr_decay,
                  lr_maxt=args.num_epochs,
                  grad_norm=args.gn,
                  q_mode=args.q_mode,
                  sampler=args.sampler,
                  expectile=args.expectile,
                  memory_size=args.memory_size,)

    writer = SummaryWriter(output_dir) 

    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    log_interval = int(args.eval_freq * args.num_steps_per_epoch)

    utils.print_banner(f"Training Start", separator="*", num_star=90)
    while (training_iters < max_timesteps+1):
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))
        done = False
        state = env.reset()
        episode_steps = 0
        episode_reward = 0.

        if args.rl_type == 'offline': # Offline RL
            # Training
            loss_metric = agent.train(replay_buffer=data_sampler,
                                      batch_size=args.batch_size,
                                      log_writer=writer)
            training_iters += 1

            writer.add_scalar('Loss/bc_loss', np.mean(loss_metric['bc_loss']), training_iters)
            writer.add_scalar('Loss/ql_loss', np.mean(loss_metric['ql_loss']), training_iters)
            writer.add_scalar('Loss/actor_loss', np.mean(loss_metric['actor_loss']), training_iters)
            writer.add_scalar('Loss/critic_loss', np.mean(loss_metric['critic_loss']), training_iters)
            
            # Logging
            if training_iters % log_interval == 0:
                if loss_metric is not None:
                    utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
                    logger.record_tabular('Trained Epochs', curr_epoch)
                    logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
                    logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
                    logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
                    logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
                    logger.dump_tabular()

                # Evaluating
                eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent, 
                                                                                       args.rl_type,
                                                                                       args.env_name, 
                                                                                       args.seed,
                                                                                       eval_episodes=args.eval_episodes)

                writer.add_scalar('Eval/avg', eval_res, training_iters)
                writer.add_scalar('Eval/std', eval_res_std, training_iters)
                writer.add_scalar('Eval/norm_avg', eval_norm_res, training_iters)
                writer.add_scalar('Eval/norm_std', eval_norm_res_std, training_iters)

                logger.record_tabular('Average Episodic Reward', eval_res)
                logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
                logger.dump_tabular()

                if args.save_checkpoints:
                    agent.save_model(output_dir, curr_epoch)
        else: # Online RL
            # Run for one whole episode
            while not done:
                # Updating the replay buffer
                if training_iters < args.online_start_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.sample_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.append_memory(state, action, reward, next_state, 1. - done)
                state = next_state
                episode_steps += 1
                episode_reward += reward

                # Training
                if training_iters >= args.online_start_steps:
                    loss_metric = agent.train(agent.memory,
                                            batch_size=args.batch_size,
                                            log_writer=writer)
                else:
                    loss_metric = None

                training_iters += 1
                
                if loss_metric is not None:
                    writer.add_scalar('Loss/bc_loss', np.mean(loss_metric['bc_loss']), training_iters)
                    writer.add_scalar('Loss/ql_loss', np.mean(loss_metric['ql_loss']), training_iters)
                    writer.add_scalar('Loss/actor_loss', np.mean(loss_metric['actor_loss']), training_iters)
                    writer.add_scalar('Loss/critic_loss', np.mean(loss_metric['critic_loss']), training_iters)
                
                # Logging
                if training_iters % log_interval == 0:
                    if loss_metric is not None:
                        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
                        logger.record_tabular('Trained Epochs', curr_epoch)
                        logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
                        logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
                        logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
                        logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
                        logger.dump_tabular()

                    # Evaluating
                    eval_res, eval_res_std, _, _ = eval_policy(agent, 
                                                               args.rl_type,
                                                               args.env_name, 
                                                               args.seed,
                                                               eval_episodes=args.eval_episodes)
                    done = True

                    writer.add_scalar('Eval/avg', eval_res, training_iters)
                    writer.add_scalar('Eval/std', eval_res_std, training_iters)

                    logger.record_tabular('Average Episodic Reward', eval_res)
                    logger.dump_tabular()

                    if args.save_checkpoints:
                        agent.save_model(output_dir, curr_epoch)

    agent.save_model(output_dir, curr_epoch)
    
    writer.close()


# Runs policy for [eval_episodes] episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, rl_type, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    scores = []
    for _ in range(eval_episodes):
        traj_return = 0.
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    if rl_type == 'offline':
        normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
        avg_norm_score = eval_env.get_normalized_score(avg_reward)
        std_norm_score = np.std(normalized_scores)

        utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
        return avg_reward, std_reward, avg_norm_score, std_norm_score
    else:
        return avg_reward, std_reward, 0, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument('--device', default=0, type=int) 

    parser.add_argument('--rl_type', default="offline", type=str, help='offline or online RL tasks (default: offline)') 
    parser.add_argument("--q_mode", default="q", type=str, help='q for CPQL and q_v for CPIQL') 

    parser.add_argument("--env_name", default="hopper-medium-expert-v2", type=str, help='Mujoco Gym environment') 
    parser.add_argument("--seed", default=0, type=int, help='random seed (default: 0)') 

    parser.add_argument("--dir", default="results", type=str) 
    parser.add_argument('--save_checkpoints', action='store_true')

    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--online_start_steps", default=10000, type=int)
    parser.add_argument("--memory_size", default=1e6, type=int)
    parser.add_argument("--batch_size", default=256, type=int, help='batch size (default: 256)')
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument("--discount", default=0.99, type=float, help='discount factor for reward (default: 0.99)')

    args = parser.parse_args()

    if args.rl_type == 'online' and args.q_mode == 'q_v':
        raise AssertionError("CPIQL is not supported for online RL tasks!")

    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    if args.rl_type == 'offline':
        args.num_epochs = offline_hyperparameters[args.env_name]['num_epochs']
        args.lr = offline_hyperparameters[args.env_name]['lr']
        args.eta = offline_hyperparameters[args.env_name]['eta']
        args.alpha = offline_hyperparameters[args.env_name]['alpha']
        args.gn = offline_hyperparameters[args.env_name]['gn']
        args.expectile = offline_hyperparameters[args.env_name]['expectile']

        if 'antmaze' in args.env_name:
            args.max_q_backup = True
            args.reward_tune = 'cql_antmaze'
            args.sampler = 'multistep'
        else:
            args.max_q_backup = False
            args.reward_tune = 'no'
            args.sampler = 'onestep'

        args.eval_freq = 50
        args.eval_episodes = 10 if 'v2' in args.env_name else 100
    else: 
        args.num_epochs = online_hyperparameters['mujoco']['num_epochs']
        args.lr = online_hyperparameters['mujoco']['lr']
        args.eta = online_hyperparameters['mujoco']['eta']
        args.alpha = online_hyperparameters['mujoco']['alpha']
        args.gn = online_hyperparameters['mujoco']['gn']
        args.expectile = 0

        args.max_q_backup = False
        args.reward_tune = 'no'
        args.sampler = 'onestep'
    
        args.eval_freq = 50
        args.eval_episodes = 10 

    # Setup Logging
    file_name = 'QL' if args.q_mode == 'q' else 'IQL'
    if args.q_mode == 'q_v':
        file_name += f'|tau-{args.expectile}'
    file_name += f'|{args.seed}'
    file_name += f"|alpha-{args.alpha}|eta-{args.eta}"

    file_name += f'|sampler_{args.sampler}'

    results_dir = os.path.join(args.output_dir, args.rl_type, args.env_name, file_name)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    # if os.path.exists(os.path.join(results_dir, 'variant.json')):
    #     raise AssertionError("Experiment under this setting has been done!")

    variant = vars(args)
    variant.update(version=f"CMQL")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(env,
                state_dim,
                action_dim,
                args.device,
                results_dir,
                args)
