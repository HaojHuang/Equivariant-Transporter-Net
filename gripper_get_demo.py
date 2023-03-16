import numpy as np
import argparse
from raven.dataset import Dataset
from raven.gripper_env import Environment
import os
from raven.gripper_block_insertion import BlockInsertion
from raven.gripper_place_red_in_green import PlaceRedInGreen
from raven.gripper_align_box_corner import AlignBoxCorner
from raven.gripper_stack_block_pyramid import StackBlockPyramid
from raven.gripper_palletizing_boxes import PalletizingBoxes
from raven.gripper_packing_boxes import PackingBoxes
parser = argparse.ArgumentParser(description='ravens_demos')

parser.add_argument('--assets_root', type=str, default='./raven/assets/')
parser.add_argument('--data_dir', type=str, default='.')
parser.add_argument('--n', type=int,default=100)
parser.add_argument('--disp', action='store_true', default=False)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--continuous', action='store_true', default=False)
parser.add_argument('--steps_per_seg', type=int, default=3)
#parser.add_argument('--task', type=str, default='align-box-corner')
#parser.add_argument('--task', type=str, default='place-red-in-green')
parser.add_argument('--task', type=str, default='block-insertion')
#parser.add_argument('--task', type=str, default='stack-block-pyramid')
#parser.add_argument('--task', type=str, default='palletizing-boxes')
#parser.add_argument('--task', type=str, default='packing-boxes')
#parser.add_argument('--task', type=str, default='assembling-kits')
args = parser.parse_args()

def main():
    enc_cls = Environment
    env = enc_cls(args.assets_root,
                  disp=args.disp,
                  shared_memory=False,
                  hz=480)
    if args.task == 'block-insertion':
        task = BlockInsertion(continuous=args.continuous)
    elif args.task == 'place-red-in-green':
        task = PlaceRedInGreen(continuous=args.continuous)
    elif args.task == 'align-box-corner':
        task = AlignBoxCorner(continuous=args.continuous)
    elif args.task == 'stack-block-pyramid':
        task = StackBlockPyramid(continuous=args.continuous)
    elif args.task == 'palletizing-boxes':
        task = PalletizingBoxes(continuous=args.continuous)
    elif args.task == 'packing-boxes':
        task = PackingBoxes(continuous=args.continuous)
    elif args.task == 'assembling-kits':
        task = AssemblingKits(continuous=args.continuous)
    else:
        raise RuntimeError('gripper version no {}'.format(args.task))

    task.mode = args.mode
    agent = task.oracle(env,steps_per_seg=args.steps_per_seg)
    dataset = Dataset(os.path.join(args.data_dir,f'{args.task}-{task.mode}'))

    # Train seeds are even and test seeds are odd.
    seed = dataset.max_seed
    if seed < 0:
        seed = -1 if (task.mode == 'test') else -2

        # Determine max steps per episode.
    max_steps = task.max_steps
    if args.continuous:
        max_steps *= (args.steps_per_seg * agent.num_poses)

        # Collect training data from oracle demonstrations.
    trial_n = 0
    while dataset.n_episodes < args.n:
        print(f'Oracle demonstration: {dataset.n_episodes + 1}/{args.n}')
        trial_n = trial_n+1
        episode, total_reward = [], 0
        seed += 2
        np.random.seed(seed)
        env.set_task(task)
        obs = env.reset()
        info = None
        reward = 0
        for _ in range(max_steps):
            act = agent.act(obs, info)
            # print('obs 0', obs['color'][0].shape, obs['color'][1].shape, obs['color'][2].shape)
            # print('obs 1', obs['depth'][0].shape, obs['depth'][0].shape, obs['depth'][0].shape)
            # print('act',act)
            # print('info',info)
            episode.append((obs, act, reward, info))
            obs, reward, done, info = env.step(act)
            # TODO FOR DEBUG CAHNGE DONE TO FALSE
            #done = False
            total_reward += reward
            print(f'Total Reward: {total_reward} Done: {done}')
            if done:
                break
        #print('=====================')
        print('\n')
        episode.append((obs, None, reward, info))
        # Only save completed demonstrations.
        if total_reward > 0.99:
            dataset.add(seed, episode)
    print('conduct {} trials to collect {} successful {} demonstration'.format(trial_n, args.task, args.n))
    print('the planner successful rate is {}'.format(trial_n/args.n))

if __name__ == '__main__':
  main()
