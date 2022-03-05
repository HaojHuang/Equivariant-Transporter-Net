import datetime
import os
import numpy as np
#from equ_transporter import TransporterAgent as equ_agent
from raven.dataset import Dataset
import argparse
import tensorflow as tf
import torch
import torch.backends.cudnn as cudnn
from raven.gripper_env import Environment
#from ravens import tasks
import pickle
from raven.gripper_block_insertion import BlockInsertion
from raven.gripper_place_red_in_green import PlaceRedInGreen
from raven.gripper_align_box_corner import AlignBoxCorner
from raven.gripper_stack_block_pyramid import StackBlockPyramid
from raven.gripper_palletizing_boxes import PalletizingBoxes
from raven.gripper_packing_boxes import PackingBoxes

from networks.equivariant_transporter import TransporterAgent as equ_agent
from networks.non_equi_transporter import TransporterAgent as non_equi_agent
from networks.femi_transporter import TransporterAgent as femi_agent
from networks.semi_transporter import TransporterAgent as semi_agent
from networks.equivariant_transporter_tail import TransporterAgent as equ_agent_tail

parser = argparse.ArgumentParser(description='ravens_test')
parser.add_argument('--root_dir', type=str, default='.')
parser.add_argument('--data_dir', type=str, default='.')
parser.add_argument('--assets_root', type=str, default='./raven/assets')
parser.add_argument('--task', type=str, default='block-insertion')
parser.add_argument('--n_demos', type=int,default=10)# the demo used for testing
parser.add_argument('--n_steps', type=int,default=1000)# the train steps per epoch
parser.add_argument('--n_runs', type=int,default=1)
#parser.add_argument('--interval', type=int,default=1000)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--disp', action='store_true', default=False)
parser.add_argument('--shared_memory', action='store_true', default=False)
parser.add_argument('--equ', action='store_true', default=False)
parser.add_argument('--lite', action='store_true', default=False)
parser.add_argument('--angle_lite', action='store_true', default=False)
parser.add_argument('--continuous', action='store_true', default=False)
parser.add_argument('--entire', action='store_true', default=False)
parser.add_argument('--femi', action='store_true', default=False)
parser.add_argument('--semi', action='store_true', default=False)
parser.add_argument('--non', action='store_true', default=False)
parser.add_argument('--tail', action='store_true', default=False)
args = parser.parse_args()

def main(args):

    # Initialize environment and task.
    env = Environment(
        args.assets_root,
        disp=args.disp,
        shared_memory=args.shared_memory,
        hz=480)
    # TODO FOR NOW JUST TEST on block-insertion
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
    task.mode = 'test'

    # Load test dataset.
    ds = Dataset(os.path.join(args.data_dir, f'{args.task}-test'))
    for train_run in range(args.n_runs):
        #train_run +=1
        name = f'{args.task}-{args.n_demos}-{train_run}'
        # set seed
        np.random.seed(train_run + 1)
        torch.set_num_threads(train_run + 1)
        torch.manual_seed(train_run + 1)
        cudnn.benchmark = False
        cudnn.deterministic = True
        # Initial Agent
        #agent = task.oracle(env, steps_per_seg=3)
        if args.equ:
            print('equi agent')
            agent = equ_agent(name=name,task=args.task,root_dir=args.data_dir,lite=args.lite, angle_lite = args.angle_lite)
        if args.femi:
            print('femi_agent')
            agent = femi_agent(name=name,task=args.task,root_dir=args.data_dir,lite=args.lite, angle_lite = args.angle_lite)
        if args.semi:
            print('semi_agent')
            agent = semi_agent(name=name,task=args.task,root_dir=args.data_dir,lite=args.lite)
        if args.non:
            print('no equivariant agent')
            agent = non_equi_agent(name=name,task=args.task,root_dir=args.data_dir)
        if args.tail:
            print('equvairant agent with tail network')
            agent = equ_agent_tail(name=name,task=args.task,root_dir=args.data_dir,lite=args.lite, angle_lite = args.angle_lite)
        
        if args.entire ==True:
            n_steps = [20000,15000,5000,2000]
        else:
            n_steps = [args.n_steps] 
        
        for test_step in n_steps:
            agent.load(test_step)
            results = []
            #print(ds.n_episodes,'============')
            for i in range(ds.n_episodes):
                print(f'Test: {i + 1}/{ds.n_episodes}')
                episode, seed = ds.load(i)
                goal = episode[-1]
                total_reward = 0
                np.random.seed(seed)
                env.seed(seed)
                env.set_task(task)
                obs = env.reset()
                info = None
                reward = 0
                for _ in range(task.max_steps):
                    act = agent.act(obs, info, goal)
                    #act = agent.act(obs, info)
                    obs, reward, done, info = env.step(act)
                    total_reward += reward
                    print(f'Total Reward: {total_reward} Done: {done}')
                    if done:
                        break
                results.append((total_reward, info))
    
                # Save results.
                if args.equ:
                  if not os.path.exists(os.path.join(args.root_dir, 'test_equi')):
                    os.makedirs(os.path.join(args.root_dir, 'test_equi'))
                  
                  with open(os.path.join(args.root_dir, 'test_equi', f'{name}-{test_step}.pkl'),'wb') as f:
                      pickle.dump(results, f)
                if args.non:
                  if not os.path.exists(os.path.join(args.root_dir, 'test_non_equi')):
                    os.makedirs(os.path.join(args.root_dir, 'test_non_equi'))
                  with open(os.path.join(args.root_dir, 'test_non_equi', f'{name}-{test_step}.pkl'),'wb') as f:
                      pickle.dump(results, f)
                      
                if args.femi:
                  if not os.path.exists(os.path.join(args.root_dir, 'test_femi')):
                    os.makedirs(os.path.join(args.root_dir, 'test_femi'))
                  with open(os.path.join(args.root_dir, 'test_femi', f'{name}-{test_step}.pkl'),'wb') as f:
                      pickle.dump(results, f)
                      
                if args.semi:
                  if not os.path.exists(os.path.join(args.root_dir, 'test_semi')):
                    os.makedirs(os.path.join(args.root_dir, 'test_semi'))
                  with open(os.path.join(args.root_dir, 'test_semi', f'{name}-{test_step}.pkl'),'wb') as f:
                      pickle.dump(results, f)
                if args.tail:
                  if not os.path.exists(os.path.join(args.root_dir, 'test_tail')):
                    os.makedirs(os.path.join(args.root_dir, 'test_tail'))
                  with open(os.path.join(args.root_dir, 'test_tail', f'{name}-{test_step}.pkl'),'wb') as f:
                      pickle.dump(results, f)
                

if __name__=="__main__":
    main(args)
