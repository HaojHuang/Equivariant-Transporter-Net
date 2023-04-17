import os
import numpy as np
import argparse
import pickle
import matplotlib
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='ravens')
parser.add_argument('--root_dir', type=str, default='./test_equi')
parser.add_argument('--task', type=str, default='block-insertion')
parser.add_argument('--n_demos', type=int,default=1)
parser.add_argument('--disp', action='store_false', default=True)

args = parser.parse_args()

def main(args):
    name = f'{args.task}-{args.n_demos}-'
    path = args.root_dir
    curve = []
    for fname in sorted(os.listdir(path)):
        fname = os.path.join(path, fname)
        if name in fname and '.pkl' in fname:
            n_steps = int(fname[(fname.rfind('-') + 1):-4])
            data = pickle.load(open(fname, 'rb'))
            rewards = []
            for reward, _ in data:
                rewards.append(reward)
            #print(len(rewards))
            score = np.mean(rewards)
            #print('score',score)
            std = np.std(rewards)
            #print('std',std)
            print(f'  {n_steps} steps:\t{score:.3f} \t± {std:.3f}')
            curve.append((n_steps, score, std))
    #
    if 'baseline' in args.root_dir:
        title = f' Tansporter Agent on {args.task} w/ {args.n_demos} demos'
    else:
        title = f' Equivalent Agent on {args.task} w/ {args.n_demos} demos'
    ylabel = 'Testing Task Success (%)'
    xlabel = '# of Training Steps'
    if args.disp:
        logs = {}
        curve = np.array(curve)
        #print(curve)
        curve = curve[curve[:,0].argsort()]
        #print(curve)
        logs[name] = (curve[:,0], curve[:,1], curve[:,2])
        fname = os.path.join(path, f'{name}-plot.png')
        plot(fname, title, ylabel, xlabel, data=logs, ylim=[0, 1])
        print(f'Done. Plot image saved to: {fname}')



COLORS = {
    'blue': [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0],
    'red': [255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0],
    'green': [089.0 / 255.0, 169.0 / 255.0, 079.0 / 255.0],
    'orange': [242.0 / 255.0, 142.0 / 255.0, 043.0 / 255.0],
    'yellow': [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0],
    'purple': [176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0],
    'pink': [255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0],
    'cyan': [118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0],
    'brown': [156.0 / 255.0, 117.0 / 255.0, 095.0 / 255.0],
    'gray': [186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0]
}

def plot(fname,  # pylint: disable=dangerous-default-value
         title,
         ylabel,
         xlabel,
         data,
         xlim=[-np.inf, 0],
         xticks= None,
         ylim=[np.inf, -np.inf],
         show_std=False):
  """Plot frame data."""
  # Data is a dictionary that maps experiment names to tuples with 3
  # elements: x (size N array) and y (size N array) and y_std (size N array)

  # Get data limits.
  for name, (x, y, _) in data.items():
    del name
    y = np.array(y)
    xlim[0] = max(xlim[0], np.min(x))
    xlim[1] = max(xlim[1], np.max(x))
    ylim[0] = min(ylim[0], np.min(y))
    ylim[1] = max(ylim[1], np.max(y))

  # Draw background.
  plt.title(title, fontsize=14)
  plt.ylim(ylim)
  plt.ylabel(ylabel, fontsize=14)
  plt.yticks(fontsize=14)

  #xlim = [1000,20000]
  #xlim = [100,1000]
  plt.xlim(xlim)
  plt.xlabel(xlabel, fontsize=14)
  plt.grid(True, linestyle='-', color=[0.8, 0.8, 0.8])
  ax = plt.gca()
  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_color('#000000')
  plt.rcParams.update({'font.size': 14})
  plt.rcParams['mathtext.default'] = 'regular'
  matplotlib.rcParams['pdf.fonttype'] = 42
  matplotlib.rcParams['ps.fonttype'] = 42
  #ax.set_xticks(np.linspace(1000,20000,1000),)

  # Draw data.
  color_iter = 0
  for name, (x, y, std) in data.items():
    del name
    x, y, std = np.float32(x), np.float32(y), np.float32(std)
    upper = np.clip(y + std, ylim[0], ylim[1])
    lower = np.clip(y - std, ylim[0], ylim[1])
    color = COLORS[list(COLORS.keys())[color_iter]]
    # print('x', x)
    # print('y', y)
    # print(y.mean())
    if show_std:
      plt.fill_between(x, upper, lower, color=color, linewidth=0, alpha=0.3)
    plt.plot(x, y, color=color, linewidth=2, marker='o', alpha=1.)
    color_iter += 1

  if xticks:
    plt.xticks(ticks=range(len(xticks)), labels=xticks, fontsize=14)
  else:
    plt.xticks(fontsize=14)

  plt.legend([name for name, _ in data.items()],
             loc='lower right', fontsize=14)
  plt.tight_layout()
  plt.show()
  plt.savefig(fname)
  plt.clf()

if __name__=="__main__":
    main(args)


