import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

full_data = pd.DataFrame()
prefix, game = 'b2000', 'InvertedPendulum-v4'
# prefix, game = 'lb', 'CartPole-v0'

for folder in os.listdir('data'):
    split = folder.split('_')
    if prefix in split and game in split:
        config_list = split[split.index(prefix):split.index(game)]
    else:
        continue
    config = '_'.join(config_list)

    logdir = os.path.join('data', folder, 'events*')
    eventfile = glob.glob(logdir)[0]
    x, y, z = [], [], []
    for e in tf.train.summary_iterator(eventfile):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                x.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                y.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn':
                z.append(v.simple_value)
    data = pd.DataFrame({'Iteration': range(len(x)),
                        'Config': np.repeat(config, len(x)),
                         'Train_EnvstepsSoFar': x,
                         'Eval_AverageReturn': y,
                         'Eval_StdReturn': z})
    data['Eval_AverageReturn_Smooth'] = data['Eval_AverageReturn'].ewm(alpha=0.6).mean()
    data['Eval_StdReturn'] = data['Eval_StdReturn'].ewm(alpha=0.6).mean()
    full_data = pd.concat([full_data, data], axis=0, ignore_index=True)

sns.set_theme()
sns.set_context("paper")
sns.lineplot(data=full_data, x='Iteration', y='Eval_AverageReturn_Smooth', hue='Config')

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
if not (os.path.exists(data_path)):
    os.makedirs(data_path)
# plt.savefig(os.path.join(data_path, 'q2_b1000.png'), dpi=200, bbox_inches='tight')

plt.show()
