import sys
from pathlib import Path

root = Path.cwd().parent          # プロジェクトルート
if str(root) not in sys.path:
    sys.path.append(str(root))    # modules の import 用

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt

import time as ti # 計算コストの評価

from wave_util import plot_wave
from wave_util.make_noise import *

from analysis.SDOF_analysis_ops import *

from analysis.frame_aisc_SMF import *


from network import learn_model
from network import masked_net
from network import CNN
blue = "#0768CF"
orange = '#F4B612'
green = '#6C963A'
red = "#bb2633"

import gc
import shutil

# 解析モデルの設定 
from pathlib import Path

root = Path.cwd().parent          # プロジェクトルート（notebook copy の1つ上）

model_pass = str(root / 'data' / 'build_data' / 'aisc-SMF-PG-4ELF_20.csv')
size_pass  = str(root / 'data' / 'build_data' / 'aisc-shapes-database.csv')

build_model(3, model_pass, size_pass, base_hinges=True)
run_gravity() # 重力解析
eigen_value, eigen_vector = run_modal() # 固有値解析
reset_analysis() # 解析の初期化
omega = [eigen_value[0][1], eigen_value[1][1]] # 固有値解析の結果
eigen_value


eq_data_dir = 'eq_data_directry/'
test_eq_raw = np.load(f'{eq_data_dir}TL_fs_train_raw_20.npy')

x_max = np.load('datasets/ESD_CS1/seed1/SMF_xmax_disp_fs_20.npy')
y_max = np.load('datasets/ESD_CS1/seed1/SMF_ymax_disp_fs_20.npy')
dataloader = torch.load('datasets/ESD_CS1/seed1/SMF_train_disp_fs_20.pth', weights_only=False)
SMF_disp = dataloader[:][1].to('cpu').detach().numpy().copy() *y_max.reshape(1, 1,  20, 1)
eq_filt = dataloader[:][0].to('cpu').detach().numpy().copy() * x_max


# 2Fやわらかい用の1質点系
DOF=1
m = np.array([1.0]) # kg
T = 2.4038627452341834 # 非減衰固有周期s
# T = 2.7
# T = 2.36
omega = np.array([2*np.pi/T])
h = 0.01 # 減衰定数
dt = 0.01 # 積分時間刻みs
batch_size = 1024
batch_size_long = 512
f_yield = np.array([4.]) # 降伏強度N
r_post = np.array([0.2])


def calc_disp(wave, m, f_yield, r_post, h, T=2.4038627452341834):
    omega = np.array([2*np.pi/T])
    k_spring=m*omega**2
    output = get_ops_response_N(DOF, m, k_spring, f_yield, np.array(wave, dtype=np.float64), dt, h, r_post=r_post, do_amplification=False)
    # 解析結果の取得
    return output['rel_disp'][0]

# for i in range(0, 10):
for i in [11, 19]:
    plt.figure(figsize=(10, 2.5))
    plt.plot(np.array(calc_disp(-test_eq_raw[i],m,  f_yield, r_post, h)))
    plt.plot(SMF_disp[i][0][19], label='frame', c=orange)
    plt.grid()
    plt.legend()
    plt.show()



# bayesian optimization to find the parameters of SDOF model

import optuna

def objective(trial):
    m = [1.0]
    f_yield = [trial.suggest_float('f_yield', 1.5, 1.9)]
    r_post =  [trial.suggest_float('r_post', 0.01, 0.1)]
    h = trial.suggest_float('h', 0.020, 0.040)
    T = trial.suggest_float('T', 2.4, 2.45)

    loss = 0
    # for i in range(len(frame_analysis)):
    for i in range(20):
    # for i in [11, 19]:
        loss += np.mean(np.array(calc_disp(-test_eq_raw[i],m,  f_yield, r_post, h, T=T))-SMF_disp[i][0][19])**2
    return loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)



import numpy as np
import matplotlib.pyplot as plt
with open('loss_data/02_create_SDOF.txt', 'r') as f:
    cnt = 0
    loss = []
    for row in f:
        if cnt >= 3:
            # print(row.split()[8])
            loss.append(float(row.split()[8]))
        cnt += 1
plt.plot(loss, color = blue)
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('$L_{\mathrm{s}}$')
plt.grid(axis='y')