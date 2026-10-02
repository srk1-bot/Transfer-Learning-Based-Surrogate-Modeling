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

import time as ti # 計算コストの評価

from wave_util.make_noise import *

from analysis.SDOF_analysis_ops import *
from analysis.make_train_data import *
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

num = 20
sampling = 'fs' # maxmin
response_out = 'accel'
response = response_out
# 解析モデルの設定
model_pass = '/home/ishikawa/ドキュメント/m1/data/build_data/aisc-SMF-PG-4ELF_20.csv'
size_pass = '/home/ishikawa/ドキュメント/m1/data/build_data/aisc-shapes-database.csv'

# 入力地震動の設定
seed = 1
datasets_dir = f'/home/ishikawa/ドキュメント/m1/data/datasets/20SMF/seed{seed}/'
model_dir = f'/mnt/itoilab/ishikawa/m1/251114_master_20SMF/model/seed{seed}/'
loss_data_dir = f'loss_data/seed{seed}/'
data_dir = '/mnt/itoilab/ishikawa/m1/251114_master_20SMF/eq_data/dataset/seed_'+str(seed)+'/'


model_1 = torch.load( f'{model_dir}SMF_'+sampling+'_'+response_out+'_'+str(num)+'_norm_huber.pth', map_location='cuda', weights_only=False)
model_1.to('cuda')


# データセットの読み込み
x_max = np.load(f'{datasets_dir}SMF_xmax_'+response+'_'+sampling+'_'+str(num)+'.npy')
y_max = np.load(f'{datasets_dir}SMF_ymax_'+response+'_'+sampling+'_'+str(num)+'.npy')

# SDOFの読み込み
SDOF_model = torch.load(model_dir+r'SDOF_' + sampling + '_model_norm.pth', map_location='cuda', weights_only=False)
SDOF_model.to('cuda')
print(None)

i = 0
amped_eq = np.load(f'/home/ishikawa/ドキュメント/m1/notebooks/250621_ESD/0818_analysis/eq_data/eq_raw_{i}.npy')
for i in range(1, 5):
    amped_eq = np.concatenate([amped_eq, np.load(f'/home/ishikawa/ドキュメント/m1/notebooks/250521_Response_analysis/eq_data/eq_raw_{i}.npy')], axis=0)




# edpの予測
edp = predict_edp(model_1, SDOF_model, amped_eq[:], x_max, y_max, batch=2*5120)
rel_accel_surrogate = edp[:, 0, :] #(batch, N, sequence) 床応答加速度(相対加速度)


i = 0
if response=='accel':
    response_file = 'rel_accel'
elif response == 'disp' or response=='SDR':
    response_file = 'rel_disp'
else:
    response_file = response
rel_accel_analysis = np.load(f'/mnt/itoilab/ishikawa/m1/251114_master_20SMF/response_analysis/{response_file}_{i}.npy')
for i in range(1, 5):
    rel_accel_analysis = np.concatenate([rel_accel_analysis, np.load(f'/mnt/itoilab/ishikawa/m1/251114_master_20SMF/response_analysis/{response_file}_{i}.npy')], axis=0)


# 応答解析の結果を絶対加速度に変換する
# タイムステップの変換
num = rel_accel_analysis.shape[0]
forward = 3
# rel_accel_analysis = np.concatenate([np.zeros_like(rel_accel_analysis[:,:,0:10]), rel_accel_analysis[:, :,10:4096]], axis=2)
abs_eq_plus = np.concatenate([amped_eq.reshape(num, 1, 4096) for i in range(20)], axis=1)
abs_accel_analysis = rel_accel_analysis + abs_eq_plus



# 床応答加速度の計算


abs_accel_analysis  = rel_accel_analysis  + amped_eq[:, np.newaxis, :] # 絶対加速度に変換
abs_accel_surrogate = rel_accel_surrogate + amped_eq[:, np.newaxis, :] # 絶対加速度に変換

# 最大床応答加速度の計算
PFA_pos = np.max(abs_accel_analysis, axis=2) # m/s^2
PFA_neg = np.max(-abs_accel_analysis, axis=2) # m/s^2
PFA_analysis = np.maximum(PFA_pos, PFA_neg)

PFA_pos = np.max(abs_accel_surrogate, axis=2) # m/s^2
PFA_neg = np.max(-abs_accel_surrogate, axis=2) # m/s^2
PFA_surrogate = np.maximum(PFA_pos, PFA_neg)

fig = plt.figure(figsize=(5, 10))
plt.rcParams["font.size"] = 18
i = 0
color_surro = orange
color_analysis = green
plt.boxplot(PFA_surrogate[:, i], vert=False,
                boxprops=dict(color=color_surro, alpha=0.5), whis = [0., 100.], 
                medianprops=dict(color=color_surro),
                whiskerprops=dict(color=color_surro),
                capprops=dict(color=color_surro),
                flierprops=dict(markerfacecolor=color_surro, marker='o', markersize=1,
                                linestyle='none', markeredgecolor=color_surro),
                positions=[i+1+0.1], 
                label='surrogate model')
    
plt.boxplot(PFA_analysis[:, i], vert=False,
        boxprops=dict(color=color_analysis, alpha=0.5),whis = [0., 100.], 
        medianprops=dict(color=color_analysis),
        whiskerprops=dict(color=color_analysis),
        capprops=dict(color=color_analysis),
        flierprops=dict(markerfacecolor=color_analysis, marker='o', markersize=1,
                        linestyle='none', markeredgecolor=color_analysis),
        positions=[i+1-0.1], 
        label='response analysis')

for i in range(1, 20):
    plt.boxplot(PFA_surrogate[:, i], vert=False,
                boxprops=dict(color=color_surro, alpha=0.5), whis = [0., 100.], 
                medianprops=dict(color=color_surro),
                whiskerprops=dict(color=color_surro),
                capprops=dict(color=color_surro),
                flierprops=dict(markerfacecolor=color_surro, marker='o', markersize=1,
                                linestyle='none', markeredgecolor=color_surro),
                positions=[i+1+0.1],)
    
    plt.boxplot(PFA_analysis[:, i], vert=False,
        boxprops=dict(color=color_analysis, alpha=0.5), whis = [0., 100.], 
        medianprops=dict(color=color_analysis),
        whiskerprops=dict(color=color_analysis),
        capprops=dict(color=color_analysis),
        flierprops=dict(markerfacecolor=color_analysis, marker='o', markersize=1,
                        linestyle='none', markeredgecolor=color_analysis),
        positions=[i+1-0.1],)

# y軸の位置を0〜19、ラベルを1〜20に設定
plt.yticks(range(1, 21), [str(j+1) for j in range(1, 21)])
plt.ylabel('Story')
plt.xlabel('Peak floor acceleration [$\mathrm{m/s^2}$]')
plt.grid(axis='x') 
plt.xscale("log")
# --- 凡例をグラフの下に外出し ---
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

plt.show()



fig, axes = plt.subplots(4, 5, figsize=(15, 15))  # 5行4列
axes = axes.flatten()
plt.rcParams["font.size"] = 12

floor_dict = [f'{i+1}F' for i in range(1, 21)]

for floor_idx in range(20):
    ax = axes[floor_idx]

    ax.scatter(PFA_analysis[:, floor_idx], PFA_surrogate[:, floor_idx], c=blue, s=1)
    ax.plot(
        [0, np.max(22)],
        [0, np.max(22)],
        c=red,
        linewidth = 0.8
    )
    # 軸スケールをログにしたい場合はコメントアウトを外す
    ax.set_xscale('log')
    ax.set_yscale('log')
    if floor_idx%5 == 0:
        ax.set_ylabel('PFA\n by surrogate model [$\mathrm{m/s^2}$]')
    if floor_idx >= 15:
        ax.set_xlabel('PFA\n by analysis [$\mathrm{m/s^2}$]')
    
    ax.text(
        0.05, 0.98,  # 左上の位置（軸座標系で指定）
        floor_dict[floor_idx],
        transform=ax.transAxes,  # 軸座標系に変換
        fontsize=13,
        verticalalignment='top',
        horizontalalignment='left'
        )

    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.2, 22)
    ax.set_ylim(-0.2, 22)
    ticks = [10E-2, 10E-1, 10E0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

# レイアウト調整
plt.tight_layout()
plt.show()

PFA_surrogate_sort = np.sort(PFA_surrogate, axis=0)
PFA_analysis_sort = np.sort(PFA_analysis, axis=0)
N = len(PFA_surrogate_sort)
P_ex = []
for i in range(N):
    P_ex.append(1-(i+1)/(N+1))
P_ex = np.array(P_ex)

color = [red, green, blue, orange, 'black']
plt.figure(figsize=(5, 5*1.25))
plt.rcParams['font.size']=18
cnt = 0
for i in [0, 4, 9, 14, 19]:
    plt.plot(PFA_surrogate_sort[:, i], P_ex, c=color[cnt], label=f'{i+1}floor')
    # plt.plot(PFA_analysis_sort[:, i], P_ex, c=color[cnt], label=f'{i+1}floor analysis', linestyle='dashed')
    cnt += 1
plt.yscale('log')
plt.xlabel('PFA[$\mathrm{m/s^2}$]')
plt.ylabel('50 years exceedance probability')
plt.legend(fontsize=10)
plt.grid()
