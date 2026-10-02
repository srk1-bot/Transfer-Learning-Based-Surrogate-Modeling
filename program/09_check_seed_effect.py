import os
# 親ディレクトリのパスを取得
import sys
sys.path.append(r'/home/ishikawa/ドキュメント/m1/modules')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt

import time as ti # 計算コストの評価

from wave_util import read_eq
from wave_util import plot_wave
from wave_util.make_noise import *
from wave_util import make_EQ_wav

from analysis.SDOF_analysis_ops import *
from analysis.make_train_data import *
from analysis.frame_pro import *
from analysis.fragility_analysis import *

from network import naname_net
from network import learn_model
from network import masked_net
from network import expansion
from network import CNN
from network import freqnet
blue = "#0768CF"
orange = "#FFAA00"
green = "#396800"
red = "#bb2633"

import gc


def calc_corrcoef(analysis, surro):
    analysis_mean = analysis.mean(axis=2, keepdims=True)
    surro_mean = surro.mean(axis=2, keepdims=True)
    analysis_std = analysis.std(axis=2, keepdims=True)
    surro_std = surro.std(axis=2, keepdims=True)

    # 共分散
    cov = np.mean((analysis-analysis_mean) * (surro-surro_mean), axis=2)

    corrs = cov/(analysis_std.squeeze() * surro_std.squeeze())
    return corrs


def calc_corr_coeff(seed, num, response_out):
    sampling = 'fs' # maxmin
    response = response_out
    # 入力地震動の設定
    datasets_dir = f'/home/ishikawa/ドキュメント/m1/data/datasets/20SMF/seed{seed}/'
    model_dir = f'/mnt/itoilab/ishikawa/m1/251114_master_20SMF/model/seed{seed}/'
    
    # モデルの読み込み
    model_1 = torch.load( f'{model_dir}SMF_'+sampling+'_'+response_out+'_'+str(num)+'_norm_huber.pth', map_location='cuda', weights_only=False)
    model_1.to('cuda')


    # データセットの読み込み
    x_max = np.load(f'{datasets_dir}SMF_xmax_'+response+'_'+sampling+'_'+str(num)+'.npy')
    y_max = np.load(f'{datasets_dir}SMF_ymax_'+response+'_'+sampling+'_'+str(num)+'.npy')

    # SDOFの読み込み
    SDOF_model = torch.load(model_dir+r'SDOF_' + sampling + '_model_norm.pth', map_location='cuda', weights_only=False)
    SDOF_model.to('cuda')
    # print(None)

    # 検証用データの読み込み
    valid_long = torch.load(datasets_dir+'SMF_valid_'+response+'_'+sampling+'_'+str(num)+'.pth', weights_only=False, map_location='cuda')

    val_resp_analysis = valid_long[:][1].to('cpu').detach().numpy().copy()
    val_eq_true = valid_long[:][0].to('cpu').detach().numpy().copy()
    val_resp_analysis = val_resp_analysis[:,0,  :, :]*y_max.reshape(1, 20, 1)# (60, 20, 4096)

    val_eq = valid_long[:][0]
    val_resp_surro = model_1(SDOF_model(val_eq)).to('cpu').detach().numpy().copy()[:, 0, :, :]*y_max.reshape(1, 20, 1)

    corrs = calc_corrcoef(val_resp_analysis, val_resp_surro)

    # メモリ解放
    del model_1, SDOF_model, val_resp_analysis, val_resp_surro
    return corrs

plt.rcParams['font.size'] = 17
red = 'tab:red'
blue = 'tab:blue'

plt.figure(figsize = (8, 6))

num_list = [10, 20, 40, 80, 120]
loss_data_dir = 'loss_data/'
corrs_accel = np.load(f'{loss_data_dir}corrs_accel.npy')
corrs_SDR = np.load(f'{loss_data_dir}corrs_SDR.npy')
corrs_accel_ave = np.average(corrs_accel, axis=3) # (5, 9, 60 ) 各地震動に対する全階の応答を見る
corrs_SDR_ave = np.average(corrs_SDR, axis=3) # (5, 9, 60 ) 各地震動に対する各階の応答を見る

positions  = np.arange(len(num_list)) * 2
plt.boxplot([corrs_accel_ave[i].flatten() for i in range(len(num_list))], positions=positions, 
            vert=True, patch_artist=True,
            # whis = [0, 100],
            boxprops=dict(color=red, alpha=0.5, facecolor=red),
            medianprops=dict(color="black"),
            whiskerprops=dict(color=red),
            capprops=dict(color=red),
            flierprops=dict(markerfacecolor=red, marker='o', markersize=5, linestyle='none', markeredgecolor=red),
            widths=0.6,
            showfliers=False
            )

# for i in range(len(num_list)):
#     plt.scatter([positions[i] for _ in range(len(corrs_accel_ave[i].flatten()))], corrs_accel_ave[i].flatten(), color=red, s=10, alpha=0.7)
# x軸ラベルをデータセットサイズに

xticks_pos = (positions)
plt.xticks(xticks_pos, [str(num) for num in num_list])

plt.xlabel("Dataset Size (num)")
plt.ylabel("Correlation Coefficient")
# plt.legend([ "Valid"], loc="upper right")
plt.grid(axis="y")
plt.ylim(0.8, 1.01)

plt.show()
