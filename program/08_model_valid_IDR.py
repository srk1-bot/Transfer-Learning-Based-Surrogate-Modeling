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


num = 20
sampling = 'fs' # maxmin
response_out = 'SDR'
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

valid_long = torch.load(datasets_dir+'SMF_valid_'+response+'_'+sampling+'_'+str(num)+'.pth', weights_only=False, map_location='cuda')


val_resp_analysis = valid_long[:][1].to('cpu').detach().numpy().copy()
val_resp_analysis = val_resp_analysis[:,0,  :, :]*y_max.reshape(1, 20, 1)# (60, 20, 4096)

val_eq = valid_long[:][0]
val_resp_surro = model_1(SDOF_model(val_eq)).to('cpu').detach().numpy().copy()[:, 0, :, :]*y_max.reshape(1, 20, 1)

def calc_corrcoef(analysis, surro):
    analysis_mean = analysis.mean(axis=2, keepdims=True)
    surro_mean = surro.mean(axis=2, keepdims=True)
    analysis_std = analysis.std(axis=2, keepdims=True)
    surro_std = surro.std(axis=2, keepdims=True)

    # 共分散
    cov = np.mean((analysis-analysis_mean) * (surro-surro_mean), axis=2)

    corrs = cov/(analysis_std.squeeze() * surro_std.squeeze())
    return corrs

corrs = calc_corrcoef(val_resp_analysis, val_resp_surro)
corrs.shape

fig, axes = plt.subplots(3, 3, figsize=(20, 10))  # 3行3列のサブプロット
unit = {
    'accel': '[$\mathrm{m/s^2}$]',
    'deformation': '[$\mathrm{rad}$]',
    'disp': '[$\mathrm{m}$]',
    'glbforc': '[$\mathrm{N/m}$]',
    'SDR': '[-]'
}
moji = ['(1)', '(2)','(3)', '(4)']
alphabet = ['(a)', '(b)', '(c)']

width = 1.0
rows = 3
cols = 3


# 上位5, 50, 95に近いものを割り出し
percentiles = [5, 50, 95] # パーセンタイル
floors = [0, 9, 19] # 対象とする階
for j, floor in enumerate(floors):  # 列方向：階
    targets = np.percentile(corrs[:, floor], percentiles)
    results = []
    for p, t in zip(percentiles, targets):
        idx = np.argmin(np.abs(corrs[:, floor] - t))
        closest_value = corrs[idx][floor]
        results.append((p, t, closest_value, idx))

    # 各パーセンタイルに対応するグラフを上から順に描画
    for i, data in enumerate(results):  # 行方向：パーセンタイル
        idx = data[3]
        ax = axes[i, j]

        # 応答のプロット
        ax.plot(time, val_resp_analysis[idx][floor], label='response analysis', c=blue, linewidth=width)
        ax.plot(time, val_resp_surro[idx][floor], label='surrogate model', c=orange, linewidth=width)

        ax.set_xlim(0, time[-1])
        ax.grid()
        
        # Y軸制限の自動設定
        maxresp = np.max((np.max(np.abs(val_resp_analysis[idx][floor])), np.max(np.abs(val_resp_surro[idx][floor]))))
        ax.set_ylim(-maxresp*1.5, maxresp*1.5)

        # Y軸ラベル (最初の列のみ)
        if j == 0:
            # {floor+1}Fの文字を消す代わりに、タイトルとして下に表示
            # ax.set_ylabel(f'{response} {unit[response]}')
            ax.set_ylabel(f'IDR \n{unit[response]}')
            # ax.set_ylabel(f"{alphabet[i]} {int(percentiles[i])}%\n{unit[response]}", fontsize=17)
            
        # X軸ラベル (一番下の行のみ)
        if i == rows - 1: # i == 2
            ax.set_xlabel('t [$\mathrm{s}$]')
            
            # **【変更点】タイトルをX軸の下に配置**
            # f"{floor+1}F" をX軸の下（y=-0.35）に配置
            ax.set_title(f"{alphabet[j]} {floor+1}F", fontsize=18, y=-0.6)
            # 一番下の行以外はX軸の目盛りラベルを非表示にする
        else:
            ax.tick_params(labelbottom=False)
            

        # タイトル（元のコードの列タイトルを削除）
        # if i == 0:
        #     ax.set_title(f"{floor+1}F", fontsize=14) # <-- これを削除

        # 凡例（左上だけに表示）
        if i != 2:
            ax.text(23, maxresp, f"{alphabet[j][:-1]}-{moji[i][1:]} {percentiles[i]}% at {floor+1}F", fontsize=17)
        else:
            ax.text(23, maxresp, f"{alphabet[j][:-1]}-{moji[i][1:]} {percentiles[i]}% at {floor+1}F", fontsize=17)
        if i == 0 and j == 2:
             # 凡例をグラフの下に外出し (loc='upper center'を維持し、bbox_to_anchorで調整)
             ax.legend(loc='upper center', bbox_to_anchor=(0.4, 1.5), ncol=2, frameon=False, fontsize=17)


# タイトルが下に収まるように、下部のマージンを調整
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()