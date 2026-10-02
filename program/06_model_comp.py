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


from wave_util.make_noise import *

from analysis.SDOF_analysis_ops import *
from analysis.make_train_data import *
from analysis.frame_aisc_SMF import *


blue = "#0768CF"
orange = "#DFA400"
green = '#6C963A'
red = "#bb2633"


seed = 1
sampling = 'fs'
data_dir = 'eq_data/dataset/seed_'+str(seed)+'/'
datasets_dir = f'data/datasets/20SMF/seed{seed}/'
model_dir = f'model/seed{seed}/'
loss_data_dir = f'loss_data/seed{seed}/'
sfac = 1.0

train_eq_raw = np.load(data_dir + r'SDOF_fs_train_raw.npy')[:, :4096]*sfac
valid_eq_raw = np.load(data_dir + r'valid_raw.npy')[:, :4096]
# test_eq_raw = np.load(data_dir + r'test_raw.npy')[:2000, :4096]


train_eq_filt = np.load(data_dir + r'SDOF_fs_train_filt.npy')[:, :4096]*sfac
valid_eq_filt = np.load(data_dir + r'valid_filt.npy')[:, :4096]
# test_eq_filt = np.load(data_dir + r'test_filt.npy')[:2000, :4096]

batch_size_long=256


result = {'m':1.0,'f_yield': 4.326941495004812, 'r_post': 0.3700337242085181, 'h': 0.032054231887479766, 'T': 2.412431911650598}
m = np.array([result['m']])
f_yield = np.array([result['f_yield']])
r_post = np.array([result['r_post']])
h = result['h']
N=1
T =result['T'] # 非減衰固有周期s
omega = np.array([2*np.pi/T])
dt = 0.01 # 積分時間刻みs
batch_size = 1024
batch_size_long = 1024
c=2*h*m*omega


print(f'{model_dir}SDOF_'+sampling + '_model_norm.pth')


# テスト用波形の読み込み
# read = 'SDOF_fs_train'
read = 'valid'
# read = 'test'

# 地震動データの読み込みtest_eq_raw = np.load(data_dir + r'test_raw.npy')[:2000, :4096]
test_eq_raw = np.load(data_dir  + read + '_raw.npy')[:, :4096]
test_eq_filt = np.load(data_dir + read + '_filt.npy')[:, :4096]

# model_1 = torch.load( 'model/SMF_'+sampling+'_'+response_out+'_'+str(num)+'.pth', map_location='cuda')
# model_1.to('cuda')
batch_size_long=256

# データセットの読み込み
x_max = np.load(f'{datasets_dir}SDOF_'+sampling+'_xmax.npy')
y_max = np.load(f'{datasets_dir}SDOF_'+sampling+'_ymax.npy')

# SDOFの読み込み
SDOF_model = torch.load(f'{model_dir}SDOF_'+sampling + '_model_norm.pth', map_location='cuda', weights_only=False)
SDOF_model.to('cuda')
print(None)


val_long = torch.load(f'{datasets_dir}SDOF_fs_long_valid.pth', weights_only=False)
train_long = torch.load(f'{datasets_dir}SDOF_fs_long_train.pth', weights_only=False)
val_resp_analysis = val_long[:][1].to('cpu').detach().numpy().copy()
val_resp_analysis = val_resp_analysis[:, :, 0, :]*y_max # (60, 4, 4096)

val_eq = val_long[:][0] # 正規化されている状態
val_resp_surro = SDOF_model(val_eq).to('cpu').detach().numpy().copy()*y_max

rel_accel_analysis = val_resp_analysis[:, 0, :]
rel_vel_analysis   = val_resp_analysis[:, 1, :]
rel_disp_analysis  = val_resp_analysis[:, 2, :]
Q_analysis         = val_resp_analysis[:, 3, :]
rel_accel_surro    = val_resp_surro[:, 0, :]
rel_vel_surro      = val_resp_surro[:, 1, :]
rel_disp_surro     = val_resp_surro[:, 2, :]
Q_surro            = val_resp_surro[:, 3, :]

def calc_corrcoef(analysis, surro):
    analysis_mean = analysis.mean(axis=1, keepdims=True)
    surro_mean = surro.mean(axis=1, keepdims=True)
    analysis_std = analysis.std(axis=1, keepdims=True)
    surro_std = surro.std(axis=1, keepdims=True)

    # 共分散
    cov = np.mean((analysis-analysis_mean) * (surro-surro_mean), axis=1)

    corrs = cov/(analysis_std.squeeze() * surro_std.squeeze())
    return corrs

corrs_accel = calc_corrcoef(rel_accel_analysis, rel_accel_surro)
corrs_vel =   calc_corrcoef(rel_vel_analysis, rel_vel_surro)
corrs_disp =  calc_corrcoef(rel_disp_analysis, rel_disp_surro)
corrs_Q =     calc_corrcoef(Q_analysis, Q_surro)
corrs = [corrs_accel, corrs_vel, corrs_disp, corrs_Q]
corrs_title = ['relative acceleratioin', 'relative velocity', 'relative displacement', 'lateral force of the spring']

fig, axes = plt.subplots(2, 2, figsize=(15, 15)) # 2行2列に変更。figsizeも調整

plt.rcParams["font.size"] = 20
axes = axes.flatten()  # 2次元配列を1次元に変換 (2x2 = 4つのサブプロットに対応)
moji = ['(1)', '(2)','(3)', '(4)'] # これはパーセンタイル表示のラベル。グラフ自体の通し番号ではない
moji_title = ['(a)', '(b)','(c)', '(d)'] # 各サブプロットのタイトル部分

for i in range(4):
    # 5%, 50%, 95%の線を引く
    percentiles = [5, 50, 95]
    color = [red, orange, green] # 色を文字列に修正
    moji_pos = [20, 30, 40] # テキストのY座標位置

    targets = np.percentile(corrs[i], percentiles)
    results = []
    for p, t in zip(percentiles, targets):
        idx = np.argmin(np.abs(corrs[i]-t))
        closest_value = corrs[i][idx]
        results.append((p, t, closest_value, idx))

    for j in range(len(results)):
        axes[i].axvline(x=results[j][1], color=color[j], linestyle='-', linewidth=1)
        axes[i].text(
            results[j][1], moji_pos[j],
            f"{moji[j]}  {percentiles[j]}%", # moji[j]は1,2,3で回す必要があるので、ここではそのままにしておく
            color=color[j], fontsize=17, alpha=0.7,
            rotation=0, va='top', ha='right'
        )

    axes[i].hist(corrs[i], bins=20, color=blue, alpha=0.8) # 色を文字列に修正
    axes[i].set_xlabel(f'{moji_title[i]}  correlation coefficient\n of {corrs_title[i]}', fontsize=17)
    axes[i].set_ylabel('frequency')
    # axes[i].set_title(corrs_title[i],fontsize=12)  # タイトルに階数を表示
    axes[i].set_ylim(0, 50)
    axes[i].set_xlim(0.7, 1.02 )

plt.tight_layout() # サブプロット間のスペースを調整
plt.show()