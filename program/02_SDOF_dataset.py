import sys
from pathlib import Path

root = Path.cwd().parent          # プロジェクトルート
if str(root) not in sys.path:
    sys.path.append(str(root))    # modules の import 用

    
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

import time as ti # 計算コストの評価


from analysis.SDOF_analysis_ops import *
from analysis.make_train_data import *
from analysis.frame_aisc_SMF import *

from network import learn_model
from network import masked_net
from network import CNN

blue = '#5893A0'
orange = '#F4B612'
green = '#6C963A'
red = '#9d363f'

import gc

num = 20
sampling = 'fs'



from multiprocessing import Pool

# def process_seed(seed):
for seed in range(1, 10):
    # 地震動データの読み込み
    data_dir = 'eq_data/dataset/seed_'+str(seed)+'/'
    datasets_dir = f'datasets/20SMF/seed{seed}/'
    if not os.path.exists(datasets_dir):
        os.makedirs( f'datasets/20SMF/seed{seed}/')
    # sfac = 1.0

    train_eq_raw = np.load(data_dir + r'SDOF_fs_train_raw.npy')[:, :4096]
    valid_eq_raw = np.load(data_dir + r'valid_raw.npy')[:, :4096]
    # test_eq_raw = np.load(data_dir + r'test_raw.npy')[:2000, :4096]*sfac


    train_eq_filt = np.load(data_dir + r'SDOF_fs_train_filt.npy')[:, :4096]
    valid_eq_filt = np.load(data_dir + r'valid_filt.npy')[:, :4096]
    # test_eq_filt = np.load(data_dir + r'test_filt.npy')[:2000, :4096]*sfac

    batch_size_long=256

    dt = 0.01
    result = {'m':1.0,'f_yield': 4.326941495004812, 'r_post': 0.3700337242085181, 'h': 0.032054231887479766, 'T': 2.412431911650598}
    m = np.array([result['m']])
    f_yield = np.array([result['f_yield']])
    r_post = np.array([result['r_post']])
    h = result['h']
    T = result['T']
    N=1
    # T = 2.4038627452341843 # 非減衰固有周期s
    omega = np.array([2*np.pi/T])
    dt = 0.01 # 積分時間刻みs
    batch_size = 1024
    batch_size_long = 1024
    c=2*h*m*omega




    # 応答解析してデータの保存を実行
    x_max=0
    y_max=0
    start = ti.time()

    
    dataloader_long_valid, x_max, y_max = make_train_dataset_N(valid_eq_raw, N, m, omega, h, dt, batch_size_long, useGPU=True, normalize=True, f_yield=f_yield, r_post=r_post, x_max=x_max, y_max=y_max, use_filtered=True, EQ_filtered=valid_eq_filt,)
    dataloader_long_train, x_max, y_max = make_train_dataset_N(train_eq_raw, N, m, omega, h, dt, batch_size_long, useGPU=True, normalize=True, f_yield=f_yield, r_post=r_post, x_max=x_max, y_max=y_max, use_filtered=True, EQ_filtered= train_eq_filt)
    dataloader_long_valid, x_max, y_max = make_train_dataset_N(valid_eq_raw, N, m, omega, h, dt, batch_size_long, useGPU=True, normalize=True, f_yield=f_yield, r_post=r_post, x_max=x_max, y_max=y_max, use_filtered=True, EQ_filtered=valid_eq_filt,)

    torch.save(dataloader_long_train.dataset, f'{datasets_dir}SDOF_fs_long_train.pth')
    torch.save(dataloader_long_valid.dataset, f'{datasets_dir}SDOF_fs_long_valid.pth')
    np.save(rf'{datasets_dir}SDOF_fs_xmax.npy', x_max)
    np.save(rf'{datasets_dir}SDOF_fs_ymax.npy', y_max)
    end = ti.time()
    print(end-start)
    print(seed)