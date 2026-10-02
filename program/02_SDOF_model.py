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


for seed in range(1, 10):
    print(f'seed{seed}')

    # 地震動データの読み込み------------------------------------------------------------------
    data_dir = 'eq_data/dataset/seed_'+str(seed)+'/'
    datasets_dir = f'datasets/20SMF/seed{seed}/'
    model_dir = f'model/seed{seed}/'
    tmp_model_dir = 'model_tmp'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    loss_data_dir = f'loss_data/seed{seed}/'
    if not os.path.exists(loss_data_dir):
        os.makedirs(loss_data_dir)
    sfac = 1.0

    train_eq_raw = np.load(data_dir + r'SDOF_fs_train_raw.npy')[:, :4096]*sfac
    valid_eq_raw = np.load(data_dir + r'valid_raw.npy')[:, :4096]
    test_eq_raw = np.load(data_dir + r'test_raw.npy')[:2000, :4096]


    train_eq_filt = np.load(data_dir + r'SDOF_fs_train_filt.npy')[:, :4096]*sfac
    valid_eq_filt = np.load(data_dir + r'valid_filt.npy')[:, :4096]
    test_eq_filt = np.load(data_dir + r'test_filt.npy')[:2000, :4096]

    batch_size_long=256

    # 5.12sの地震動の作成----------------------------------------------------------------
    train_short_num = 1200
    valid_short_num = 800
    test_short_num = 2000
    # train_short_num = 120
    # valid_short_num = 80
    # test_short_num = 2000
    do_amplification=False

    short_T = 5.12
    dt = 0.01


    
    # 一質点系パラメータの設定------------------------------------------------------------
    result =  {'m':1.0,'f_yield': 4.326941495004812, 'r_post': 0.3700337242085181, 'h': 0.032054231887479766, 'T': 2.412431911650598}
    m = np.array([result['m']])
    f_yield = np.array([result['f_yield']])
    r_post = np.array([result['r_post']])
    h = result['h']
    N=1
    T =2.403 # 非減衰固有周期s
    omega = np.array([2*np.pi/T])
    dt = 0.01 # 積分時間刻みs
    batch_size = 1024
    batch_size_long = 1024
    c=2*h*m*omega

    

    # short用のモデルの定義
    branch = 4
    past = 512
    future = 2
    layer = 8

    # データセットの読み込み
    x_max = np.load(f'{datasets_dir}SDOF_fs_xmax.npy')
    y_max = np.load(f'{datasets_dir}SDOF_fs_ymax.npy')

    train_long = torch.load(f'{datasets_dir}SDOF_fs_long_train.pth', weights_only=False)
    val_long = torch.load(f'{datasets_dir}SDOF_fs_long_valid.pth', weights_only=False)

    train_long = DataLoader(train_long, batch_size=batch_size_long, shuffle=False)
    val_long = DataLoader(val_long, batch_size=batch_size_long, shuffle=False)



    model_2 = masked_net.MNN_PINN_channel(512*8, 512*8, past, future, layer, branch, 0.01)

    model_2.to('cuda')
    print('model has been defined')


    
    # trainig -----------------------------------------------------------------------------
    start = ti.time()
    with torch.autograd.set_detect_anomaly(True):
        train_loss_data_2, val_loss_data_2, learn_time_2 = learn_model.learn_model_SDOF(model_2, 1000, train_long, val_long, c, m, y_max, dt,  lr=0.0005, int_loss_fac=0.00000005 , PINN_start=100, criterion=learn_model.NormalizedHuber(delta=0.3), save_model=True, model_dir_name = tmp_model_dir + 'SDOF_fs_model_norm.pth')# weight_decay=0.002)
    end = ti.time()
    print(end-start)

    model_dir_name = model_dir + 'SDOF_fs_model_norm.pth'
    shutil.move(tmp_model_dir + 'SDOF_fs_model_norm.pth',model_dir + 'SDOF_fs_model_norm.pth')

    np.save(f'{loss_data_dir}train_loss_fs_SDOF_{seed}.npy', train_loss_data_2)
    np.save(f'{loss_data_dir}val_loss_fs_SDOF_{seed}.npy', val_loss_data_2)


    # メモリ開放などなど----------------------------------------------------------------------------
    del model_2, train_eq_filt, train_eq_raw, valid_eq_filt, valid_eq_raw
    gc.collect()

    torch.cuda.empty_cache()