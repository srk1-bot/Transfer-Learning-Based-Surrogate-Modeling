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
from network import CNN
blue = "#0768CF"
orange = '#F4B612'
green = '#6C963A'
red = "#bb2633"


from multiprocessing import Pool

import gc


# キーの設定
for seed in range(3, 10):
# seed = 1
    for num in [10, 20, 40, 80, 120]:
    # for num in [20]:
        # for response in['accel', 'SDR']:
        for response in['SDR', 'accel']:
            # num = 20
            sampling = 'fs'
            response_out = response
            steps=4096

            # 解析モデルの設定 
            model_pass = 'data/build_data/aisc-SMF-PG-4ELF_20.csv'
            size_pass = 'data/build_data/aisc-shapes-database.csv'

            datasets_dir = f'datasets/20SMF/seed{seed}/'
            model_dir = f'model/seed{seed}/'
            tmp_model_dir = f'model_tmp/'
            loss_data_dir = f'loss_data/seed{seed}/'
            # data_dir = 'eq_data/seed_'+str(seed)+'/'

            # train_eq_raw = np.load(data_dir + 'TL_'+sampling+'_train_raw_' + str(num) + '.npy')[:, :steps]
            # train_eq_filt = np.load(data_dir + 'TL_'+sampling+'_train__filt_' + str(num) + '.npy')[:, :steps]

            # valid_eq_raw = np.load(data_dir + 'valid_raw.npy')[:, :steps]
            # valid_eq_filt = np.load(data_dir + 'valid_filt.npy')[:, :steps]
            batch_size_long = 256

            # データセットの作成と保存
            x_max_frame = np.load(datasets_dir + 'SDOF_' + sampling + '_xmax.npy')
            y_max_frame = 0.0

            
            in_channels = 4
            in_mode = 1
            out_channels = 1 #story_displacement floor_acceleration
            out_mode = 20
            kernel_size = 4096//2
            stride = 1
            layer = 1
            lstm_hidden_size = 1
            lstm_num_layers = 1
            use_LSTM =False
            in_features = 4096
            out_features = 4096
            dt = 0.01
            use_integrate=False
            use_masked=True
            use_integrate_in_mask = False
            past = 512
            future=1

            model_1 = CNN.CNN_for_frame(in_channels, 
                                    in_mode, 
                                    out_channels,
                                    out_mode,   
                                    kernel_size, 
                                    stride,
                                    layer,
                                    in_features,
                                    out_features,
                                    dt, 
                                    past, 
                                    future,
                                    use_integrate, 
                                    use_masked,
                                    use_integrate_in_mask
                                    )
        
            model_1.to('cuda')

            # データセットの読み込み
            x_max=np.load(datasets_dir + 'SMF_xmax_'+response+'_'+sampling+'_'+str(num)+'.npy')
            y_max=np.load(datasets_dir + 'SMF_ymax_'+response+'_'+sampling+'_'+str(num)+'.npy')

            train_long = torch.load(datasets_dir+'SMF_train_'+response+'_'+sampling+'_'+str(num)+'.pth', weights_only=False, map_location='cuda')
            valid_long = torch.load(datasets_dir+'SMF_valid_'+response+'_'+sampling+'_'+str(num)+'.pth', weights_only=False, map_location='cuda')

            train_long = DataLoader(train_long, batch_size=batch_size_long, shuffle=False)
            valid_long = DataLoader(valid_long, batch_size=batch_size_long, shuffle=False)

            # SDOF_modelの読み込み
            SDOF_model = torch.load(f'{model_dir}/SDOF_'+sampling + '_model_norm.pth', map_location='cuda', weights_only=False)
            SDOF_model.to('cuda')
            print(None)

            with torch.autograd.set_detect_anomaly(True):
                # PIN ha dummy
                train_loss_data, valid_loss_data, learn_time_1 = learn_model.learn_model_NDOF(model_1, SDOF_model,  500, train_long, valid_long, 1.0, 0, y_max, 0.01,  lr=0.001, int_loss_fac=0.005 , PINN_start=30000, criterion=learn_model.NormalizedHuber(delta=0.3), weight_decay=0.0001, save_model=True, model_dir_name = f'{tmp_model_dir}/SMF_'+sampling+'_'+response_out+'_'+str(num)+'_norm_huber.pth')

            shutil.move(f'{tmp_model_dir}/SMF_'+sampling+'_'+response_out+'_'+str(num)+'_norm_huber.pth', f'{model_dir}/SMF_'+sampling+'_'+response_out+'_'+str(num)+'_norm_huber.pth')
            # 損失関数の保存
            np.save(f'{loss_data_dir}train_loss_'+sampling+'_'+response_out+'_'+str(num)+'_norm.npy', train_loss_data)
            np.save(f'{loss_data_dir}valid_loss_'+sampling+'_'+response_out+'_'+str(num)+'_norm.npy', valid_loss_data)
            np.save(f'{loss_data_dir}learn_time'+sampling+'_'+response_out+'_'+str(num)+'_norm.npy', valid_loss_data)

            torch.save(model_1, f'{model_dir}/SMF_'+sampling+'_'+response_out+'_'+str(num)+'.pth')


            # メモリ開放などなど----------------------------------------------------------------------------
            del model_1, SDOF_model, train_long, valid_long, x_max, y_max
            del train_loss_data, valid_loss_data
            gc.collect()
            torch.cuda.empty_cache()