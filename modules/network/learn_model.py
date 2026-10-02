import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import time as ti
from network.masked_net import *
# from network.soft_dtw_cuda import SoftDTW


# モデルの学習を行う関数
def learn_model(model, num_epochs,
                train_dataloader, 
                val_dataloader,  
                criterion=nn.MSELoss(),lr = 0.001, 
                flatten = False,
                ):
    """
    モデルの学習を行う関数．train_lossとvalidation_lossの経過を返す．
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start = ti.perf_counter()
    train_loss_data = []
    val_loss_data = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # 順伝搬
            outputs = model(inputs)
            outputs = outputs.unsqueeze(dim=1)
            targets = targets.unsqueeze(dim=1)
            print(outputs.shape)
            print(targets.shape)
            # print(inputs.shape, 'input')
            # print(targets.shape, 'target')
            # print(outputs.shape, 'output')


            # 損失の計算
            loss = criterion(outputs, targets)

            # 逆伝搬とオプティマイザのステップ
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avf_train_loss = running_loss/len(train_dataloader.dataset)*1000

        # 検証用データセットでの評価
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            avf_val_loss = val_loss / len(val_dataloader.dataset)*1000

            train_loss_data.append(avf_train_loss)
            val_loss_data.append(avf_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]', 
              f'Step [{(i + 1 )/len(train_dataloader)}]', 
              f'Train_Loss:{avf_train_loss : .4f}', 
              f'Validation Loss :{avf_val_loss:.4f}')
    print('Finished Training')

    end = ti.perf_counter()
    learn_time = end-start
    return train_loss_data, val_loss_data, learn_time


# モデルの学習を行う関数
def learn_model_hk(model, num_epochs,
                train_dataloader, 
                val_dataloader,  
                criterion=nn.MSELoss(),lr = 0.001, 
                ):
    """
    モデルの学習を行う関数．train_lossとvalidation_lossの経過を返す．
    hとkも含めて学習させる
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start = ti.perf_counter()
    train_loss_data = []
    val_loss_data = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, targets, m_omega_2, h) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # 順伝搬
            # print(m_omega_2.shape, h.shape)
            outputs = model(inputs, m_omega_2, h)


            # 損失の計算
            loss = criterion(outputs, targets)

            # 逆伝搬とオプティマイザのステップ
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avf_train_loss = running_loss/len(train_dataloader.dataset)*1000

        # 検証用データセットでの評価
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, m_omega_2, h in val_dataloader:
                outputs = model(inputs, m_omega_2, h)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avf_val_loss = val_loss / len(val_dataloader.dataset)*1000

        train_loss_data.append(avf_train_loss)
        val_loss_data.append(avf_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]', 
              f'Step [{(i + 1 )/len(train_dataloader)}]', 
              f'Train_Loss:{avf_train_loss : .4f}', 
              f'Validation Loss :{avf_val_loss:.4f}')
    print('Finished Training')

    end = ti.perf_counter()
    learn_time = end-start
    return train_loss_data, val_loss_data, learn_time




# モデルの学習を行う関数
def learn_model_4_PINNs(model, num_epochs,
                train_dataloader, 
                val_dataloader,  
                c, 
                m, 
                dt, 
                int_loss_fac = 0.05, 
                PINN_start = 60, 
                criterion=nn.MSELoss(),lr = 0.001, 
                ):
    """
    モデルの学習を行う関数．train_lossとvalidation_lossの経過を返す．
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start = ti.perf_counter()
    train_loss_data = []
    val_loss_data = []

    # 積分関係を入れるかどうか
    # 積分関係
    integrate = IntegrateLayer(in_features=train_dataloader.dataset[0][1].shape[1], 
                                out_features=train_dataloader.dataset[0][1].shape[1], 
                                dt=dt)
    integrate.to('cuda')
    for param in integrate.parameters():
        # integrateのパラメータは固定する．
        param.requires_grad = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # 順伝搬
            outputs = model(inputs)# (batch_size,  channels=4, timestep)
            outputs_trans = torch.transpose(outputs, 0, 1) #([4, time])
            outputs = outputs.unsqueeze(dim=1) # [batchsize, 1, 4, timestep]
            targets = targets.unsqueeze(dim=1) # [batchsize, 1, 4, timestep]

            accel, vel, disp, force = outputs_trans
            zeros = torch.zeros_like(outputs_trans[0])
            # accel = accel # -10~10
            # vel = vel*2.0 # -2~2
            # disp = disp/5.0 # -0.2 ~ 0.2
            # force = m*9.8*force # -m*9.8 ~ m*9.8


            # 運動方程式 m ddy + cdy + Q +m ddy0
            # equation_of_motion = m*accel + c*vel + force + m*inputs
            # print(accel.shape)

            # 積分関係
            int_accel = integrate.forward(accel.detach()) * 10.0 / 2.0
            int_vel = integrate.forward(vel.detach()) * 2.0 * 5.0
            # 積分して出てきたoutput
            int_outputs = torch.concatenate([torch.unsqueeze(accel, dim=0), 
                                             torch.unsqueeze(int_accel, dim=0), 
                                             torch.unsqueeze(int_vel, dim=0), 
                                             torch.unsqueeze(force, dim=0)]) # [4, batchsize, timestep]
            int_outputs2 = torch.transpose(int_outputs, 0, 1) #[batchsize, 4, timestep]
            int_outputs3 = torch.unsqueeze(int_outputs2, dim=1) #[batchsize, 1, 4, timestep]

            # エネルギー保存則
            # integrate = torch.tril(torch.ones((len(vel), len(vel)))) * dt
            # energy = (1*m*vel**2)/2 + integrate@(c*vel**2)-integrate@((-m*gm)*y_dot) + integrate@(Q*9.8*y_dot)


            # # 損失の計算
            if epoch <= PINN_start:
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs, targets) + int_loss_fac * criterion(int_outputs3, outputs)
            running_loss += loss.item()
            
            # 積分関係のロス
            int_loss = criterion(int_outputs3, outputs)

            # 逆伝搬とオプティマイザのステップ
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avf_train_loss = running_loss/len(train_dataloader.dataset)*1000

        # 検証用データセットでの評価
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            avf_val_loss = val_loss / len(val_dataloader.dataset)*1000

            train_loss_data.append(avf_train_loss)
            val_loss_data.append(avf_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]', 
              f'Step [{(i + 1 )/len(train_dataloader)}]', 
              f'Train_Loss:{avf_train_loss : .4f}', 
              f'Validation Loss :{avf_val_loss:.4f}')
            print('integrate_loss=', int_loss.item())
    print('Finished Training')

    end = ti.perf_counter()
    learn_time = end-start
    return train_loss_data, val_loss_data, learn_time

# モデルの学習を行う関数
def learn_model_th(model, num_epochs,
                train_dataloader, 
                val_dataloader,  
                criterion=nn.MSELoss(),lr = 0.001, 
                ):
    """
    モデルの学習を行う関数．train_lossとvalidation_lossの経過を返す．
    hとkも含めて学習させる
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start = ti.perf_counter()
    train_loss_data = []
    val_loss_data = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, targets, t, h) in enumerate(train_dataloader):
            optimizer.zero_grad()
            t = t.unsqueeze(dim = 1)
            h = h.unsqueeze(dim = 1)

            # 順伝搬
            # print(m_omega_2.shape, h.shape)
            # print(t.shape, 't')
            # print(h.shape, 'h')
            outputs = model(inputs, t, h)
            # print(outputs.shape, "output")
            # print(targets.shape, 'target')


            # 損失の計算
            loss = criterion(outputs, targets)

            # 逆伝搬とオプティマイザのステップ
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avf_train_loss = running_loss/len(train_dataloader.dataset)*1000

        # 検証用データセットでの評価
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, t, h in val_dataloader:
                t = t.unsqueeze(dim = 1)
                h = h.unsqueeze(dim = 1)
                outputs = model(inputs, t, h)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            avf_val_loss = val_loss / len(val_dataloader.dataset)*1000

            train_loss_data.append(avf_train_loss)
            val_loss_data.append(avf_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]', 
              f'Step [{(i + 1 )/len(train_dataloader)}]', 
              f'Train_Loss:{avf_train_loss : .4f}', 
              f'Validation Loss :{avf_val_loss:.4f}')
    print('Finished Training')

    end = ti.perf_counter()
    learn_time = end-start
    return train_loss_data, val_loss_data, learn_time





# -----------------------------------------------------
# NDOF_PINNのモデルを学習する関数
def learn_model_SDOF(model, 
                  num_epochs,
                  train_dataloader, 
                  val_dataloader, 
                  c, 
                  m, 
                  y_max, 
                  dt, 
                  int_loss_fac=0.05, 
                  PINN_start = 60, 
                  criterion=nn.MSELoss(), 
                  lr = 0.001, 
                  weight_decay = 0.00, 
                  save_model=False,
                  model_dir_name = ''):
    """
    NDOFのPINNsを学習するための関数。
    train_lossとvalidation_lossの経過を返す
    """
    optimizer= optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    start = ti.perf_counter()
    train_loss_data = []
    val_loss_data = []

    # 積分関係 # (timestep, timestep)の0.01した三角行列
    integrate = IntegrateLayer(in_features=train_dataloader.dataset[0][1].shape[2], 
                               out_features=train_dataloader.dataset[0][1].shape[2], 
                               dt=dt)
    integrate.to('cuda')
    for param in integrate.parameters():
        param.requires_grad=False
    min_loss = 1.0e5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # inputs--(batch_size, timestep)
            # targets---(batch_size, channels=4, number_of_DOF=1, timestep)
            outputs = model(inputs) # (batch_size,  channels=4, timestep)
            if len(outputs.shape) == 2:
                outputs = outputs.unsqueeze(dim=1)
            outputs= outputs.unsqueeze(dim=2) # (batch_size, channels=4, number_of_DOF=1, timestep)
            # print(outputs.shape, targets.shape)
            if epoch >= PINN_start:

                outputs_trans = torch.transpose(outputs, 1, 3) # (batch_size, channels=4, NDOF=1, timestep)

                # 積分関係のモデル化--------------------------
                y_max_torch = torch.tensor(np.array(y_max, dtype=np.float32))
                # y_max_torch | (channels, NDOF)
                y_max_torch = y_max_torch.unsqueeze(dim=0)
                y_max_torch = y_max_torch.unsqueeze(dim=3)
                y_max_torch = y_max_torch.to('cuda')
                output_real = outputs * y_max_torch #(batch_size, channels, number_of_DOF, sequence)
                output_integrated = integrate(output_real) #(batch_size, channels, number_of_DOF, sequence)
                int_ddy_hat = output_integrated[:, 0, :, :] #(batch_size, channels=1, sequence)
                int_dy_hat = output_integrated[:, 1, :, :] #(batch_size, channels=1, sequence)
                ddy = output_real[:, 0, :, :] #(batch_size, channels=1, sequence)
                Q = output_real[:, 3, :, :]#(batch_size, channels=1, sequence)
                output_int = torch.concatenate([ddy, int_ddy_hat, int_dy_hat, Q], dim=1) #(batch_size, channels=4, sequence)
                output_int = output_int.unsqueeze(dim=2) #(batch_size, channels=4, NDOF=1 , sequence) NDOFならコメントアウト

                # 積分のロス計算用の関数
                zeros = torch.zeros_like(output_real)
                int_loss = criterion(output_real-output_int, zeros)


            # 運動方程式の損失----------------------------



            # 平均2乗誤差
            # outputs = outputs.squeeze()
            # print(outputs.shape)
            # targets = targets.squeeze()
            # print(targets.shape)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # 逆伝搬とオプティマイザのステップ
            if epoch >= PINN_start:
                loss += int_loss_fac *int_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avf_train_loss = running_loss/len(train_dataloader.dataset)*1000

        # 検証用データセットでの評価
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                outputs = model(inputs)
                if len(outputs.shape) == 2:
                    outputs = outputs.unsqueeze(dim=1)
                outputs = outputs.unsqueeze(dim=2)
                # print(outputs.shape)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            avf_val_loss = val_loss / len(val_dataloader.dataset)*1000

            train_loss_data.append(avf_train_loss)
            val_loss_data.append(avf_val_loss)
        if epoch > 50 and min_loss >= avf_val_loss and save_model == True:
            min_loss = avf_val_loss
            torch.save(model, model_dir_name)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]', 
              f'Step [{(i + 1 )/len(train_dataloader)}]', 
              f'Train_Loss:{avf_train_loss : .4f}', 
              f'Validation Loss :{avf_val_loss:.4f}')
            if epoch >= PINN_start:
                print('integrate_loss=', int_loss.item())
    print('Finished Training')

    end = ti.perf_counter()
    learn_time = end-start
    return train_loss_data, val_loss_data, learn_time

# -----------------------------------------------------
# NDOF_PINNのモデルを学習する関数
def learn_model_NDOF(model, 
                     base_model,  
                  num_epochs,
                  train_dataloader, 
                  val_dataloader, 
                  c, 
                  m, 
                  y_max, 
                  dt, 
                  int_loss_fac=0.05, 
                  PINN_start = 60, 
                  criterion=nn.MSELoss(), 
                  lr = 0.001, 
                  weight_decay = 0.0, 
                  outputs_int_loss=False, 
                  save_model=False, 
                  model_dir_name = ''):
    """
    NDOFのPINNsを学習するための関数。
    train_lossとvalidation_lossの経過を返す
    model - 学習するCNN
    base_model - SDOFの応答予測のモデル
    """
    optimizer= optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    start = ti.perf_counter()
    train_loss_data = []
    val_loss_data = []

    # 積分関係 # (timestep, timestep)の0.01した三角行列
    integrate = IntegrateLayer(in_features=train_dataloader.dataset[0][1].shape[2], 
                               out_features=train_dataloader.dataset[0][1].shape[2], 
                               dt=dt)
    integrate.to('cuda')
    for param in integrate.parameters():
        param.requires_grad=False
    
    min_loss = 1.0e5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # inputs--(batch_size, timestep)
            # targets---(batch_size, channels=4, number_of_DOF, timestep)
            # with torch.no_grad():
            #     outputs = base_model(inputs) # (batch_size,  channels=4, timestep)
                # print(outputs.shape)
            outputs = base_model(inputs) # (batch_size,  channels=4, timestep)
            outputs = model(outputs) # (batch_size, channels, NDOF, sequence)
            if epoch >= PINN_start:
                outputs_trans = torch.transpose(outputs, 1, 3) # (batch_size, channels=4, NDOF=1, timestep)

                # 積分関係のモデル化--------------------------
                y_max_torch = torch.tensor(np.array(y_max, dtype=np.float32))
                # y_max_torch | (channels, NDOF)
                y_max_torch = y_max_torch.unsqueeze(dim=0)
                y_max_torch = y_max_torch.unsqueeze(dim=3)
                y_max_torch = y_max_torch.to('cuda')
                # print(outputs.shape)
                # print(y_max_torch.shape)
                # _realといっているが，これはモデル本体から出てきたものそのままであるという意味
                output_real = outputs * y_max_torch #(batch_size, channels, number_of_DOF, sequence)
                targets_real = targets * y_max_torch
                output_integrated = integrate(output_real) #(batch_size, channels, number_of_DOF, sequence)
                int_ddy_hat = output_integrated[:, 0, :, :].unsqueeze(dim=1) #(batch_size, 1, NDOF, sequence)
                int_dy_hat = output_integrated[:, 1, :, :].unsqueeze(dim=1) #(batch_size, 1, NDOF, sequence)
                ddy = output_real[:, 0, :, :].unsqueeze(dim=1) #(batch_size, 1, NDOF, sequence)
                dy = output_real[:, 1, :, :].unsqueeze(dim=1) #(batch_size, 1, NDOF, sequence)
                Q = output_real[:, 3, :, :].unsqueeze(dim=1) # (batch_size, 1, NDOF, sequence)
                # output_int = torch.concatenate([ddy, int_ddy_hat, int_dy_hat, Q], dim=1) #(batch_size, channels=4, NDOF,  sequence)
                output_int = torch.concatenate([ddy, dy, int_dy_hat, Q], dim=1) #(batch_size, channels=4, NDOF,  sequence)
                # output_real = torch.concatenate([ddy, dy, int_dy_hat, Q], dim=1) #(batch_size, channels=4, NDOF,  sequence)
                # 積分のロス計算用の関数
                zeros = torch.zeros_like(output_real)
                int_loss = int_loss_fac*criterion(output_real-output_int, zeros)


            # 運動方程式の損失----------------------------



            # 平均2乗誤差
            # print('outputs', outputs.shape)
            # print('target', targets.shape)
            # print(outputs.shape[3])
            targets = targets[:outputs.shape[0], :outputs.shape[1], :outputs.shape[2], :outputs.shape[3]]
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # 逆伝搬とオプティマイザのステップ
            if epoch >= PINN_start:
                loss += int_loss
            loss.backward()
            # del loss
            optimizer.step()

            running_loss += loss.item()
        avf_train_loss = running_loss

        # 検証用データセットでの評価
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                outputs = base_model(inputs)
                # print(outputs.shape)
                outputs = model(outputs) # (batch_size, channels, NDOF, sequence)
                targets = targets[:outputs.shape[0], :outputs.shape[1], :outputs.shape[2], :outputs.shape[3]]
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            avf_val_loss = val_loss

            train_loss_data.append(avf_train_loss)
            val_loss_data.append(avf_val_loss)
        if avf_val_loss <= min_loss and save_model == True:
            min_loss = avf_val_loss
            torch.save(model, model_dir_name)


        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]', 
              f'Step [{(i + 1 )/len(train_dataloader)}]', 
              f'Train_Loss:{avf_train_loss : .4f}', 
              f'Validation Loss :{avf_val_loss:.4f}')
            if outputs_int_loss:
                print('integrate_loss=', int_loss.item())
    print('Finished Training')

    end = ti.perf_counter()
    learn_time = end-start
    return train_loss_data, val_loss_data, learn_time


def learn_freqnet(model, 
                  basemodels, 
                  num_epochs, 
                  train_dataloader, 
                  val_dataloader, 
                  y_max, 
                  dt, 
                  int_loss_fac=0.05, 
                  PINN_start=60, 
                  criterion=nn.MSELoss(), 
                  lr = 0.001):
    """
    freq_netの学習をするための関数
    """
    optimizer= optim.Adam(model.parameters(), lr=lr)
    start = ti.perf_counter()
    train_loss_data = []
    val_loss_data = []

    # 積分関係 # (timestep, timestep)の0.01した三角行列
    integrate = IntegrateLayer(in_features=train_dataloader.dataset[0][1].shape[2], 
                               out_features=train_dataloader.dataset[0][1].shape[2], 
                               dt=dt)
    integrate.to('cuda')
    for param in integrate.parameters():
        param.requires_grad=False
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            with torch.no_grad():
                tmp = []
                for base_model in basemodels:
                    base_model.train()
                    tmp.append(base_model(inputs).unsqueeze(dim=2)) # (batch_size,  channels=4, NDOF, timestep)
            
            base_output = torch.concatenate(tmp, dim=2)
            
            outputs = model(base_output) # (batch_size, channels, NDOF, sequence)

            outputs_trans = torch.transpose(outputs, 1, 3) # (batch_size, channels=4, NDOF=1, timestep)

            # 積分関係のモデル化--------------------------
            # y_max_torch = torch.tensor(np.array(y_max, dtype=np.float32))
            # buturiryou = outputs = 


            # 運動方程式の損失----------------------------



            # 平均2乗誤差
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # 逆伝搬とオプティマイザのステップ
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avf_train_loss = running_loss/len(train_dataloader.dataset)*1000

         # 検証用データセットでの評価
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                tmp = []
                for base_model in basemodels:
                    tmp.append(base_model(inputs).unsqueeze(dim=2)) # (batch_size,  channels=4, NDOF, timestep)
                base_output = torch.concatenate(tmp, dim=2)
            
                outputs = model(base_output) # (batch_size, channels, NDOF, sequence)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            avf_val_loss = val_loss / len(val_dataloader.dataset)*1000

            train_loss_data.append(avf_train_loss)
            val_loss_data.append(avf_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]', 
              f'Step [{(i + 1 )/len(train_dataloader)}]', 
              f'Train_Loss:{avf_train_loss : .4f}', 
              f'Validation Loss :{avf_val_loss:.4f}')
            # print('integrate_loss=', int_loss.item())
    print('Finished Training')
    end = ti.perf_counter()
    learn_time = end-start
    return train_loss_data, val_loss_data, learn_time


# カスタム損失関数
class NormalizedMSE(nn.Module):
    def __init__(self):
        super(NormalizedMSE, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target):
        # 各サンプルの最大絶対値で正規化
        target_max_amp = torch.max(torch.abs(target), dim=3, keepdim=True)[0]
        # 0で割るを防ぐために小さな値を加える
        target_max_amp = target_max_amp + 1e-6

        # 正規化
        normalized_pred = pred / target_max_amp
        normalized_target = target/target_max_amp
        # 正規化されたデータでMSEを計算
        # reduction = 'sum' or 'mean'を設定
        loss = self.mse(normalized_pred, normalized_target).mean()
        return loss


# カスタム損失関数
class NormalizedHuber(nn.Module):
    def __init__(self, delta):
        super(NormalizedHuber, self).__init__()
        self.mse = nn.HuberLoss(reduction='none', delta=delta)
    
    def forward(self, pred, target):
        # 各サンプルの最大絶対値で正規化
        target_max_amp = torch.max(torch.abs(target), dim=3, keepdim=True)[0]
        # 0で割るを防ぐために小さな値を加える
        target_max_amp = target_max_amp + 1e-6

        # 正規化
        normalized_pred = pred / target_max_amp
        normalized_target = target/target_max_amp
        # 正規化されたデータでMSEを計算
        # reduction = 'sum' or 'mean'を設定
        loss = self.mse(normalized_pred, normalized_target).mean()
        return loss
    



# -----------------------------------------------------
# NDOF_PINNのモデルを学習する関数
def learn_model_NDOF_dist(model, 
                     base_model,  
                  num_epochs,
                  train_dataloader, 
                  val_dataloader, 
                  c, 
                  m, 
                  y_max, 
                  dt, 
                  int_loss_fac=0.05, 
                  PINN_start = 60, 
                  criterion=nn.MSELoss(), 
                  lr = 0.001, 
                  weight_decay = 0.0, 
                  outputs_int_loss=False, 
                  save_model=False, 
                  model_dir_name = '',
                  base_model_dev = 'cuda:0', 
                  model_dev = 'cuda:1'):
    """
    NDOFのPINNsを学習するための関数。
    train_lossとvalidation_lossの経過を返す
    model - 学習するCNN
    base_model - SDOFの応答予測のモデル
    二つのGPUに分散して配置してメモリのオーバーフローを予防
    """
    optimizer= optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    start = ti.perf_counter()
    train_loss_data = []
    val_loss_data = []
    
    print(train_dataloader.dataset)
    # 積分関係 # (timestep, timestep)の0.01した三角行列
    integrate = IntegrateLayer(in_features=train_dataloader.dataset[0][1].shape[2], 
                               out_features=train_dataloader.dataset[0][1].shape[2], 
                               dt=dt)
    integrate.to(model_dev)
    for param in integrate.parameters():
        param.requires_grad=False
    
    min_loss = 1.0e5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # inputをbase_modelがあるcuda_1に移動
            inputs = inputs.to(base_model_dev)
            with torch.no_grad():
                outputs = base_model(inputs) # (batch_size,  channels=4, timestep)
            outputs = outputs.to(model_dev)
            outputs = model(outputs) # (batch_size, channels, NDOF, sequence)
            targets = targets.to(model_dev)
            if epoch >= PINN_start:
                outputs_trans = torch.transpose(outputs, 1, 3) # (batch_size, channels=4, NDOF=1, timestep)

                # 積分関係のモデル化--------------------------
                y_max_torch = torch.tensor(np.array(y_max, dtype=np.float32))
                # y_max_torch | (channels, NDOF)
                y_max_torch = y_max_torch.unsqueeze(dim=0)
                y_max_torch = y_max_torch.unsqueeze(dim=3)
                y_max_torch = y_max_torch.to(model_dev)
                # print(outputs.shape)
                # print(y_max_torch.shape)
                output_real = outputs * y_max_torch #(batch_size, channels, number_of_DOF, sequence)
                targets_real = targets * y_max_torch
                output_integrated = integrate(output_real) #(batch_size, channels, number_of_DOF, sequence)
                int_ddy_hat = output_integrated[:, 0, :, :].unsqueeze(dim=1) #(batch_size, 1, NDOF, sequence)
                int_dy_hat = output_integrated[:, 1, :, :].unsqueeze(dim=1) #(batch_size, 1, NDOF, sequence)
                ddy = output_real[:, 0, :, :].unsqueeze(dim=1) #(batch_size, 1, NDOF, sequence)
                dy = output_real[:, 1, :, :].unsqueeze(dim=1) #(batch_size, 1, NDOF, sequence)
                Q = output_real[:, 3, :, :].unsqueeze(dim=1) # (batch_size, 1, NDOF, sequence)
                # output_int = torch.concatenate([ddy, int_ddy_hat, int_dy_hat, Q], dim=1) #(batch_size, channels=4, NDOF,  sequence)
                output_int = torch.concatenate([ddy, dy, int_dy_hat, Q], dim=1) #(batch_size, channels=4, NDOF,  sequence)
                # output_real = torch.concatenate([ddy, dy, int_dy_hat, Q], dim=1) #(batch_size, channels=4, NDOF,  sequence)
                # 積分のロス計算用の関数
                zeros = torch.zeros_like(output_real)
                int_loss = int_loss_fac*criterion(output_real-output_int, zeros)


            # 運動方程式の損失----------------------------



            # 平均2乗誤差
            # print('outputs', outputs.shape)
            # print('target', targets.shape)
            # print(outputs.shape[3])
            targets = targets[:outputs.shape[0], :outputs.shape[1], :outputs.shape[2], :outputs.shape[3]]
            targets = targets.to(model_dev)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # 逆伝搬とオプティマイザのステップ
            if epoch >= PINN_start:
                loss += int_loss
            loss.backward()
            # del loss
            optimizer.step()

            running_loss += loss.item()
        avf_train_loss = running_loss

        # 検証用データセットでの評価
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs.to(base_model_dev)
                outputs = base_model(inputs)
                # print(outputs.shape)
                outputs = outputs.to(model_dev)
                outputs = model(outputs) # (batch_size, channels, NDOF, sequence)
                targets = targets[:outputs.shape[0], :outputs.shape[1], :outputs.shape[2], :outputs.shape[3]]
                targets = targets.to(model_dev)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            avf_val_loss = val_loss

            train_loss_data.append(avf_train_loss)
            val_loss_data.append(avf_val_loss)
        if avf_val_loss <= min_loss and save_model == True:
            min_loss = avf_val_loss
            torch.save(model, model_dir_name)


        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]', 
              f'Step [{(i + 1 )/len(train_dataloader)}]', 
              f'Train_Loss:{avf_train_loss : .4f}', 
              f'Validation Loss :{avf_val_loss:.4f}')
            if outputs_int_loss:
                print('integrate_loss=', int_loss.item())
    print('Finished Training')

    end = ti.perf_counter()
    learn_time = end-start
    return train_loss_data, val_loss_data, learn_time



# SDOFの設定
#(batch_size, channels=4, NDOF=1 , sequence) NDOFならコメントアウト