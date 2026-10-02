# ライブラリのインポート------------------------------------------------
import copy
import torch.nn as nn
import torch
import math
# --------------------------------------------------------------------

class CustomLinear(nn.Module):
    """
    ## CustomLinear
    :in_features: 入力次元数
    :out_features: 出力次元数
    :past: 何ステップ前まで見るかを表す変数
    :future: 何ステップ未来までを見るかを表す変数
    """
    def __init__(self, in_features, out_features, past, future):        
        super(CustomLinear, self).__init__()
        self.future = future
        self.past = past
        self.linear = nn.Linear(in_features, out_features)
        self.mask = self.create_mask(in_features, out_features)
        # マスクは最初に1回作るだけである．
        self.apply_mask()
        # print('mask was applied')
        
    
    # マスクの作成の高速化
    def create_mask(self, in_features, out_features):
        # 行と列のインデックスを作成
        # deviceの設定
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        row_indices = torch.arange(out_features).unsqueeze(1)
        col_indices = torch.arange(in_features).unsqueeze(0)
        
        # マスクの作成条件をブロードキャストで計算
        mask = torch.logical_and(col_indices < row_indices + self.future, col_indices > row_indices - self.past).float()
        # mask.to('cuda')
        
        return mask

    
    def apply_mask(self):
        with torch.no_grad():
            # print(self.linear.weight.shape, self.mask.shape)
            if self.linear.weight.is_cuda:
                self.linear.weight *= self.mask.to('cuda')
            else:
                self.linear.weight *= self.mask
    
    def forward(self, x):
        self.apply_mask()
        return self.linear(x)
    
    def weight(self):
        return self.linear.weight
    
    def bias(self):
        return self.linear.bias




class IntegrateLayer(nn.Module):
    """
    数値積分を行うレイヤー(重みを学習しないように使用すること)
    """
    def __init__(self, in_features, out_features, dt):
        super(IntegrateLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dt = dt
        self.future = 1
        self.past = in_features

        weight = dt * torch.tril(torch.ones(out_features, in_features))
        # weight = torch.transpose(weight, 0, 1)
        self.linear.weight.data = weight
        self.linear.bias.data = torch.zeros(out_features)
        self.mask = self.create_mask(in_features, out_features)

    def forward(self, x):
        self.apply_mask()
        x = self.linear(x)
        return x
    
    # マスクの作成の高速化
    def create_mask(self, in_features, out_features):
        # 行と列のインデックスを作成
        # deviceの設定
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        row_indices = torch.arange(out_features).unsqueeze(1)
        col_indices = torch.arange(in_features).unsqueeze(0)
        
        # マスクの作成条件をブロードキャストで計算
        mask = torch.logical_and(col_indices < row_indices + self.future, col_indices > row_indices - self.past).float()
        # mask.to('cuda')
        
        return mask

    
    def apply_mask(self):
        with torch.no_grad():
            # print(self.linear.weight.shape, self.mask.shape)
            if self.linear.weight.is_cuda:
                self.linear.weight *= self.mask.to('cuda')
            else:
                self.linear.weight *= self.mask
    
    def weight(self):
        return self.linear.weight
    
    def bias(self):
        return self.linear.bias

class thLayer(nn.Module):
    """
    減衰と剛性定数による条件付けを行うレイヤ(0)
    """
    def __init__(self, out_features):
        super(thLayer, self).__init__()
        self.linear = nn.Linear(1, out_features)

        # weight =torch.zeros(out_features, 1, requires_grad=True)
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        # self.linear.weight.data = weight
        # self.linear.bias.data = torch.zeros(out_features, requires_grad=True)

    def forward(self, x):
        return self.linear(x)


class MNN_layer(nn.Module):
    def __init__(self,
                 in_features, 
                 out_features, 
                 past, 
                 future, 
                 use_Integrate=False
                 ):
        super(MNN_layer, self).__init__()
        self.cl = CustomLinear(in_features, out_features, past, future)
        # self.bn = nn.BatchNorm1d(in_features)
        self.tanh = nn.Tanh()
        # self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.use_Integrate=use_Integrate
    
    def _initialize_weights(self):
        nn.init.kaming_normal_(self.cl.weight(), nonlinearity = 'tanh')
        if self.cl.bias is not None:
            nn.init.zeros_(self.cl.bias())
    
    def forward(self, x):
        x_dash = x
        x = self.cl(x)
        # x = self.bn(x)
        gated_tanh = self.tanh(x)
        gated_sigmoid = self.sigmoid(x)
        # element wise multiplication
        gated = gated_tanh * gated_sigmoid

        # Residual network
        # print(x)
        # print(x_dash)
        x = gated + x_dash

        return x
    

class MNN_layer_TL(nn.Module):
    def __init__(self,
                 in_features, 
                 out_features, 
                 past, 
                 future, 
                 use_Integrate=False
                 ):
        super(MNN_layer_TL, self).__init__()
        self.cl = CustomLinear(in_features, out_features, past, future)
        # self.bn = nn.BatchNorm1d(in_features)
        self.tanh = nn.Tanh()
        # self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.use_Integrate=use_Integrate
    
    def _initialize_weights(self):
        nn.init.kaming_normal_(self.cl.weight(), nonlinearity = 'tanh')
        if self.cl.bias is not None:
            nn.init.zeros_(self.cl.bias())
    
    def forward(self, x):
        x_dash = x
        # x = self.cl(x)
        # gated_tanh = self.tanh(x)
        # gated_sigmoid = self.sigmoid(x)
        # # element wise multiplication
        # gated = gated_tanh * gated_sigmoid

        # Residual network
        # print(x)
        # print(x_dash)
        x = x + x_dash

        return x

class MNN(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 past, 
                 future, 
                 layer, 
                 use_Integrate=False):
        super(MNN, self).__init__()
        
        self.layers = nn.ModuleList([MNN_layer(in_features, out_features, past, future) for i in range(layer)])
        self.tanh = nn.Tanh()
        # self.conv = nn.Conv1d(in_channels=1, 
        #                       out_channels=1, 
        #                       kernel_size=5, 
        #                       padding='same')
        self.use_Integrate=use_Integrate
        if self.use_Integrate:
            self.integrate = IntegrateLayer(in_features, out_features, dt=0.01)

    def forward(self, x):
        for layer in self.layers:
            # print(x.shape)
            x = layer(x)
        
        # x = self.tanh(x)
        if self.use_Integrate:
            x = self.integrate(x)
        return x


class MNN_3(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 past, 
                 future, 
                 layer, 
                 dt):
        super(MNN_3, self).__init__()
        
        self.layers = nn.ModuleList([MNN_layer(in_features, out_features, past, future) for i in range(layer)])
        self.tanh = nn.Tanh()
        self.integrate0 = IntegrateLayer(in_features, out_features, dt=dt)
        self.integrate1 = IntegrateLayer(in_features, out_features, dt=dt)
        self.integrate2 = IntegrateLayer(in_features, out_features, dt=dt)

        # # integrate1, integrate2の重みを固定
        # # CausalLi0near層の重みを固定
        # for param in self.integrate1.parameters():
        #     param.requires_grad = False            
        # for param in self.integrate2.parameters():
        #     param.requires_grad = False


    def forward(self, x):
        for layer in self.layers:
            # print(x.shape)
            x = layer(x)
        x = self.integrate0(x)
        
        
        vel = self.integrate1(x)
        disp = self.integrate2(vel)
        x = x.unsqueeze(dim=0)
        vel = vel.unsqueeze(dim=0)
        vel = vel*5
        disp = disp.unsqueeze(dim=0)
        disp = disp * 10
        # print('x', x.shape, type(x))
        # print('vel', vel.shape, type(vel))
        # print('disp', disp.shape, type(disp))
        
        res = torch.concatenate([x, vel, disp])
        res = torch.transpose(res, 0, 1)
        # print(res.shape)

        return res
    
class MNN_3_res(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 past, 
                 future, 
                 layer, 
                 dt):
        super(MNN_3_res, self).__init__()
        
        self.layers = nn.ModuleList([MNN_layer(in_features, out_features, past, future) for i in range(layer)])
        self.tanh = nn.Tanh()
        self.integrate0 = IntegrateLayer(in_features, out_features, dt=dt)
        self.integrate1 = IntegrateLayer(in_features, out_features, dt=dt)
        self.integrate2 = IntegrateLayer(in_features, out_features, dt=dt)
        # self.integrate1 = MNN_layer(in_features, out_features, in_features, in_features)
        # self.integrate2 = MNN_layer(in_features, out_features, in_features, in_features)

        # # integrate1, integrate2の重みを固定
        # # CausalLinear層の重みを固定
        # for param in self.integrate1.parameters():
        #     param.requires_grad = False            
        # for param in self.integrate2.parameters():
        #     param.requires_grad = False


    def forward(self, x):
        for layer in self.layers:
            # print(x.shape)
            x = layer(x)
        x = self.integrate0(x)
        
        vel = self.integrate1(x)
        disp = self.integrate2(vel)
        x = x.unsqueeze(dim=0)
        vel = vel.unsqueeze(dim=0)
        vel = vel*5
        disp = disp.unsqueeze(dim=0)
        disp = disp * 10
        # print('x', x.shape, type(x))
        # print('vel', vel.shape, type(vel))
        # print('disp', disp.shape, type(disp))
        
        res = torch.concatenate([x, vel, disp])
        res = torch.transpose(res, 0, 1)
        # print(res.shape)

        return res


class MNN_th(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 past, 
                 future,
                 th_in_layer, 
                 layer, 
                 th_init=False):
        super(MNN_th, self).__init__()
        
        self.layers = nn.ModuleList([MNN_layer(in_features, out_features, past, future) for i in range(layer)])
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()
        self.t1_layers = nn.ModuleList([thLayer(in_features) for i in range(th_in_layer)])
        # self.t2_layers = nn.ModuleList([nn.Linear(in_features, in_features) for i in range(layer)])
        self.h1_layers = nn.ModuleList([thLayer(in_features) for i in range(th_in_layer)])
        # self.h2_layers = nn.ModuleList([nn.Linear(in_features, in_features) for i in range(layer)])
        self.integrate = IntegrateLayer(in_features, out_features, dt=0.01)
        self.th_in_layer = th_in_layer

    def forward(self, x, t, h):
        for i in range(len(self.layers)):
            # print(x.shape)
            # print(type(t))
            if i <= self.th_in_layer-1:
                t_out = (self.tanh(self.t1_layers[i](t)))
                h_out = (self.tanh(self.h1_layers[i](h)))
                x = x + t_out + h_out
                # x = self.tanh(x)
            x = self.layers[i](x)
        x = self.integrate(x)
        return x



# 速度加速度チャネルバージョン
class MNN_PINN_channel(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 past, 
                 future, 
                 layer,
                 branch, 
                 dt):
        super(MNN_PINN_channel, self).__init__()

        self.layer=layer
        self.branch=branch

        # self.integrate0 = IntegrateLayer(in_features, out_features, dt=dt)
        # self.integrate1 = IntegrateLayer(in_features, out_features, dt=dt)
        # self.integrate2 = IntegrateLayer(in_features, out_features, dt=dt)
        # self.integrate3 = IntegrateLayer(in_features, out_features, dt=dt)


        # self.layers = nn.ModuleList([MNN_layer(in_features, out_features, past, future) for i in range(layer-branch + branch*4)])
        self.layers = nn.ModuleList([MNN_layer(in_features, out_features, past, future) for i in range(layer-branch)])
        self.tanh = nn.Tanh()
        self.forcelayers = nn.ModuleList([MNN_layer(in_features, out_features, past, future) for i in range(branch)] + [IntegrateLayer(in_features, out_features, dt=dt)])
        self.accellayers = nn.ModuleList([MNN_layer(in_features, out_features, past, future) for i in range(branch)] + [IntegrateLayer(in_features, out_features, dt=dt)])
        self.vellayers = nn.ModuleList([MNN_layer(in_features, out_features, past, future) for i in range(branch)] + [IntegrateLayer(in_features, out_features, dt=dt)])
        self.displayers = nn.ModuleList([MNN_layer(in_features, out_features, past, future) for i in range(branch)] + [IntegrateLayer(in_features, out_features, dt=dt)])

        # displayers.0.cl.linear.weight ~ displayers.(branch-1).cl.linear.weight--------MNN_layer
        # displayers.branch.cl.linear.weight --------integrate_layer


    def forward(self, x):
        for layer in self.layers[:self.layer-self.branch]:
            # print(x.shape)
            x = layer(x)
        cnt = 0

        for layer in self.forcelayers:
            if cnt == 0:
                force1 = layer(x)
            else:
                force1 = layer(force1)
            cnt += 1
        # force1 = self.integrate0(force1)
        force = torch.unsqueeze(force1, dim=0)
        # print("force")

        
        cnt = 0
        for layer in self.accellayers:
            if cnt == 0:
                accel = layer(x)
            else:
                accel = layer(accel)
            cnt += 1
        # accel = self.integrate1(accel)
        accel = torch.unsqueeze(accel, dim=0)
        # print("accel")
        cnt = 0
        for layer in self.displayers:
            if cnt == 0:
                disp = layer(x)
            else:
                disp = layer(disp)
            cnt += 1
        # disp = self.integrate3(disp)
        disp = torch.unsqueeze(disp, dim=0)
        # print("disp")
        
        cnt = 0
        for layer in self.vellayers:
            if cnt == 0:
                vel = layer(x)
            else:
                vel = layer(vel)
            cnt += 1
        # vel = self.integrate3(vel)
        vel = torch.unsqueeze(vel, dim=0)
        
        # print("vel")
        
        res = torch.concatenate([accel, vel, disp, force])
        res = torch.transpose(res, 0, 1)
        # print(res.shape)

        return res
    

class MNN_TL(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 out_channels,
                 past, 
                 future, 
                 layer, 
                 dt):
        """
        in_features:入力時刻歴ステップ数
        out_features: 出力時刻歴ステップ数
        out_channels: 出力チャネル数
        past, MNNの前何ステップ見るか
        future, 未来何ステップ見るか
        dt, 積分時間刻み
        """
        super(MNN_TL, self).__init__()
        self.CNN = nn.Conv1d(in_channels=4, 
                             out_channels=out_channels, 
                             kernel_size=past+future, 
                             stride=1, 
                             padding='same')
        # ここResblockのあるCNNモジュールに直してもいいかもしれん
        self.layers = nn.ModuleList([nn.ModuleList([MNN_layer(in_features, out_features, past, future)for i in range(layer)] + [IntegrateLayer(out_features, out_features, dt=dt)])
                                     for i in range(out_channels)]) #出力4チャネルから予測する
        self.tanh = nn.Tanh()

    def forward(self, x):
        #x (batch_size, channels=4, sequence)
        x = self.CNN(x) # (batch_size, channel*NDOF, sequence)
        x = self.tanh(x)
        batch_size = x.shape[0]
        channels = 4
        NDOF = int(x.shape[1]//4)
        channels_NDOF = x.shape[1]
        sequence = x.shape[2]
        x_out = torch.tensor([[[]]]).to('cuda')
        for i in range(channels_NDOF):
            # 2次元目のチャネルごとにMNNと積分レイヤをかける
            tmp = self.layers[i][0](x[:, i, :])
            for layer in self.layers[i][1:]:
                tmp = layer(tmp) #この書き方でいいのかわからん
            # print(tmp.shape)
            if i != 0:
                x_out = torch.concatenate([x_out, tmp.unsqueeze(dim=1)], dim=1)
            else:
                x_out = tmp.unsqueeze(dim=1)
            
        x_reshaped = x.view(batch_size, channels, NDOF, sequence)
        return x_reshaped


class MNN_for_TL(nn.Module):
    """
    SDOFのMNNなどから転移学習をするためのCNN
    """
    def __init__(self,
                 in_channels, 
                 in_mode, 
                 out_channels,
                 out_mode, 
                 sequence, 
                 past, 
                 future,
                 layer, 
                 use_Integrate):
        super(MNN_for_TL, self).__init__()
        self.layers = nn.ModuleList([nn.ModuleList([MNN(sequence, sequence, past, future,  layer, use_Integrate=use_Integrate) for i in range(out_channels)]) for i in range(out_mode)])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_mode = out_mode


    def forward(self, x):
        """x: # [batch_size, channels=4, sequence]"""
        cnt = 0 # 今何チャネル目か

        for channel in range(self.in_channels):
            if cnt ==0:
                for mode in range(self.out_mode):
                    tmp = self.layers[mode][channel](x[:, channel, :].unsqueeze(dim=1)) # (batch_size, 1, sequence)
                    if mode == 0:
                        y = tmp.unsqueeze(dim=1) #(batch_size, 1, 1, sequence)
                    else:
                        y = torch.concatenate([y, tmp.unsqueeze(dim=1)], dim=2) # (batch_size, 1, NDOF, sequence)
                z = y
            else:
                for mode in range(self.out_mode):
                    tmp = self.layers[mode][channel](x[:, channel, :].unsqueeze(dim=1)) # (batch_size, 1, sequence)
                    if mode == 0:
                        y = tmp.unsqueeze(dim=1) #(batch_size, 1, 1, sequence)
                    else:
                        y = torch.concatenate([y, tmp.unsqueeze(dim=1)], dim=2) # (batch_size, 1, NDOF, sequence)
                z = torch.concatenate([z, y], dim=1) # (batch_size, channel, NDOF, sequence)
            cnt +=1

        return z
