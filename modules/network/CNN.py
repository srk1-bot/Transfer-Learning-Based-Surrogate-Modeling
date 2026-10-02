# import modules ----------------------------------------------
import sys
sys.path.append(r'/home/ishikawa/ドキュメント/m1')
import copy
from modules.network import masked_net
import torch.nn as nn
import torch
import math
import numpy
# -------------------------------------------------------------

# architecture of convolutional neural net for non-linear time history response analysis
# 


class Residual_block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride
                 ):
        super(Residual_block, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding='same',
                              dilation=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_dash = x
        x = self.conv(x)
        gated_tanh = self.tanh(x)
        gated_sigmoid = self.sigmoid(x)
        gated = gated_tanh * gated_sigmoid

        x = gated + x_dash
        x = self.tanh(x)

        return x

def pad_left(x, pad_size):
    return nn.functional.pad(x, (pad_size, 0))



class CNN(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,  
                 kernel_size, 
                 stride,
                 layer, 
                 lstm_hidden_size, 
                 lstm_num_layers, 
                 use_LSTM = True):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList([Residual_block(in_channels, out_channels, kernel_size=kernel_size, stride=stride) for i in range(layer)]
                                     + [Residual_block(1, 1, 490, 1)])
        self.tanh = nn.Tanh()

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=out_channels, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True)
        
        # Fully connected Layer to map LSTM output to final output size
        self.fc = nn.Linear(lstm_hidden_size, 1)
        self.use_LSTM= use_LSTM
        self.out_channels = out_channels


    def forward(self, x):
        
        x = x.unsqueeze(dim=0)
        x = torch.transpose(x, 0, 1) # [batch, feature, sequence]
        for layer in self.layers:
            x = layer(x)
        
        
        if self.use_LSTM:
            x = torch.transpose(x, 1, 2) #[batch, sequence, feature]
            lstm_out, _ = self.lstm(x)
            x = lstm_out[:, :, :]
            batch_size = x.shape[0]
            sequence = x.shape[1]
            hidden_size = x.shape[2]
            x = torch.reshape(x, [batch_size*sequence, hidden_size])
            x = self.fc(x)
            x = torch.reshape(x, [batch_size, sequence, self.out_channels])
            x = torch.transpose(x, 1, 2) # [batch, feature, sequence]
        else:
            x = torch.transpose(x, 0, 1)

        x = x.squeeze()
        return x

class CNN_for_TL(nn.Module):
    """
    SDOFのMNNなどから転移学習をするためのCNN
    inchannelと同じチャネル数で出力する前提で作られている。
    """
    def __init__(self,
                 in_channels, 
                 in_mode, 
                 out_channels,
                 out_mode,   
                 kernel_size, 
                 stride,
                 layer, 
                 lstm_hidden_size, 
                 lstm_num_layers, 
                 use_LSTM = True):
        super(CNN_for_TL, self).__init__()
        # self.layers = nn.ModuleList([nn.ModuleList([Residual_block(in_channels=in_mode, 
        #                                         out_channels=out_mode, 
        #                                         kernel_size=kernel_size, 
        #                                         stride=stride, )] + 
        #                             [Residual_block(out_mode, 
        #                                             out_mode, 
        #                                             kernel_size=kernel_size, 
        #                                             stride=stride) for i in range(layer-1)])])
        self.layers = nn.ModuleList([nn.ModuleList([nn.Conv1d(in_channels=in_mode, 
                                                out_channels=out_mode, 
                                                kernel_size=kernel_size, 
                                                stride=stride, padding="same")] + 
                                    [nn.Conv1d(out_mode, 
                                                    out_mode, 
                                                    kernel_size=kernel_size, 
                                                    stride=stride, padding="same") for i in range(layer-1)]) for i in range(in_channels)])
        self.tanh = nn.Tanh()

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=out_channels*out_mode, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True)
        
        # Fully connected Layer to map LSTM output to final output size
        self.fc = nn.Linear(lstm_hidden_size, out_channels*out_mode)
        self.use_LSTM= use_LSTM
        self.out_channels = out_channels*out_mode


    def forward(self, x):
        """x: # [batch_size, channels=4, sequence]"""
        cnt = 0 # 今何チャネル目か
        
        for channel in range(x.shape[1]): #　現在は1次モードまでを想定
            # print(channel)
            # print(len(self.layers), print(x.shape))
            tmp = self.layers[channel][0](x[:, channel, :].unsqueeze(dim=1)) # (batch_size,2, sequence)
            
            for layer in self.layers[channel][1:]:
                tmp = layer(tmp)
            if cnt == 0:
                y = tmp.unsqueeze(dim=1)
            else:
                y = torch.concatenate([y, tmp.unsqueeze(dim=1)], dim=1) # (batch_size, channel, N=2, sequence)
            cnt +=1
        
        if self.use_LSTM:
            batch_size = y.shape[0]
            channel = y.shape[1]*y.shape[2]
            sequence = y.shape[3]
            y = y.view(batch_size, channel, sequence)
            # print(y.shape)
            x = torch.transpose(y, 1, 2) #[batch, sequence, feature]
            lstm_out, _ = self.lstm(x)
            x = lstm_out[:, :, :] #[batch, sequence, hidden_size]
            # print(x.shape)
            batch_size = x.shape[0]
            sequence = x.shape[1]
            hidden_size = x.shape[2]
            x = torch.reshape(x, [batch_size*sequence, hidden_size])
            x = self.fc(x)
            x = torch.reshape(x, [batch_size, sequence, self.out_channels])
            x = torch.transpose(x, 1, 2) # [batch, feature, sequence]

            batch_size = x.shape[0]
            channels=4
            NDOF = int(x.shape[1]/channels)
            sequence=x.shape[2]
            x_reshaped = x.view(batch_size, channels, NDOF, sequence)
            return x_reshaped
        # return x_reshaped # (batch_size, channels, NDOF, sequence)
        else:
            return y


class element_sum_model(nn.Module):
    def __init__(self, 
                 model_1, 
                 model_2):
        super(element_sum_model, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
    
    def forward(self, x):
        return (1.0*self.model_1(x) + 0.0*self.model_2(x))
    

class concatenate_model(nn.Module):
    def __init__(self,
                 model_1, 
                 model_2):
        super(concatenate_model, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
    
    def forward(self, x):
        return self.model_2(self.model_1(x))

class ident_func(nn.Module):
    def __init__(self,
                  
                 ):
        super(ident_func, self).__init__()
    
    def forward(self, x):
        return x





class CNN_for_frame(nn.Module):
    """
    SDOFのMNN等から転移学習をして任意のチャネル数
    と自由度を予測するためのCNN
    """
    def __init__(self,
                 in_channels, # channel
                 in_mode, # DOF
                 out_channels, #channel
                 out_dof, # DOF
                 kernel_size, 
                 stride, 
                 layer,
                 in_features, 
                 out_features, 
                 dt,
                 past, 
                 future,
                 use_integrate=False, 
                 use_masked=False, 
                 use_integrate_in_mask=False):
        super(CNN_for_frame, self).__init__()
        self.layers = nn.ModuleList([nn.ModuleList([nn.Conv1d(in_channels, 
                                               out_dof, 
                                               kernel_size, 
                                               stride,
                                               padding='same')] + 
                                    [Residual_block(out_dof, 
                                               out_dof,
                                               kernel_size,
                                               stride, 
                                               ) for i in range(layer-1)]) for j in range(out_channels)])
        if use_integrate:
            self.integrate_layers = nn.ModuleList([nn.ModuleList([masked_net.IntegrateLayer(in_features, out_features, dt=dt) for i in range(out_channels)])for j in range(out_dof)])
        if use_masked:
            self.masked_layers = nn.ModuleList([nn.ModuleList([masked_net.MNN_layer(in_features, out_features,past, future,use_integrate_in_mask) for i in range(out_channels)])for j in range(out_dof)])
        self.use_integrate=use_integrate
        self.use_masked=use_masked
    
    def forward(self, x):
        """
        x batch, channels, sequence
        """

        for channel in range(len(self.layers)):
            # out_channelだけ繰り返す
            # print('x', x.shape)
            tmp = self.layers[channel][0](x)
            for layer in range(1, len(self.layers[0])):
                # レイヤー数だけ繰り返す
                tmp = self.layers[channel][layer](tmp) # (batch_size, out_DOF, channel)
            # １channel以降はテンソルを結合する
            if channel == 0:
                y = tmp.unsqueeze(dim=1)
            else:
                # print('y', y.shape)
                # print('tmp', tmp.shape)
                y = torch.concatenate([y, tmp.unsqueeze(dim=1)], dim=1) # (batch_size, channel, DOF, sequence)
        # print(y.shape)
        
        if self.use_masked and not(self.use_integrate):
            for dof in range(len(self.masked_layers)):
                for channel in range(len(self.masked_layers[0])):
                    tmp = self.masked_layers[dof][channel](y[:, channel, dof, :])
                    if channel == 0:
                        tmp = tmp.unsqueeze(dim=1) #batch, channle, sequence
                        output_channel = tmp.unsqueeze(dim=2) # batch, channel, dof, sequence
                    else:
                        tmp = tmp.unsqueeze(dim=1) #batch, channle, sequence
                        output_channel = torch.concatenate([output_channel, tmp.unsqueeze(dim=2)], dim=1)  # batch, channel, dof, sequence

                if dof == 0:
                    output = output_channel
                else:
                    output = torch.concatenate([output, output_channel], dim=2)
            return output
        

        if self.use_integrate and not(self.use_masked):
            for dof in range(len(self.integrate_layers)):
                for channel in range(len(self.integrate_layers[0])):
                    tmp = self.integrate_layers[dof][channel](y[:, channel, dof, :])
                    if channel == 0:
                        tmp = tmp.unsqueeze(dim=1) #batch, channle, sequence
                        output_channel = tmp.unsqueeze(dim=2) # batch, channel, dof, sequence
                    else:
                        tmp = tmp.unsqueeze(dim=1) #batch, channle, sequence
                        output_channel = torch.concatenate([output_channel, tmp.unsqueeze(dim=2)], dim=1)  # batch, channel, dof, sequence

                if dof == 0:
                    output = output_channel
                else:
                    output = torch.concatenate([output, output_channel], dim=2)
            return output
        

        # ここから下はなくてもいい(MNN_layerのところでuse_integrateをTrueにすれば動く)
        if self.use_integrate and self.use_masked:
            #maskedを通ったあとにintegrate layer を通す。
            for dof in range(len(self.masked_layers)):
                for channel in range(len(self.masked_layers[0])):
                    tmp = self.masked_layers[dof][channel](y[:, channel, dof, :])
                    if channel == 0:
                        tmp = tmp.unsqueeze(dim=1) #batch, channle, sequence
                        output_channel = tmp.unsqueeze(dim=2) # batch, channel, dof, sequence
                    else:
                        tmp = tmp.unsqueeze(dim=1) #batch, channle, sequence
                        output_channel = torch.concatenate([output_channel, tmp.unsqueeze(dim=2)], dim=1)  # batch, channel, dof, sequence

                if dof == 0:
                    output = output_channel
                else:
                    output = torch.concatenate([output, output_channel], dim=2)
            # print(output.shape)
            for dof in range(len(self.integrate_layers)):
                for channel in range(len(self.integrate_layers[0])):
                    tmp = self.integrate_layers[dof][channel](output[:, channel, dof, :])
                    if channel == 0:
                        tmp = tmp.unsqueeze(dim=1) #batch, channle, sequence
                        output_channel = tmp.unsqueeze(dim=2) # batch, channel, dof, sequence
                    else:
                        tmp = tmp.unsqueeze(dim=1) #batch, channle, sequence
                        output_channel = torch.concatenate([output_channel, tmp.unsqueeze(dim=2)], dim=1)  # batch, channel, dof, sequence

                if dof == 0:
                    output_int = output_channel
                else:
                    output_int = torch.concatenate([output_int, output_channel], dim=2)
            return output_int

        else:
            return y
        # 目標outputは，(batch_size, channel, DOF, sequence)

