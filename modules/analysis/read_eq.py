import numpy as np
import torch
from analysis.SDOF_analysis_ops import *
from analysis.make_train_data import *
def read_EQ(filenum, dt):
    with open(filenum, 'r') as f:
        tmp = f.read().split('\n')
        time = []
        result = []
        cnt = 0
        for row in tmp:
            if ' ' in row:
                k = row.split(' ')
            elif ',' in row:
                k = row.split(',')
            else:
                k = row.split(' ')
            # print(k)print(row)
            if cnt >= 5 and k != ['']:
                time.append(float((cnt-5) * dt))
                result.append(float(k[0]))
                
            cnt += 1
    return time, result

def unit_change(EQ):
    """
    地震動の波形の単位を変換する関数
    """
    EQ = [EQ[i] * 0.01 for i in range(len(EQ))]
    return EQ


def split_EQ(time, EQ, dt, T, many=True):
    """
    地震動を分割する関数
    many | (bool) 地震動の分割をdt刻みでするかT刻みでするか
    """
    if len(time) != len(EQ):
        print('Error : the length of time is different from that of EQ')
    step = T//dt # 何ステップ分切り出すか
    if many:
        result =np.array( [EQ[int(i): int(i + step)] for i in range(int(len(EQ)-step))], dtype=np.float32)
    else:
        result = np.array([EQ[int(i*step): int(i*step + step)] for i in range(int(len(EQ)//step))], dtype=np.float32)
    time_res = time[0:int(0 + step)]
    return time_res, result


def restore(output, y_max):
    """
    学習したモデルから出てくるoutputを-1~1から元の単位に復元する関数
    output torch.tensor
    y_max np.array
    """
    if type(output) == torch.tensor:
        output = output.to('cpu').detach().numpy().copy()
    y_max_broadcast = y_max[np.newaxis,:, :, np.newaxis]
    result = output * y_max_broadcast
    return result


def test_model(model, EQ_split, N, m, k_spring, f_yield,  h, dt, r_post, x_max, y_max):
    # prediction by ML model
    input = torch.Tensor(np.array(EQ_split/x_max, dtype=np.float32)).to('cuda')
    # 今回は、model_output---(batch_size, channels=4, timestep)
    model_output = model(input)
    if len(model_output.shape) == 3:
        model_output = model_output.unsqueeze(dim=2) # (batch_size, channel, Ndof, timestep)
    model_output = restore(model_output, y_max)

    # calculation by openseespy
    response_analysis = get_ops_response_N(N, m, k_spring, f_yield, EQ_split, dt, h, r_post=r_post)
    # (channels, NDOF, timestep)
    return model_output, response_analysis
