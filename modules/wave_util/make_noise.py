import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate
import torch
from network import masked_net


# white_noise---------------------------
def white_noise(T, dt, output_dt = 0.01, mean=0.0, sd = 1.0):
    """
    T: float: length of the noise
    dt: float: time step of the white noise
    output_dt: time step of the output noise
    mean: float: mean
    sd: float: standard deviation
    """
    result = []
    tmp = np.random.normal(mean, sd, int(T/dt) + 1)
    # print('tmp', len(tmp))
    interpolate_length = int(dt/output_dt)
    for i in range(len(tmp)-1):
        # 元のホワイトノイズに対して補間
        interpolate = calc_interpolate_point(tmp[i], tmp[i+1], interpolate_length)
        for j in interpolate:
            result.append(j)
    result.append(tmp[-1]) # 最後の1行を加える
    # print(int((T)/output_dt))
    
    time = [int(i)*output_dt for i in range(len(result))] # 時間
    time = np.array(time, dtype=np.float32)
    result = np.array(result, dtype=np.float32)
    return time[:int((T)/output_dt)], result[:int((T)/output_dt)]




def calc_interpolate_point(x1, x2, interpolate_length):
    # x1 手前の点
    # x2 奥の点
    # interpolate_length 補間する点の数
    return [int(i)*(x2-x1)/interpolate_length + x1 for i in range(interpolate_length)]

def make_whitenosees(num_of_waves, T, dt, output_dt = 0.01, mean=0.0, sd = 1.0):
    noises = []
    for i in range(num_of_waves):
        time_short, tmp = white_noise(T, dt, output_dt, mean, sd)
        noises.append(tmp)
    return time_short, noises


def spline_whitenoise(T, dt, output_dt=0.01, mean=0.0, sd = 1.0):
    """
    T: float: length of the noise
    dt: float: time step of the white noise
    output_dt: time step of the output noise
    mean: float: mean
    sd: float: standard deviation
    """
    result = []
    tmp = np.random.normal(mean, sd, int(T/dt) + 1)
    tmp_time = np.array([int(i) * dt for i in range(int(T/dt) + 1)])
    f = scipy.interpolate.Akima1DInterpolator(tmp_time, tmp)
    
    result = np.array([f(int(i)*output_dt) for i in range(int((T)/output_dt))], dtype=np.float32)
    time = np.array([int(i)*output_dt for i in range(int((T)/output_dt))], dtype=np.float32)
    return time, result


def make_spline_whitenosees(num_of_waves, T, dt, output_dt = 0.01, mean=0.0, sd = 1.0):
    noises = []
    for i in range(num_of_waves):
        time_short, tmp = spline_whitenoise(T, dt, output_dt, mean, sd)
        noises.append(tmp)
    return time_short, noises



def calc_vel(waves, dt = 0.01):
    # 加速度波形を速度波形にする関数
    # waves (num_of_waves, sequence)
    integrator =dt *  np.tril(np.ones(waves.shape[1]))

    result = []
    for i in range(len(waves)):
        result.append(integrator@waves[i])
    return np.array(result)


def calc_vel_torch(waves, dt=0.01):
    # 加速度波形を速度波形に直す関数(GPU使用)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)
    waves = np.array(waves, dtype=np.float32)
    waves = torch.tensor(waves, requires_grad=False, device=device) # (num_of_waves, dt)
    length = waves.shape[0]
    sequence = waves.shape[1]

    IntegrateLayer = masked_net.IntegrateLayer(sequence, sequence, dt=dt)
    IntegrateLayer.to(device)
    result = IntegrateLayer(waves)
    result = result.to('cpu').detach().numpy().copy()
    return result
    
    


    # integrator = dt * torch.tril(torch.ones(sequence, sequence))

# test----------------------------------
# time, result = spline_whitenoise(0.28, 0.04)

# print(len(result))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(time, result)
# fig.savefig('hogehoge.png')


