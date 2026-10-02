import matplotlib.pyplot as plt
import numpy as np
import torch

# 複数の波を同一のグラフ上に描画して比較する関数------------------------------------------------
def plot_waves(time,
               waves, 
               labels, 
               ymin, 
               ymax, 
               ylabel,  
               figsize=(10, 5), 
               savefig=False, 
               filename=''):
    

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for i in range(len(waves)):
        ax.plot(time, waves[i], label = labels[i])
    ax.set_xlabel('t [$s$]')
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin, ymax)
    ax.legend()
    fig.show()
    if savefig:
        fig.savefig(filename)


# 応答解析と学習結果を同じ波で試す．-----------------------------------------------------------
def test_model(model, waves, idx, mass, omega, f_yield, h, res_type,  factor, r_post=0.00, dt=0.01,):
    """
    model | 試すモデル
    waves | splitされた加速度波形の入った配列
    idx | wavesの何番目の波を試すか
    mass | 応答解析の質量
    omega | 応答解析の固有円振動数
    f_yield | 降伏
    h | 減衰定数
    res_type   | rel_disp等の結果を指定する文字列
    r_post | 降伏した後の傾き?
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Openseespyによる時刻歴応答解析
    res_anslisys = analysis.SDOF_analysis_ops.ops_get_response(mass, 
                                                               k_spring=omega**2*mass, 
                                                               f_yield=f_yield, 
                                                               EQ_accel=np.array(waves[idx], dtype=np.float64), 
                                                               dt=dt, 
                                                               xi=h, 
                                                               r_post=r_post)[res_type]
    
    # 機械学習モデルによる応答の推定
    with torch.no_grad():
        model_output = model.forward(torch.tensor([waves[idx], waves[idx]]).to(device))
    
    model_output = model_output.to('cpu').detach().numpy().copy()[0]/factor

    return res_anslisys, model_output


def restore(output, y_max):
    """
    学習したモデルから出てくるoutputを-1~1から元の単位に復元する関数
    output torch.tensor
    y_max np.array
    """
    if type(output) == torch.tensor:
        output = output.to('cpu').detach().numpy().copy()
    y_max_broadcast = y_max[:, np.newaxis, np.newaxis, :]
    result = output * y_max_broadcast
    return result