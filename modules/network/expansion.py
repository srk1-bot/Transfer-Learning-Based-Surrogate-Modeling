import torch
import torch.nn as nn
import numpy as np
from network.naname_net import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 重み行列の拡張
def w_expansion(fc_w, expand, right_up = False):
    """
    ----------------
    fc_w | torch.Tensor/ndarray, 拡張する元の行列
    expand | 何回分拡張するか指定する
    right_up | bool 右上の行列をとってくるか途中の値を使うか，，
    """
    # もし重み行列がテンソルできたのならndarrayに変換する
    if type(fc_w) == 'torch.Tensor':
        fc_w = (fc_w.to('cpu').detach().numpy().copy())
    
    zero_mat = np.zeros((fc_w.shape[0], fc_w.shape[1]), dtype=np.float32)
    if right_up:
        from_back = np.split(fc_w, 2, axis=0)
        from_back = np.split(from_back[1], 2, axis=1)
        from_back = np.array(from_back, dtype=np.float32)[0]
        zero_mat_back = np.zeros((fc_w.shape[0]//2, fc_w.shape[1]//2), dtype=np.float32)

        back_up = np.concatenate([zero_mat_back, zero_mat_back], axis=0)
        back_down = np.concatenate([from_back, zero_mat_back], axis=0)
        from_back = np.concatenate([back_up, back_down], axis=1)
    else:
        from_back = zero_mat


    toconcat = [] # 最後に行方向に結合するためのndarrayを入れておく配列
    for i in range(expand):
        tmp = np.concatenate([return_mat(fc_w, from_back, zero_mat, i, j) for j in range(expand)], axis=0)
        toconcat.append(tmp)
    result = np.concatenate(toconcat, axis=1)
    result = torch.from_numpy(result)
    return result


# --------------------------
def return_mat(A, B, C, i, j):
    # i == jならA, 層でないならBを返す
    if i == j:
        return A
    elif i == j - 1:
        return B
    else:
        return C


# -------------------------
def b_expansion(fc_b, expand):
    # もし重み行列がテンソルできたのならndarrayに変換する
    if type(fc_b) == 'torch.Tensor':
        fc_b = (fc_b.to('cpu').detach().numpy().copy())
    return torch.from_numpy(np.concatenate([fc_b for i in range(expand)], axis=0))

# -------------------------
# modelからweightとbiasを取り出す
def weight_bias(model, num_layers=4, layer_name = 'layers.', weight_name = '.cl.linear', num_start=0):
    state_dict = model.state_dict()
    weight = []
    bias = []
    for i in range(num_start, num_layers):
        w_tmp = state_dict[layer_name + str(i) + weight_name + '.weight'].to('cpu').detach().numpy().copy()
        bias_tmp = state_dict[layer_name + str(i) + weight_name + '.bias'].to('cpu').detach().numpy().copy()
        weight.append(w_tmp)
        bias.append(bias_tmp)
    
    return weight, bias



# weightとbiasを取得，拡張して別のモデルに適応する関数-------------------

def exp_apply(model_2, model_1, exp_factor, num_layers=4):
    """
    -----------------
    model_1のweight, biasを取得，拡張してmodel_2に適用する関数
    Args:
        :exp_factor: int model_2がmodel_1の何倍の入力次元数を持つか
        :num_layers: int model_2とmodel_1の層数
    return:
        : model_2の重みにmodel_1の学習結果が適用される
    """
    # model_1の重みとバイアスを取得-------------------
    weight_1, bias_1 = weight_bias(model_1, num_layers)

    # レイヤの情報を入手
    linear_layers = [module for module in model_2.modules() if isinstance(module, nn.Linear)]

    # カスタム初期化関数の定義------------------------
    def custom_weight_init(m):
        if isinstance(m, nn.Linear):
            cnt = 0
            for layer in linear_layers:
                if m == layer:
                    # print(m, 'fc1')
                    # print(cnt)
                    m.weight = nn.Parameter(w_expansion(weight_1[cnt], exp_factor))
                    # m.weight = nn.Parameter(torch.tensor([[0.1] * output_dim] * input_dim, dtype=torch.float32))
                    m.bias = nn.Parameter(b_expansion(bias_1[cnt], exp_factor))
                cnt += 1


    # カスタム初期化の適用
    model_2.apply(custom_weight_init)
    return None



# weightとbiasを取得，拡張して別のモデルに適応する関数 減衰固有周期学習バージョン-------------------

def exp_apply_th(model_2, model_1, exp_factor, th_in_layer, num_layers=4):
    """
    -----------------
    model_1のweight, biasを取得，拡張してmodel_2に適用する関数
    Args:
        :exp_factor: int model_2がmodel_1の何倍の入力次元数を持つか
        :num_layers: int model_2とmodel_1の層数
    return:
        : model_2の重みにmodel_1の学習結果が適用される
    """
    # model_1の重みとバイアスを取得-------------------
    weight_1, bias_1 = weight_bias(model_1, num_layers)
    # print(len(weight_1))

    # レイヤの情報を入手
    linear_layers = [module for module in model_2.modules() if isinstance(module, nn.Linear)]
    # print(len(linear_layers))


    # カスタム初期化関数の定義------------------------
    def custom_weight_init(m):
        if isinstance(m, nn.Linear):
            cnt = 0
            # for layer in linear_layers[:-2]:
            for layer in linear_layers[:-2*th_in_layer]:
                if m == layer:
                    # print(cnt)
                    # print(m, 'fc1')
                    m.weight = nn.Parameter(w_expansion(weight_1[cnt], exp_factor))
                    # m.weight = nn.Parameter(torch.tensor([[0.1] * output_dim] * input_dim, dtype=torch.float32))
                    m.bias = nn.Parameter(b_expansion(bias_1[cnt], exp_factor))
                cnt += 1


    # カスタム初期化の適用
    model_2.apply(custom_weight_init)
    return None


# weightとbiasを取得，拡張して別のモデルに適応する関数最後積分情報入れるやつ-------------------

def exp_apply_3(model_2, model_1, exp_factor, num_layers):
    """
    -----------------
    model_1のweight, biasを取得，拡張してmodel_2に適用する関数
    Args:
        :exp_factor: int model_2がmodel_1の何倍の入力次元数を持つか
        :num_layers: int model_2とmodel_1の層数
    return:
        : model_2の重みにmodel_1の学習結果が適用される
    """
    # model_1の重みとバイアスを取得-------------------
    weight_1, bias_1 = weight_bias(model_1, num_layers)
    # print(len(weight_1))

    # レイヤの情報を入手
    linear_layers = [module for module in model_2.modules() if isinstance(module, nn.Linear)]
    # print(len(linear_layers))


    # カスタム初期化関数の定義------------------------
    def custom_weight_init(m):
        if isinstance(m, nn.Linear):
            cnt = 0
            for layer in linear_layers[:-3]:
            # for layer in linear_layers[:-2*num_layers]:
                if m == layer:
                    # print(cnt)
                    # print(m, 'fc1')
                    m.weight = nn.Parameter(w_expansion(weight_1[cnt], exp_factor))
                    # m.weight = nn.Parameter(torch.tensor([[0.1] * output_dim] * input_dim, dtype=torch.float32))
                    m.bias = nn.Parameter(b_expansion(bias_1[cnt], exp_factor))
                cnt += 1


    # カスタム初期化の適用
    model_2.apply(custom_weight_init)
    return None



# weightとbiasを取得，別チャンネルに分けて最後積分して応答予測するやつ-------------------
# -------------------------
# modelからweightとbiasを取り出す

def exp_apply_PINN_channel(model_2, model_1, exp_factor, num_layers, branch):
    """
    -----------------
    model_1のweight, biasを取得，拡張してmodel_2に適用する関数
    Args:
        :exp_factor: int model_2がmodel_1の何倍の入力次元数を持つか
        :num_layers: int model_2とmodel_1の層数
    return:
        : model_2の重みにmodel_1の学習結果が適用される
    """

    # print(len(weight_1))
    linear_layers_model1 = [module for module in model_1.modules() if isinstance(module, nn.Linear)]

    # レイヤの情報を入手
    linear_layers_model2 = [module for module in model_2.modules() if isinstance(module, nn.Linear)]
    


    # # カスタム初期化関数の定義------------------------
    # def custom_weight_init(m):
    #     # if isinstance(m, nn.Linear):
    cnt = 0
    # 初期レイヤーの
    for layer in linear_layers_model2[:]:
        layer.weight = nn.Parameter(w_expansion(linear_layers_model1[cnt].weight.to('cpu').detach(), exp_factor), requires_grad=True)
        layer.bias = nn.Parameter(b_expansion(linear_layers_model1[cnt].bias.to('cpu').detach(), exp_factor), requires_grad=True)
        cnt += 1
        
        # print(layer.weight[0])
        # print(w_expansion(linear_layers_model1[cnt].weight.to('cpu').detach(), exp_factor)[0])
        # m.weight = nn.Parameter(torch.tensor([[0.1] * output_dim] * input_dim, dtype=torch.float32))


    # カスタム初期化の適用
    # model_2.apply(custom_weight_init)
    return None




# modelのweightをプロットする関数
def plot_weight(fc1_w):
    x = []
    y = []
    z = []
    for i in range(len(fc1_w)):
        for j in range(len(fc1_w[0])):
            z.append(fc1_w[i][j])
            x.append(i)
            y.append(j)
    ax = plt.figure().add_subplot(111)
    sc = ax.scatter(x, y, c=z, cmap=cm.seismic, vmin=-1, vmax=1, s=2)
    plt.colorbar(sc)
    ax.invert_yaxis()
    # print(fc1_w)
    plt.show()

