import openseespy.opensees as ops
import numpy as np
from dyneq.run_dyneq import*
### Generating Constants ###
class opensees_constants:
    def __init__(self):
        self.free = 0
        self.fixed = 1

        self.FREE = 0
        self.FIXED = 1

        self.X = 1
        self.Y = 2
        self.ROTZ = 3

opc = opensees_constants()

# ------------------------------------------------------------
# N質点系の応答解析
# ------------------------------------------------------------


def get_ops_response_N(N, mass, k_spring, f_yield, EQ_accel, dt, xi=0.05, r_post=0.0, do_amplification=False, short = False):
    """
    ## get_ops_response_N
    get response of N-DOF system
    #### Args
        :N: [-] number of degree of freedom (int)
        :mass: [kg] **mass** of N-DOF mass points list, not weight(float)
        :k_spring: [N/m] list of spring constnts of N-springs (list(float))
        :f_yield: [N] yield strength of N-springs (list(float)) 
        :EQ_accel: [m/s^2] absolute ground acceleration (list(float))
        :dt: [s] time step for output (float)
        :xi: [-] damping ratio for the entire system $ (float)
        :r_post: [-] (young's coefficient after yield) / (young's coefficient before yield)(float)
        NOTE in this program, the damping ratio of each springs is the same.
    """

    # initialization for openseespy
    ops.wipe()
    # define model space
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # 一番下のノードの定義
    bottom = 1
    ops.node(bottom, 0.0, 0.0)
    ops.fix(bottom, opc.fixed, opc.fixed, opc.fixed)

    # 質点のある位置のノードの定義
    for n in range(N):
        ops.node(n+2, 0.0, 0.0)
        ops.fix(n+2, opc.free, opc.fixed, opc.fixed) #固定の定義 x, y, Rz
    
    # y方向，Rz方向の変位はすべてのノードで同じなので拘束する(必要ないと思われる)
    for n in range(N):
        ops.equalDOF(n+1, n+2, *[2, 3])

    for n in range(N):
        ops.mass(n+2, mass[n], 0.0, 0.0)
    
    # Define material
    for n in range(N):
        bilinear_mat_tag = n
        mat_type = 'Steel01'
        mat_props = [f_yield[n], k_spring[n], r_post[n]]
        ops.uniaxialMaterial(mat_type, bilinear_mat_tag, *mat_props)
    
    # Define zero_length elements
    for n in range(N):
        beam_tag = n
        ops.element('zeroLength', 
                    beam_tag, 
                    n+1, 
                    n+2, 
                    '-mat', 
                    n, 
                    '-dir', 
                    1, 
                    '-doRayleigh', 
                    1)
    
    # 応答解析の定義
    load_tag_dynamic = 1
    pattern_tag_dynamic = 1

    if do_amplification:
        # 地盤増幅解析
        if short:
            accelerogram = amplification(EQ_accel, 'test_wave.txt', '/home/ishikawa/ドキュメント/m1/modules/dyneq/dyneq-j', 'dyneq401.out', 'DATA_short.dat', '/home/ishikawa/ドキュメント/m1/modules/dyneq/dyneq-j/out')
        else:
            accelerogram = amplification(EQ_accel, 'test_wave.txt', '/home/ishikawa/ドキュメント/m1/modules/dyneq/dyneq-j', 'dyneq401.out', 'DATA.dat', '/home/ishikawa/ドキュメント/m1/modules/dyneq/dyneq-j/out')
    else:
        accelerogram = EQ_accel  # Loads accelerogram file
    
    values = accelerogram
    ops.timeSeries('Path', load_tag_dynamic, '-dt', dt, '-values', *values)
    ops.pattern('UniformExcitation', 
                pattern_tag_dynamic, 
                opc.X, 
                '-accel', 
                load_tag_dynamic)
    
    # 減衰の設定
    eigen_1 = ops.eigen('-fullGenLapack', 1)
    angular_freq = eigen_1[0]**0.5
    # print('T_1', np.pi*2/angular_freq)
    # print('T_2',np.pi*2/(eigen_1[1]**0.5))
    # print('T_3',np.pi*2/(eigen_1[2]**0.5))
    alpha_m = 0.0
    beta_k = 0.0
    beta_k_comm = 0.0
    beta_k_init = 2*xi/ angular_freq

    ops.rayleigh(alpha_m, beta_k, beta_k_init, beta_k_comm)

    ops.wipeAnalysis()

    ops.algorithm('Newton')
    ops.system('SparseGeneral')
    ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    tol = 1.0e-3
    iterations = 10
    ops.test('EnergyIncr', tol, iterations, 0, 2)
    analysis_time = (len(values)) * dt
    analysis_dt = 0.01
    outputs = {
        "time": [],
        "rel_disp": [[] for i in range(N)],
        "rel_accel": [[] for i in range(N)],
        "rel_vel": [[] for i in range(N)],
        "force": [[] for i in range(N)], 
        'eleforce': [[] for i in range(N)],
        "damping": [[] for i in range(N)]
    }

    while ops.getTime() <= analysis_time:
        curr_time = ops.getTime()
        ops.analyze(1, analysis_dt)
        outputs['time'].append(curr_time)
        for n in range(N):
            outputs["rel_disp"][n].append(ops.nodeDisp(n+2, 1))
            outputs["rel_vel"][n].append(ops.nodeVel(n+2, 1))
            outputs["rel_accel"][n].append(ops.nodeAccel(n+2, 1))
            # 反力
            ops.reactions('-dynamic')
            outputs["force"][n].append(-ops.eleForce(n, 1)) # n番目のばねの反力
            # 減衰力
            ops.reactions('-rayleigh')
            outputs["damping"][n].append(-ops.nodeReaction(n+1, 1)) # n+1番目の節点の減衰力 -\dot{x_n}c_{n-1} + (\dot{x}_{n+1}-\dot{x}_n)

    for item in outputs:
        outputs[item] = np.array(outputs[item])

    return outputs





def get_ops_eigen(N, mass, k_spring, f_yield, EQ_accel, dt, xi=0.05, r_post=0.0):
    """
    ## get_ops_response_N
    非減衰固有周期
    #### Args
        :N: [-] number of degree of freedom (int)
        :mass: [kg] **mass** of N-DOF mass points list, not weight(float)
        :k_spring: [N/m] list of spring constnts of N-springs (list(float))
        :f_yield: [N] yield strength of N-springs (list(float)) 
        :EQ_accel: [m/s^2] absolute ground acceleration (list(float))
        :dt: [s] time step for output (float)
        :xi: [-] damping ratio for the entire system $ (float)
        :r_post: [-] (young's coefficient after yield) / (young's coefficient before yield)(float)
        NOTE in this program, the damping ratio of each springs is the same.
    """

    # initialization for openseespy
    ops.wipe()
    # define model space
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # 一番下のノードの定義
    bottom = 1
    ops.node(bottom, 0.0, 0.0)
    ops.fix(bottom, opc.fixed, opc.fixed, opc.fixed)

    # 質点のある位置のノードの定義
    for n in range(N):
        ops.node(n+2, 0.0, 0.0)
        ops.fix(n+2, opc.free, opc.fixed, opc.fixed) #固定の定義 x, y, Rz
    
    # y方向，Rz方向の変位はすべてのノードで同じなので拘束する(必要ないと思われる)
    for n in range(N):
        ops.equalDOF(n+1, n+2, *[2, 3])

    for n in range(N):
        ops.mass(n+2, mass[n], 0.0, 0.0)
    
    # Define material
    for n in range(N):
        bilinear_mat_tag = n
        mat_type = 'Steel01'
        mat_props = [f_yield[n], k_spring[n], r_post[n]]
        ops.uniaxialMaterial(mat_type, bilinear_mat_tag, *mat_props)
    
    # Define zero_length elements
    for n in range(N):
        beam_tag = n
        ops.element('zeroLength', 
                    beam_tag, 
                    n+1, 
                    n+2, 
                    '-mat', 
                    n, 
                    '-dir', 
                    1, 
                    '-doRayleigh', 
                    1)
    
    # 応答解析の定義
    load_tag_dynamic = 1
    pattern_tag_dynamic = 1

    values = EQ_accel
    ops.timeSeries('Path', load_tag_dynamic, '-dt', dt, '-values', *values)
    ops.pattern('UniformExcitation', 
                pattern_tag_dynamic, 
                opc.X, 
                '-accel', 
                load_tag_dynamic)
    
    # 減衰の設定
    eigen_1 = ops.eigen('-fullGenLapack', N)
    angular_freq = eigen_1[0]**0.5
    # print('T_1', np.pi*2/angular_freq)
    # print('T_2',np.pi*2/(eigen_1[1]**0.5))
    return np.array(eigen_1)




# -------------------^-^--------------------------------
#                   ...Ja...
#               .-H9C>>>>>+OTHJ.
#             .d5>>>>>>>>>>>>>+Tm.
#            (8>>>>>>;>;>>>>>>>>+W
#           (5>>>>;>>>>>;>;>>>>>>>d,
#          .#>>;>>>>;>>>>>>;>;>>>>+N  m
#          .P>>>>;>>>;>>;>>>>>;>>>>M.
#          .b>>>>>>>>>>>>>>>>>>;>>>M`
#           N+>>;>>;>>;>>;>;>>>>;>jF
#           ,N+>>>>>;>>>>>>>;>>>>j#`
#            ,Mx>>>>>>>;>>;>>>;>u@
#              ?Hg+>>;>>>>>>>jgY'
#                 7YWgg&&ggV"^
#                      |
#                      |
#                      |
#                      |
#                      |
#   -------------------|     k
#   |              /\    /\    /\ 
#   |         --  /  \  /  \  /  \--|
#   |         | \/    \/    \/      |
#   ----------|                     |----------
#             |    -------|         |         |
#             |      |    |         |         |
#             -------|    |---------|         |
#                    |    |                   |
#                  -------|    h              |
#                                             |
#                      -----------------------」
#                      |
#                      |
#                      |
#                   ...Ja...
#               .-H9C>>>>>+OTHJ.
#             .d5>>>>>>>>>>>>>+Tm.
#            (8>>>>>>;>;>>>>>>>>+W
#           (5>>>>;>>>>>;>;>>>>>>>d,
#          .#>>;>>>>;>>>>>>;>;>>>>+N  m
#          .P>>>>;>>>;>>;>>>>>;>>>>M.
#          .b>>>>>>>>>>>>>>>>>>;>>>M`
#           N+>>;>>;>>;>>;>;>>>>;>jF
#           ,N+>>>>>;>>>>>>>;>>>>j#`
#            ,Mx>>>>>>>;>>;>>>;>u@
#              ?Hg+>>;>>>>>>>jgY'
#                 7YWgg&&ggV"^
#                      |
#                      |
#                      |
#                      |
#                      |
#   -------------------|     k
#   |              /\    /\    /\ 
#   |         --  /  \  /  \  /  \--|
#   |         | \/    \/    \/      |
#   ----------|                     |----------
#             |    -------|         |         |
#             |      |    |         |         |
#             -------|    |---------|         |
#                    |    |                   |
#                  -------|    h              |
#                                             |
#                      -----------------------」
#                      |
#                      |
#                      |
#                   ...Ja...
#               .-H9C>>>>>+OTHJ.
#             .d5>>>>>>>>>>>>>+Tm.
#            (8>>>>>>;>;>>>>>>>>+W
#           (5>>>>;>>>>>;>;>>>>>>>d,
#          .#>>;>>>>;>>>>>>;>;>>>>+N  m
#          .P>>>>;>>>;>>;>>>>>;>>>>M.
#          .b>>>>>>>>>>>>>>>>>>;>>>M`
#           N+>>;>>;>>;>>;>;>>>>;>jF
#           ,N+>>>>>;>>>>>>>;>>>>j#`
#            ,Mx>>>>>>>;>>;>>>;>u@
#              ?Hg+>>;>>>>>>>jgY'
#                 7YWgg&&ggV"^
#                      |
#                      |
#                      |
#                      |
#                      |
#   -------------------|     k
#   |              /\    /\    /\ 
#   |         --  /  \  /  \  /  \--|
#   |         | \/    \/    \/      |
#   ----------|                     |----------
#             |    -------|         |         |
#             |      |    |         |         |
#             -------|    |---------|         |
#                    |    |                   |
#                  -------|    h              |
#                                             |
#                      -----------------------」
#                      |
#                      |
#                      |
# //////////////////////////////////////////////////////