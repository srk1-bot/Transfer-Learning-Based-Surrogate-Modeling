import openseespy.opensees as ops
import numpy as np
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
#                       ......................|
#                      J\----------------------
#                      J}
#                      J}
#                      J}
# //////////////////////////////////////////////////////


def ops_get_response(mass, k_spring, f_yield, EQ_accel, dt, xi=0.05, r_post=0.0):
    # print(mass, k_spring, f_yield, dt)
    """
    :Args
        :param mass: SDOF mass [kg]
        :param k_spring: spring_stiffness
        :param f_yield: yield strength
        :param h :damping ratio
        :param EQ_accel: list, acceleration values m/s**2
        :param dt: float, time step of acceleration values
        :param xi: damping ratio
        :param r_post: post-yield stiffness
        :return:
    """
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3) 

    bottom = 1
    top = 2
    ops.node(bottom, 0.0, 0.0)
    ops.node(top, 0.0, 0.0)

    # FIX
    ops.fix(top, opc.free, opc.fixed, opc.fixed)
    ops.fix(bottom, opc.fixed, opc.fixed, opc.fixed)

    # y, Rz方向の変位はtopとbottomで同じなので拘束する
    ops.equalDOF(1, 2, *[2, 3])

    # nodal mass (weight / g):
    ops.mass(top, mass, 0.0, 0.0)

    # Define material
    bilinear_mat_tag = 1
    mat_type = 'Steel01'
    mat_props = [f_yield, k_spring, r_post]
    # print('ops_get_response', f_yield)
    ops.uniaxialMaterial(mat_type, bilinear_mat_tag, *mat_props)

    # zero length elementを指定する
    beam_tag = 1
    ops.element('zeroLength', beam_tag, bottom, top, '-mat', bilinear_mat_tag, '-dir', 1, '-doRayleigh', 1)


    # 応答解析の定義
    load_tag_dynamic = 1
    pattern_tag_dynamic = 1

    #外力の形で与える
    values = np.array(EQ_accel)

    # print(values)
    ops.timeSeries('Path', load_tag_dynamic, '-dt', dt, '-values', *values)
    ops.pattern('UniformExcitation', pattern_tag_dynamic, opc.X, '-accel', load_tag_dynamic)
    

    # 減衰の設定
    eigen_1 = ops.eigen('-fullGenLapack', 1)
    angular_freq = eigen_1[0] ** 0.5 #固有値の0.5乗, omega
    alpha_m = 0.0
    beta_k = 2 * xi / angular_freq
    beta_k_comm = 0.0
    beta_k_init = 0.0

    ops.rayleigh(alpha_m, beta_k, beta_k_init, beta_k_comm)

    # 応答解析の実行
    ops.wipeAnalysis()

    ops.algorithm('Newton')
    ops.system('SparseGeneral')
    ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')
    
    tol = 1.0e-5
    # tol = 1.0e-10
    iterations = 10
    ops.test('EnergyIncr', tol, iterations, 0, 2)
    analysis_time = (len(values)) * dt #ここ変えた8/29 (len(values)) -> (len(values)-1)
    analysis_dt = 0.01
    outputs = {
        "time": [],
        "rel_disp": [],
        "rel_accel": [],
        "rel_vel": [],
        "force": [], 
        "damping": []
    }
    

    while ops.getTime() <= analysis_time:
        curr_time = ops.getTime()
        ops.analyze(1, analysis_dt)
        outputs['time'].append(curr_time)
        outputs["rel_disp"].append(ops.nodeDisp(top, 1))
        outputs["rel_vel"].append(ops.nodeVel(top, 1))
        outputs["rel_accel"].append(ops.nodeAccel(top, 1))
        ops.reactions('-dynamic')
        outputs["force"].append(-ops.eleForce(bottom, 1))  # Negative since diff node
        ops.reactions('-rayleigh')
        outputs['damping'].append(ops.nodeReaction(top, 1))

        
    ops.wipe()
    for item in outputs:
        outputs[item] = np.array(outputs[item])
    
    return outputs



# # test_zone-------------------------^-^-------------------------------

# m = 100 # kg
# T = 0.5 # 非減衰固有周期s
# omega = 2*np.pi/T
# h = 0.1 # 減衰定数
# dt = 0.01 # 積分時間刻みs
# batch_size = 1024
# batch_size_long = 512
# f_yield = m*9.8*0.2
#  # 降伏強度kn
# r_post = 0.1

# mass = m / 9.8 # N
# k_spring = omega**2 * mass
# f_yield =  f_yield
# xi = h
# r_post = r_post
# c=2*h/omega*k_spring
# idx = 10
# gm = np.array([1 for i in range(3920)])
# output = ops_get_response(mass, k_spring, f_yield, EQ_accel = np.array(gm, dtype=np.float64), dt = dt, xi=xi, r_post=r_post)