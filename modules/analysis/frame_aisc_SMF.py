import pandas as pd
import openseespy.opensees as ops
import numpy as np
import time
import torch.utils
import torch
from torch.utils.data import DataLoader, TensorDataset


# units
inc = 25.4/1000
m = 1.0 # Meters
N = 1.0 # KiloNewtons
sec = 1.0 # seconds
kg = 1.0 # kilogram

mm = 0.001*m # Milimeters 1mm = 0.001m
cm = 0.01*m # centimeters
ton = 0.001*kg
KN = 1000*N # 1 newton = 0.001 kilo newton
Pa = N/m**2
MPa = 10**(6)*Pa

# yard pond
ksi = 6.89*MPa # 
ft = 0.3048*m # feet
lb = 0.45359237*kg # pound
psf = lb/(ft**2) # pound per square feet


def judge_experiment(h, tw, bf, tf, Lb, ry, d, L, Fy, c_unit_1, c_unit_2, RBS=False):
    judge = False

    if RBS:
        if 20<=h/tw<=55 and 20<=Lb/ry<=80 and 4<=bf/(2*tf)<=8 and 2.5<=L/d<=7 and 4*inc<=d<=36*inc and 35*ksi/MPa<=Fy<=65*ksi/MPa:
            judge = True
    else:
        if 21<=h/tw<=55 and 20<=Lb/ry<=65 and 4.5<=bf/(2*tf)<=7.5 and 2.3<=L/d<=6.3 and 18*inc<=d<=36*inc and 35*ksi/MPa<=Fy<=63*ksi/MPa:
            judge = True
    return judge



def theta_p(h, tw, bf, tf, Lb, ry, d, L, Fy, c_unit_1, c_unit_2, RBS=False, ignore_exp=True):
    '''
    - h: Clear distance between fillets (Distance between web fillets)
    - tw: Web thickness
    - bf: Flange width
    - Lb: Unbraced length / Unbraced span length
    - ry: Radius of gyration about the weak axis (Radius of gyration, y-axis)
    - d: Depth of beam / Structural depth
    - Fy: Specified yield strength / Yield strength
    '''
    # 梁のサイズが実験データを内挿するものとなっているかチェック
    if judge_experiment(h, tw, bf, tf, Lb, ry, d, L, Fy, c_unit_1, c_unit_2, RBS) == False:
        print('size of beam is out of experimental data for calculating theta_p, theta_pc and Lamda')
        if not ignore_exp:
            return None
    
    # calculate theta_p
    if not RBS:
        # with non-RBS connections
        result = 0.087*(h/tw)**(-0.365) * (bf/2/tf)**(-0.14) * (L/d)**0.34 * (d/c_unit_1/(21*inc))**(-0.721) * (c_unit_2*Fy/50)**(-0.23)
    else:
        # with RBS connections
        result = 0.19*(h/tw)**(-0.314) * (bf/2/tf)**(-0.10) * (Lb/ry)**(-0.1185) * (L/d)**0.113 * (d/c_unit_1/(21*inc))**(-0.76) * (c_unit_2*Fy/50)**(-0.07)
    return result


def theta_pc(h, tw, bf, tf, Lb, ry, d, L, Fy, c_unit_1, c_unit_2, RBS=False, ignore_exp=True):
    if not RBS:
        # with non-RBS connections
        result = 5.70*(h/tw)**(-0.565) * (bf/2/tf)**(-0.80) * (d/c_unit_1/(21*inc))**(-0.28) * (c_unit_2*Fy/50)**(-0.43)
    else:
        # with RBS connections
        result = 9.62*(h/tw)**(-0.513) * (bf/2/tf)**(-0.863) * (Lb/ry)**(-0.108) * (c_unit_2*Fy/50)**(-0.36)
    return result



def Lamda(h, tw, bf, tf, Lb, ry, d, L, Fy, c_unit_1, c_unit_2, RBS=False, ignore_exp=True):
    if RBS == False:
        # with non-RBS connections
        result = 500*(h/tw)**(-1.34) * (bf/2/tf)**(-0.595) * (c_unit_2*Fy/50)**(-0.36)
    else:
        # with RBS connections
        result = 592*(h/tw)**(-1.138) * (bf/2/tf)**(-0.632) * (Lb/ry)**(-0.205) * (c_unit_2*Fy/50)**(-0.391)
    return result


def Mp(Fy, d, bf, tf, tw):
    Frange = Fy * bf*tf*(d-tf) # N・m
    Web = Fy * (d-2*tf)/2 * tw * (d-2*tf)/2 # N・m
    return Frange + Web

def construct_bilin_material(mat_tag, h, tw, bf, tf, Lb, ry, d, L, Fy, c_unit_1=0.0254, c_unit_2=0.145 , RBS=False, ignore_exp=True):
    '''
    output parameters for bilin materials of IMK model
    '''
    
    if RBS:
        My = 1.1 * Mp(Fy, d, bf*0.5, tf, tw) # N・m 
        I_x = (bf*0.5*d**3-(bf*0.5-tw)*(d-2*tf)**3)/12 # m^4
    else:
        My = 1.1 * Mp(Fy, d, bf, tf, tw) # N・m 
        I_x = (bf*d**3-(bf-tw)*(d-2*tf)**3)/12 # m^4
    
    theta_u = 0.4 # rad
    
    E = 200000*N/mm**2 # N/m^2
    # print(E)


    Fy_MPa=  Fy/MPa # MPa
    # unit : [m], [Mpa]
    thetap = theta_p(h, tw, bf, tf, Lb, ry, d, L, Fy_MPa, c_unit_1, c_unit_2, RBS, ignore_exp)
    thetapc = theta_pc(h, tw, bf, tf, Lb, ry, d, L, Fy_MPa, c_unit_1, c_unit_2, RBS, ignore_exp)
    Lam = Lamda(h, tw, bf, tf, Lb, ry, d, L, Fy_MPa, c_unit_1, c_unit_2, RBS, ignore_exp)

    result = {'ID':'Bilin',
                 'matTag':mat_tag,
                 'K0':E*I_x,
                 'as_Plus':0.0020,
                 'as_Neg':0.0020,
                 'My_Plus':My,
                 'My_Neg':-My,
                 'Lamda_S':Lam,
                 'Lamda_C':Lam,
                 'Lamda_A':0,
                 'Lamda_K':Lam,
                 'c_S':1, 'c_C':1,
                 'c_A':1, 'c_K':1,
                 'theta_p_Plus':thetap,
                 'theta_p_Neg': thetap,
                 'theta_pc_Plus':thetapc, 
                 'theta_pc_Neg':thetapc,
                 'Res_Pos': 0.4,
                 'Res_Neg': 0.4,
                 'theta_u_Plus': theta_u,
                 'theta_u_Neg': theta_u,
                 'D_Plus':1,
                 'D_Neg':1 }
    

    
    return result


def calc_ry(I, A):
    '''
    adius of gyration
    '''
    return np.sqrt(I / A)


def create_bilin(df, beam_code, Length, mattag, RBS=True):
    Fy = 55*ksi # N/m**2
    # create a bilin material with beam code and Length of it, attaching the mattag number to it
    
    index = np.array(df[df['AISC_Manual_Label'].str.contains(beam_code, na=False, regex=False)].index)[0]
    d = float(df.loc[index, 'd'])*inc # m
    tw =float(df.loc[index, 'tw'])*inc # m
    bf =float(df.loc[index, 'bf'])*inc # m
    tf =float(df.loc[index, 'tf'])*inc # m
    A = float(df.loc[index, 'A'])*inc**2 # m^2
    I1 = float(df.loc[index, 'Ix'])*inc**4 # m^4
    I2 = float(df.loc[index, 'Iy'])*inc**4 # m^4
    ry = float(df.loc[index, 'ry'])*inc # m
    h = d-2*tf # 確認必要
    result = construct_bilin_material(mattag, h, tw, bf, tf, Length/2, ry, d, Length/2, Fy, RBS=False)
    return result


def set_section(csv_pass, sec_names):
    '''
    create a section dictionary
    '''
    global inc
    sections = dict()
    df = pd.read_csv(csv_pass)
    for sec_name in sec_names:
        # print(sec_name)
        # print(np.array(df[df['AISC_Manual_Label'].str.contains(sec_name, na=False, regex=False)].index))
        index = np.array(df[df['AISC_Manual_Label'].str.contains(sec_name, na=False, regex=False)].index)[0]
        d = float(df.loc[index, 'd'])
        tw =float(df.loc[index, 'tw'])
        bf =float(df.loc[index, 'bf'])
        tf =float(df.loc[index, 'tf'])
        A = float(df.loc[index, 'A'])
        I1 = float(df.loc[index, 'Ix'])
        I2 = float(df.loc[index, 'Iy'])
        sections[sec_name] = {'d': d*inc, 
                              'tw': tw*inc,
                              'bf': bf*inc, 
                              'tf': tf*inc, 
                              'A': (A*inc**2), 
                              'I1': I1*(inc**4), 
                              'I2': I2*(inc**4)}
        # print(sections[sec_name])
    return sections

def calc_key(f, x):
    if f < 10:
        key_tmp = '0' + str(f)
    else:
        key_tmp = str(f)
    if x < 10:
        key_tmp += '0' + str(x)
    else:
        key_tmp += str(x)
    return key_tmp

def build_model(numBayX, model_pass, size_pass, save=False, base_hinges=False, md = 1.0, kd = 1.0, cd = 0.05, tmd=False):
    """
    Builds the 2D Steel Moment resisting Frame Model
    : numBayX: number of spans of X axis
    : model_size_pass: file path of beam sections used for the structural model
    : size_pass: file path of the section list of beams
    """
    df_model = pd.read_csv(model_pass)
    df_size = pd.read_csv(size_pass)

    # initialize openseespy
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    L_x = 20*ft
    # read floor levels
    Floor_level = df_model.loc[:, 'Elevation'].to_numpy()
    Floor = len(Floor_level)
    # add ground level
    Floor_level = np.insert(Floor_level, 0, 0)

    node_coords = []  # node coordinations
    connectivity = [] 

    save_json = dict()

    # define location of node for column head and bottom
    total_node = 1

    for f in range(Floor+1):
        for x in range(numBayX + 1):
            node_coords.append([x*L_x, Floor_level[f]*inc])
            # 柱用ノードの保存
            key_tmp = calc_key(f, x)
            save_json[str(key_tmp)] = dict()
            save_json[str(key_tmp)]['node_column'] = total_node
            save_json[str(key_tmp)]['X'] = x*L_x
            save_json[str(key_tmp)]['Y'] = Floor_level[f]*inc
            total_node += 1
    num_hinge = len(node_coords)-numBayX-1
    column_node = len(node_coords) # the number of nodes for colums


    node_hinges = dict() # return hinge Tag from node Tag
    node_hinges_beam_size = dict() # returns beam codes and shear spans of connected beam.
    loaded_nodes = [] # list of loaded nodes


    # define nodes for hinges
    hinge_tmp = 1
    beam_tmp = 1
    for f in range(Floor+1):
        for x in range(numBayX + 1):
            if f != 0:
                if x == 0 or x == numBayX:
                    node_coords.append([x*L_x, Floor_level[f]*inc])
                    node_hinges[str(beam_tmp)] = [column_node + hinge_tmp]
                    hinge_tmp += 1
                    
                    key_tmp = calc_key(f, x)
                    if x==numBayX:
                        save_json[str(key_tmp)]['node_leftbeam'] = total_node
                        total_node += 1
                    elif x==0:
                        save_json[str(key_tmp)]['node_rightbeam'] = total_node
                        total_node += 1

                else:
                    node_coords.append([x*L_x, Floor_level[f]*inc])
                    node_coords.append([x*L_x, Floor_level[f]*inc])
                    node_hinges[str(beam_tmp)] = [column_node + hinge_tmp, column_node + hinge_tmp + 1]
                    hinge_tmp += 2
                    key_tmp = calc_key(f, x)
                    save_json[str(key_tmp)]['node_leftbeam'] = total_node
                    total_node += 1
                    save_json[str(key_tmp)]['node_rightbeam'] = total_node
                    total_node += 1
                loaded_nodes.append(beam_tmp)
            beam_tmp += 1
    hinge_tmp -= 1

    # 柱脚用のノードを追加
    if base_hinges:
        for x in range(numBayX + 1):
            node_coords.append([x*L_x, Floor_level[0]*inc])

    beam_tag = []
    column_tag = []
    hinge_tag = []
    hinge_code_set = set()
    node_hinges_hinge_code = dict() # return hinge-code from beam-end node Tag

    # define beam connectivity
    num_beam = 0
    for f in range(1, Floor+1):
        for x in range(numBayX):
            column_i = (numBayX + 1)*f + x + 1
            column_j = column_i + 1
            key_i = calc_key(f, x)
            key_j = calc_key(f, x + 1)
            # get node Tag of both sides of beam end
            beam_i=  max(node_hinges[str(column_i)])
            beam_j = min(node_hinges[str(column_j)])
            connectivity.append([beam_i, beam_j])
            num_beam += 1
            beam_tag.append(df_model.loc[f-1, 'Beam Size'])
            save_json[str(key_i)]['right_beam'] = num_beam
            save_json[str(key_j)]['left_beam'] = num_beam

            # ヒンジを作るために，梁の符号とスパンを管理する．
            span = L_x
            node_hinges_beam_size[str(beam_i)] = [df_model.loc[f-1, 'Beam Size'], span]
            node_hinges_beam_size[str(beam_j)] = [df_model.loc[f-1, 'Beam Size'], span]
            node_hinges_hinge_code[str(beam_i)] = str(df_model.loc[f-1, 'Beam Size']) + '_' + str(span)
            node_hinges_hinge_code[str(beam_j)] = str(df_model.loc[f-1, 'Beam Size']) + '_' + str(span)
            hinge_code_set.add(str(df_model.loc[f-1, 'Beam Size']) + '_' + str(span))


    # column 
    num_column = 0
    for f in range(Floor):
        for x in range(numBayX+1):
            column_i = (numBayX + 1)*f + x+1
            column_j = column_i + numBayX + 1
            key_i = calc_key(f, x)
            key_j = calc_key(f + 1, x)
            connectivity.append([column_i, column_j])
            num_column += 1
            if x==0 or x==numBayX:
                # 側柱
                column_tag .append(df_model.loc[f, 'Exterior Column Size'])
            else:
                # 中柱
                column_tag .append(df_model.loc[f, 'Interior Column Size'])
            save_json[str(key_i)]['upper_column'] = num_column + num_beam
            save_json[str(key_j)]['under_column'] = num_column + num_beam
    
    # create bilin material for hinges
    hinge_code_hinge_tag = dict()
    hinge_materials = []
    mat_Tag = 2
    for hinge_code in hinge_code_set:
        beam_size, span = hinge_code.split('_')
        span = float(span)
        # create bilin material from beam size.
        df_size = pd.read_csv(size_pass)
        hinge_material = create_bilin(df_size, beam_size, span, mat_Tag)
        hinge_materials.append(hinge_material)
        hinge_code_hinge_tag[hinge_code] = mat_Tag
        mat_Tag += 1
   

    num_member = num_column+ num_beam
    
    # the connectivity of hinges
    for f in range(1, Floor+1):
        for x in range(numBayX + 1):
            # current_column_node
            key_i = calc_key(f, x)
            curr_column_node = (numBayX + 1)*f + x + 1

            left_or_right = 0
            for node in node_hinges[str(curr_column_node)]:
                num_member += 1
                connectivity.append([curr_column_node, node])
                if x == numBayX:
                    save_json[key_i]['left_hinge'] = num_member
                elif x==0:
                    save_json[key_i]['right_hinge'] = num_member
                elif left_or_right==0:
                    save_json[key_i]['left_hinge'] = num_member
                elif left_or_right==1 or x == 0:
                    save_json[key_i]['right_hinge'] = num_member
                hinge_code_tmp = node_hinges_hinge_code[str(node)]
                hinge_tag_tmp = hinge_code_hinge_tag[hinge_code_tmp]
                hinge_tag.append(hinge_tag_tmp)
                left_or_right += 1
    # Get Number of elements
    nel = len(connectivity)

    # Distinguish beams, columns & hinges by their element tag ID
    all_the_beams = [int(i+1) for i in range(num_beam)]
    all_the_cols = [int(num_beam+i + 1) for i in range(num_column)]
    all_the_hinges = [int(num_beam+num_column+i + 1) for i in range(hinge_tmp)]
    
    mat_S355 = {'ID': 'Steel01', 
                'matTag':1, 
                'Fy':55*ksi, 
                'E0':200000*N/mm**2, 
                'b':0.01}
    
    # read beam column section used for this structural model.
    sections = set_section(size_pass, set(beam_tag).union(set(column_tag)))

    # Main Nodes
    [ops.node(n+1,*node_coords[n])
     for n in range(len(node_coords))]

    # Boundary Conditions    
    # Fixing the Base Nodes
    if base_hinges:
        [ops.fix(n, 1, 1, 1) for n in [len(node_coords)-int(i) for i in range(numBayX+1)]]
        [ops.fix(n, 1, 1, 0) for n in [int(i+1) for i in range(numBayX+1)]]
    else:
        [ops.fix(n, 1, 1, 1) for n in [int(i+1) for i in range(numBayX+1)]]

    # Tie the displacements (not rotations) in plastic hinges:
    cnt = 0

    for node in node_hinges.keys():
        for tmp in node_hinges[str(node)]:
            ops.equalDOF(int(node), tmp, *[1, 2])

    # Material
    ops.uniaxialMaterial(*mat_S355.values())
    for hinge_material in hinge_materials:
        ops.uniaxialMaterial(*hinge_material.values())

    if base_hinges==True:
        # to apply hinges at column base.
        base_hinge_connectivity = []
        base_column_codes = []
        for x in range(numBayX + 1):
            base_hinge_connectivity.append([int(x+1), len(node_coords)-numBayX+int(x)]) #最初numBayX+1点と最後numBayX+1点を接続する．
        
            # read the parameters of column
            if x==0 or x==numBayX:
                # side column
                base_column_codes.append(df_model.loc[0, 'Exterior Column Size'])
            else:
                # inner column
                base_column_codes.append(df_model.loc[0, 'Interior Column Size'])
            
            # create bilin material for column
            Floor_span=  Floor_level[1]-Floor_level[0] # floor hight
            hinge_material = create_bilin(df_size, base_column_codes[-1], Floor_span, mat_Tag, RBS=False)
            ops.uniaxialMaterial(*hinge_material.values())
            # create zerolength element for column base.
            ops.element('zeroLength', all_the_hinges[-1]+x+1, *base_hinge_connectivity[-1],
                '-mat', mat_Tag, '-dir', 6)
            

            mat_Tag += 1
            ops.equalDOF(*base_hinge_connectivity[-1],  *[1, 2, 3])
    

    # Transformations
    ops.geomTransf('PDelta', 1)
    
    # Adding Elements
    # Beams
    # Beams
    [ops.element('elasticBeamColumn', all_the_beams[e], *connectivity[all_the_beams[e]-1],
            sections[beam_tag[e]]['A'], mat_S355['E0'],
            sections[beam_tag[e]]['I1'], 1) 
     for e in range (len(all_the_beams))]
    
    # Column
    [ops.element('elasticBeamColumn', all_the_cols[e], *connectivity[all_the_cols[e]-1],
            sections[column_tag[e]]['A'], mat_S355['E0'],
            sections[column_tag[e]]['I1'], 1) 
     for e in range(len(all_the_cols))]

    [ops.element('zeroLength', all_the_hinges[e], *connectivity[all_the_hinges[e]-1],
                '-mat', hinge_tag[e], '-dir', 6) for e in range(len(all_the_hinges))]
    
    # print(len(node_coords))
    total_node = len(node_coords) + 1
    if tmd == True:
        tmd_base_node_tag = save_json['2003']['node_column']
        tmd_x = save_json['2001']['X']
        tmd_y = save_json['2001']['Y']
        tmd_node_tag = total_node
        tmd_mat_tag = mat_Tag
        mat_Tag += 1
        damp_mat_tag = mat_Tag
        mat_Tag += 1
        mat_type = 'Steel01'
        mat_props = [10.0, kd, 1.0]
        
        c_d = cd  # damping factor
        alpha = 1.0  # 線形粘性
        ops.uniaxialMaterial('Viscous', damp_mat_tag, cd, alpha)
        ops.uniaxialMaterial(mat_type, tmd_mat_tag, *mat_props)

        ops.node(tmd_node_tag, tmd_x, tmd_y)
        ops.element('zeroLength',len(ops.getEleTags()) + 1 , tmd_base_node_tag, tmd_node_tag, '-mat', tmd_mat_tag, '-dir', 1)
        ops.element('zeroLength',len(ops.getEleTags()) + 2 , tmd_base_node_tag, tmd_node_tag, '-mat', damp_mat_tag, '-dir', 1)
        ops.mass(tmd_node_tag, md, 0.0, 0.0)
        ops.equalDOF(tmd_base_node_tag, tmd_node_tag, *[2, 3])
    
    global m_1
    global m_1
    Dead_Load = 90*psf*L_x + 25*psf*L_x # kg/m
    Live_Load_floor = 50*psf * L_x # kg/m
    Live_Load_roof = 20 * psf * L_x # kg/m
    
    D_L_floor = 9.80*Dead_Load*N + 9.80*Live_Load_floor*N # N/m
    D_L_roof = 9.80*Dead_Load*N + 9.80*Live_Load_roof*N    # N/m

    # C_L = 50.0*(KN)      # Concentrated load
    m_floor = Dead_Load*L_x + Live_Load_floor*L_x # kg
    m_roof = Dead_Load*L_x + Live_Load_roof*L_x  # kg

    # Now, loads & lumped masses will be added to the domain.
    # loaded_nodes = [3,4,5,6,7,8,9,10, 11, 12]
    loaded_elems = all_the_beams

    ops.timeSeries('Linear',1,'-factor',1.0)
    ops.pattern('Plain', 1, 1)

    ops.eleLoad('-ele', *loaded_elems[:-3],'-type', '-beamUniform',-D_L_floor)
    ops.eleLoad('-ele', *loaded_elems[-3:],'-type', '-beamUniform',-D_L_roof)
    [ops.mass(n, *[m_floor,0,0]) for n in loaded_nodes[:-4]]
    [ops.mass(n, *[m_roof,0,0]) for n in loaded_nodes[-4:]]
    
    print('Model built successfully!')
    if save:
        for key in save_json.keys():
            for key_key in ['node_column', 'X', 'Y', 'node_leftbeam', 'node_rightbeam', 'left_beam', 'right_beam', 'under_column', 'upper_column', 'left_hinge', 'right_hinge']:
                if key_key not in save_json[key].keys():
                    save_json[key][key_key] = ' '
        return save_json

def run_gravity(steps = 10):
        
    """
    Runs gravity analysis.
    Note that the model should be built before
    calling this function.
    
    Keyword arguments:
    steps -- total number of analysis steps

    """
    
    ops.initialize()
    # Records the response of a number of nodes at every converged step
    ops.recorder('Node', '-file', 'FGU_2SSMRF_files/Gravity_Reactions.out',
                 '-time','-node', *[1,2], '-dof', *[1,2,3], 'reaction')

    # plain constraint handler enforces homogeneous single point constraints
    ops.constraints('Plain')

    # RCM numberer uses the reverse Cuthill-McKee scheme to order the matrix equations
    ops.numberer('RCM')

    # Constructs a profileSPDSOE (Symmetric Positive Definite) system of equation object
    ops.system('ProfileSPD')

    # Uses the norm of the left hand side solution vector of the matrix equation to
    # determine if convergence has been reached
    ops.test('NormDispIncr', 1.0e-6, 100, 0, 2)

    # Uses the Newton-Raphson algorithm to solve the nonlinear residual equation
    ops.algorithm('Newton')

    # Uses LoadControl integrator object
    ops.integrator('LoadControl', 0.1)

    # Constructs the Static Analysis object
    ops.analysis('Static')

    # Records the current state of the model
    ops.record()
    # Performs the analysis
    ops.analyze(steps)    
    
    print("Gravity analysis Done!")

def run_modal(n_evs = 2):
    """
    Runs Modal analysis.
    Note that the model should be built before calling this function.

    ### Keywardarguments:
    n_envs -- number of eigenvalues
    """

    ops.initialize()

    # Constructs a transformation constraint handler, 
    # which enforces the constraints using the transformation method.
    ops.constraints('Transformation')

    # Constructs a Plain degree-of-freedom numbering object
    # to provide the mapping between the degrees-of-freedom at
    # the nodes and the equation numbers.
    ops.numberer('Plain')

    # Construct a BandGeneralSOE linear system of equation object
    ops.system('BandGen')

    # Uses the norm of the left hand side solution vector of the matrix equation to
    # determine if convergence has been reached
    ops.test('NormDispIncr', 1.0e-6, 25, 0, 2)

    # Uses the Newton-Raphson algorithm to solve the nonlinear residual equation
    ops.algorithm('Newton')

    # Create a Newmark integrator.
    ops.integrator('Newmark', 0.5, 0.25)

    # Constructs the Transient Analysis object
    ops.analysis('Transient')

    # Eigenvalue analysis
    lamda = np.array(ops.eigen(n_evs))
    print('lamda', lamda)

    # engen values
    eigen_values = []
    for l in lamda:
        lamda, omega, period, frequency = l, l**0.5, 2*np.pi/(l**0.5), (l**0.5)/(2*np.pi)
        eigen_values.append([lamda, omega, period, frequency])
    

    eigen_vectors = []
    eigen_nodes = [1, 3, 5, 7, 9, 11]
    for i in range(n_evs):
        eigen_vectors.append([ops.nodeEigenvector(j, i+1, 1) for j in eigen_nodes])


    
    ops.record()
    print('Modal analysis Done!')
    return eigen_values, eigen_vectors


def run_pushover(eigen_vector, steps = 5000):
    
    """
    Runs Pushover analysis.
    Note that the model should be built before
    calling this function. Also, Gravity analysis
    should be called afterwards. Morover, the function
    requires some components of eigenvectors obtained 
    by calling the `run_modal` function.
    
    Keyword arguments:
    eigen_vector -- list of eigen vector (mode 1)
    steps -- total number of analysis steps

    """    
    
    ## Records the response of a number of nodes at every converged step
    # Global behaviour
    # records horizontal reactions of node 1 & 2
    ops.recorder('Node', '-file',
                 'FGU_2SSMRF_files/Pushover_Horizontal_Reactions.out',
                 '-time','-node', *[1,2], '-dof', 1, 'reaction')
    # records horizontal displacements of node 3 & 5
    ops.recorder('Node','-file',
                 'FGU_2SSMRF_files/Pushover_Story_Displacement.out',
                 '-time','-node', *[3,5], '-dof',1, 'disp')

    # Local behaviour
    # records Mz_1 & Mz_2 for each hinge. other forces are zero
    ops.recorder('Element','-file',
                 'FGU_2SSMRF_files/Pushover_BeamHinge_GlbForc.out',
                 '-time','-ele', *[7,8,9,10],'force') 
    # records the rotation of each hinges, ranging from 7 to 10
    ops.recorder('Element','-file',
                 'FGU_2SSMRF_files/Pushover_BeamHinge_Deformation.out',
                 '-time','-eleRange',*[7, 10], 'deformation') 

    # records Px_1,Py_1,Mz_1,Px_2,Py_2,Mz_2 for elements 1 to 4 
    ops.recorder('Element','-file',
                 'FGU_2SSMRF_files/Pushover_Column_GlbForc.out',
                 '-time','-eleRange',*[1,4], 'globalForce')
    # eps, theta_1, theta_2 for elements 1 to 4
    ops.recorder('Element','-file', 
                 'FGU_2SSMRF_files/Pushover_Column_ChordRot.out',
                 '-time','-ele', *[1,2,3,4], 'chordRotation')    

    # Measure analysis duration
    tic = time.time()

    # load eigenvectors for mode 1
    phi = np.abs(np.array(eigen_vector))
    

    # Apply lateral load based on first mode shape in x direction (EC8-1)
    ops.pattern('Plain', 2, 1)
    [ops.load(n, *[m_1*phi[1],0,0]) for n in [3,4]]  
    [ops.load(n, *[m_1*phi[2],0,0]) for n in [5,6]]
    [ops.load(n, *[m_1*phi[3],0,0]) for n in [7,8]]  
    [ops.load(n, *[m_1*phi[4],0,0]) for n in [9,10]]
    [ops.load(n, *[m_1*phi[5],0,0]) for n in [11,12]]


    # Define step parameters
    step = +1.0e-04
    number_of_steps = steps

    # Constructs a transformation constraint handler, 
    # which enforces the constraints using the transformation method.
    ops.constraints('Transformation')

    # RCM numberer uses the reverse Cuthill-McKee scheme to order the matrix equations
    ops.numberer('RCM')

    # Construct a BandGeneralSOE linear system of equation object
    ops.system('BandGen')

    # Uses the norm of the left hand side solution vector of the matrix equation to
    # determine if convergence has been reached
    ops.test('NormDispIncr', 0.0001, 100)

    # Line search increases the effectiveness of the Newton method
    # when convergence is slow due to roughness of the residual.
    ops.algorithm('NewtonLineSearch',True, 0.8,
                  1000, 0.1, 10.0)

    # Displacement Control tries to determine the time step that will
    # result in a displacement increment for a particular degree-of-freedom
    # at a node to be a prescribed value.
    # Target node is 5 and dof is 1
    ops.integrator('DisplacementControl',5,1, step)

    # Constructs the Static Analysis object
    ops.analysis('Static')

    # Records the current state of the model
    ops.record()

    outputs = {
        'time':[], 
        'horizontal_reaction': [[], []],
        'story_displacement':[[], []], 
        'story_displacement':[[], []], 
    }

    # Performs the analysis
    # ops.analyze(number_of_steps)
    for _ in range(number_of_steps):
        ops.analyze(1)
        curr_time = ops.getTime()
        outputs['time'].append(curr_time)
        outputs['horizontal_reaction'][0].append(ops.nodeReaction(1, 1))
        outputs['horizontal_reaction'][1].append(ops.nodeReaction(2, 1))

        outputs['story_displacement'][0].append(ops.nodeDisp(3, 1))
        outputs['story_displacement'][1].append(ops.nodeDisp(5, 1))

    # calculate analysis time 
    toc = time.time()

    print('Pushover Analysis Done in {:1.2f} seconds'.format(toc-tic))
    return outputs


def run_time_history(omega, ground_accel, g_motion_id = 1, scaling_id = 1,
                     lamda = 1):

    """
    Runs Time history analysis.
    Note that the model should be built before
    calling this function. Also, Gravity analysis
    should be called afterwards.
    
    ### Keyword arguments:
    omega --First natural circular frequency 一次固有円振動数 [mode1, mode2]
    ground_accel -- ground acceleration np.array([float, float, ])
    g_motion_id -- Groundmotion id (in case you run many GMs, like in an IDA)
    scaling_id -- Scaling id (in case you run many GMs, like in an IDA)
    lamda -- Scaling of the GM
    acc_dir -- file directory of GM to read from
    """

    ## Records the response of a number of nodes at every converged step
    # Global behaviour
    # records horizontal reactions of node 1 & 2
    ops.recorder('Node','-file',
                ('FGU_2SSMRF_files/TimeHistory_Horizontal_Reactions.'
                 +str(g_motion_id)+'.'+str(scaling_id)+'.out'),
                '-time','-node',*[1,2],'-dof',1,'reaction')
    # records horizontal displacements of node 3 & 5
    ops.recorder('Node','-file',
                ('FGU_2SSMRF_files/TimeHistory_Story_Displacement.'
                 +str(g_motion_id)+'.'+str(scaling_id)+'.out'),
                '-time','-node',*[3,5],'-dof',1,'disp')

    # Local behaviour
    # records Mz_1 & Mz_2 for each hinge. other forces are zero
    ops.recorder('Element','-file',
                ('FGU_2SSMRF_files/TimeHistory_BeamHinge_GlbForc.'
                 +str(g_motion_id)+'.'+str(scaling_id)+'.out'),
                '-time','-ele',*[7,8,9,10],'force')
    # records the rotation of each hinges, ranging from 7 to 10
    ops.recorder('Element','-file',
                ('FGU_2SSMRF_files/TimeHistory_BeamHinge_Deformation.'
                 +str(g_motion_id)+'.'+str(scaling_id)+'.out'),
                '-time','-eleRange',*[7,10],'deformation')
    # records Px_1,Py_1,Mz_1,Px_2,Py_2,Mz_2 for elements 1 to 4 
    ops.recorder('Element','-file',
                ('FGU_2SSMRF_files/TimeHistory_Column_GlbForc.'
                 +str(g_motion_id)+'.'+str(scaling_id)+'.out'),
                '-time','-eleRange',*[1,4],'globalForce')
    # eps, theta_1, theta_2 for elements 1 to 4
    ops.recorder('Element','-file',
                ('FGU_2SSMRF_files/TimeHistory_Column_ChordRot.'
                 +str(g_motion_id)+'.'+str(scaling_id)+'.out'),
                '-time','-ele',*[1,2,3,4],'chordRotation')   


    # Reading omega for obraining Rayleigh damping model
    xis = np.array([0.03, 0.03]) 
    a_R, b_R = 2*((omega[0]*omega[1])/(omega[1]**2-omega[0]**2))*(
        np.array([[omega[1],-omega[0]],
                  [-1/omega[1],1/omega[0]]])@xis)
    
    
    accelerogram = ground_accel  # Loads accelerogram file
    ## Analysis Parameter
    dt = 0.01                               # Time-Step
    n_steps = len(accelerogram)             # Number of steps
    tol = 5.0e-5                            # prescribed tolerance
    # tol = 1.0e-6
    max_iter = 50000                        # Maximum number of iterations per step




    # Uses the norm of the left hand side solution vector of the matrix equation to
    # determine if convergence has been reached
    ops.test('NormDispIncr', tol, max_iter,0,0)

    # RCM numberer uses the reverse Cuthill-McKee scheme to order the matrix equations
    ops.numberer('RCM')

    # Construct a BandGeneralSOE linear system of equation object
    ops.system('BandGen')

    # The relationship between load factor and time is input by the user as a 
    # series of discrete points
    ops.timeSeries('Path', 2, '-dt', dt, '-values', *accelerogram, '-factor', lamda)

    # allows the user to apply a uniform excitation to a model acting in a certain direction
    ops.pattern('UniformExcitation', 3, 1,'-accel', 2)

    # Constructs a transformation constraint handler, 
    # which enforces the constraints using the transformation method.
    ops.constraints('Transformation')

    # Create a Newmark integrator.
    ops.integrator('Newmark', 0.5, 0.25)

    # assign damping to all previously-defined elements and nodes
    ops.rayleigh(a_R, b_R, 0.0, 0.0)

    # Introduces line search to the Newton algorithm to solve the nonlinear residual equation
    ops.algorithm('NewtonLineSearch',True,False,False,False,0.8,100,0.1,10.0)
    # ops.algorithm('ModifiedNewton', False, False)

    # Constructs the Transient Analysis object
    ops.analysis('Transient')

    # Measure analysis duration
    t = 0
    ok = 0
    print('Running Time-Histroy analysis with lambda=',lamda)
    start_time = time.time()
    final_time = ops.getTime() + n_steps*dt
    dt_analysis = dt

    outputs = {
        'time': [], 
        'horizontal_reaction':[[] for _ in range(2)], 
        'story_displacement':[[] for _ in range(2, 22)], 
        'floor_acceleration':[[] for _ in range(2, 22)], # relative acceleraion
        'beamhinge_glbforc':[[] for _ in range(2)],  # 梁端部のヒンジの力
        'beamhinge_deformation': [[] for _ in range(2)],  # Beam end hinge deformations
        'column_glbforc': [[] for i in range(2)], #  # Column forces / Global forces in columns
        'column_chordrot': [[]for _ in range(4)], # Column chord rotations
    }

    hinges = [146, 260]
    observe_nodes = [int(i)*4 for i in range(2, 22)]

    while (ok == 0 and t <= final_time):
        ok = ops.analyze(1, dt_analysis)
        t = ops.getTime()    
        outputs['time'].append(t)
        outputs['horizontal_reaction'][0].append(ops.nodeReaction(1, 1))
        outputs['horizontal_reaction'][1].append(ops.nodeReaction(2, 1))

        for i in range(len(observe_nodes)):
            outputs['story_displacement'][i].append(ops.nodeDisp(observe_nodes[i], 1))
        
            outputs['floor_acceleration'][i].append(ops.nodeAccel(observe_nodes[i], 1))

        # hinge_glbforc
        for i in range(len(hinges)):
            hinge = hinges[i]
            # print(ops.eleForce(hinge, 0))
            outputs['beamhinge_glbforc'][i].append(ops.eleForce(hinge, 0)[2])
            if hinge == hinges[0]:
                outputs['beamhinge_deformation'][0].append(ops.eleResponse(hinge, "deformation")[0])
            if hinge == hinges[1]:
                outputs['beamhinge_deformation'][1].append(ops.eleResponse(hinge, "deformation")[0])
    # print(ops.eleForce(hinge, 0), ops.eleForce(hinge, 1), ops.eleForce(hinge, 2))
    finish_time = time.time()
    
    if ok == 0:
        print('Time-History Analysis Done in {:1.2f} seconds'.format(finish_time-start_time))
    else:
        print('Time-History Analysis Failed in {:1.2f} seconds'.format(finish_time-start_time))
    
    ops.wipe()
    return outputs


def reset_analysis():
    """
    Resets the analysis by setting time to 0,
    removing the recorders and wiping the analysis.
    """    
    
    # Reset for next analysis case
    ##  Set the time in the Domain to zero
    ops.setTime(0.0)
    ## Set the loads constant in the domain
    ops.loadConst()
    ## Remove all recorder objects.
    ops.remove('recorders')
    ## destroy all components of the Analysis object
    ops.wipeAnalysis()    


def make_train_dataset_SMF(model_pass, size_pass, response, EQ_raw, batch_size, use_GPU=True, normalize=False, x_max=0.0, y_max=0.0, use_filtered=False, EQ_filtered=None, half=False,base_hinges=False, md = 1.0, kd = 1.0, cd = 0.05, tmd=False):
    """
    学習用データセットを作成する関数
    --------------
    :EQ_raw: 生の地震動波形
    :EQ_filtered: バンドパスフィルタをかけた地震動波形
    """
    if torch.cuda.is_available() and use_GPU:
        device='cuda'
    else:
        device='cpu'
    
    x_data = []
    y_data = []
    output_length = len(EQ_raw[0])
    for i in range(len(EQ_raw)):
        ops.wipe()
        build_model(3, model_pass, size_pass, base_hinges=base_hinges, md = 1.0, kd = 1.0, cd = 0.05, tmd=False)
        run_gravity()
        eigen_value, eigen_vector = run_modal() # eigen value analysis
        print(eigen_value)
        reset_analysis()
        run_gravity()
        omega = [eigen_value[0][1], eigen_value[1][1]] # result of eigen value analysis
        outputs = run_time_history(omega, EQ_raw[i]) # NLTHA
        ops.wipe()
        if use_filtered:
            x_data.append(EQ_filtered[i][:output_length])
        else:
            x_data.append(EQ_raw[i][:output_length])
        
        # y_data.append([[np.array(outputs['beamhinge_deformation'][0]), # deformation of hinge 7 
        #                np.array(outputs['beamhinge_deformation'][1])], # deformation of hinge 10
        #                [np.array(outputs['beamhinge_glbforc'][0][0]),  # glbforce of hinge 7
        #                np.array(outputs['beamhinge_glbforc'][3][0]),]  # glbforce of hinge 10
        #                ])
        if response == 'deformation':
            # hinge_deformation
            y_data.append([ # deformation of hinge 10
                    [np.array(outputs['beamhinge_deformation'][0]), # deformation of hinge 7 
                    np.array(outputs['beamhinge_deformation'][1])]  # glbforce of hinge 10
                    ])

        elif response == 'disp':
            y_data.append([[
                        np.array(outputs['story_displacement'][i]) for i in range(len(outputs['story_displacement']))   # displacement of node 3 
                        ]
                    ])
        
        elif response == 'accel':
            y_data.append([
                    [
                        np.array(outputs['floor_acceleration'][i]) for i in range(len(outputs['floor_acceleration'])) # acceleration of node 5
                        ]
                    ])
        elif response == 'glbforc':
            y_data.append([
                [
                np.array(outputs['beamhinge_glbforc'][0]),  # glbforce of hinge 7
                np.array(outputs['beamhinge_glbforc'][1]),
                ]
            ])


        
    if normalize:
        x_data = np.array(x_data)
        y_data = np.array(y_data) # convert to ndarray
        if type(y_max) == float:
            # 最初ymaxが0.0で設定されているので、その場合は次元をそろえる。
            # print('y_data_shape', y_data.shape)
            y_max= np.zeros_like(np.amax(np.abs(y_data), axis=(0, 3)))
        
        x_max = max(np.max(np.abs(x_data)), x_max)
        # print('y_data_shape', y_data.shape)
        y_max = np.maximum(np.amax(np.abs(y_data), axis=(0, 3)), y_max)

        x_data = x_data / x_max
        # y_maxを欲しい形にブロードキャストする
        y_max_broadcast = y_max[np.newaxis, :, :, np.newaxis]
        y_data=y_data / y_max_broadcast
    
    if half:
        x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
        y_data = torch.tensor(y_data, dtype=torch.float32).to(device)
    else:    
        x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
        y_data = torch.tensor(y_data, dtype=torch.float32).to(device)

    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if normalize: 
        return dataloader, x_max, y_max
    else:
        return dataloader
