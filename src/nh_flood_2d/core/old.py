from typing import no_type_check

@no_type_check
def run_flood(shared, solution_data, resource_path, step, re_coo_index, watergroups = None):
    import os           # 操作系统接口
    import math         # 数学函数库
    import time         # 时间处理
    import numpy as np   # 数值计算库
    import taichi as ti  # 高性能并行计算框架
    from re_coo import re_coo  # 坐标重构模块
    from datetime import datetime  # 日期时间处理

    ti.init(arch=ti.gpu,kernel_profiler=True)  # 初始化Taichi，使用GPU加速并启用性能分析
    
    # 获取时间步索引
    time_index = step
    # 从解决方案数据中提取各种输入数据
    inp_path = solution_data['inp_path']        # 输入文件路径
    ne_data = solution_data['ne_data']          # 网格单元数据
    ns_data = solution_data['ns_data']          # 网格边界数据
    rainfall_data = solution_data['rainfall_data']  # 降雨数据
    tides_data = solution_data['tides_data']    # 潮汐数据
    gate_data = solution_data['gate_data']      # 闸门数据
    # 提取闸门相关参数
    ud = gate_data.ud_stream_list               # 闸门上下游标识列表
    gate_h = gate_data.gate_height_list         # 闸门高度列表
    gate_grid_id = gate_data.grid_id_list       # 闸门对应的网格ID列表
    gate_list = []                              # 初始化闸门列表
    
    # 构建闸门数据列表
    for i in range(len(gate_h)):
        # 每个闸门包含：[上游水位, 下游水位, 闸门高度, 网格ID...]
        gate_list.append([ud[2*i], ud[2*i+1],gate_h[i]])
        gate_list[i].extend(gate_grid_id[i])  # 添加闸门影响的网格ID

    # 定义物理参数和常数
    h_min: ti.f32=0.02     # 最小水深阈值(m)
    g: ti.f32=9.8          # 重力加速度(m/s²)
    n: ti.f32=0.033        # 曼宁粗糙系数
    ns:ti.i32=len(ns_data.edge_id_list)  # 网格边界总数
    ne:ti.i32=len(ne_data.grid_id_list)  # 网格单元总数
    nbd_ie = ti.field(ti.i32, shape=())  # 边界网格数量

    # 数值计算参数
    afa:ti.f32 = 0.5       # CFL条件系数
    sita:ti.f32 = 1.0      # 时间权重系数
    pi=3.1415926           # 圆周率


    # 定义时间相关的Taichi字段（标量）
    dt=ti.field(ti.f32, shape=())           # 时间步长
    rainq_t= ti.field(ti.f32, shape=())     # 当前时刻降雨强度
    tide=ti.field(ti.f32, shape=())         # 当前时刻潮汐水位
    timesteps=ti.field(ti.f32, shape=())    # 累计时间步数
    total_time=ti.field(ti.f32, shape=())   # 总模拟时间
    
    # 定义网格几何信息字段
    ze= ti.field(dtype=ti.f32, shape=(ne+1))  # 网格底高程
    xe=ti.field(dtype=ti.f32, shape=(ne+1))  # 网格X坐标
    ye=ti.field(dtype=ti.f32, shape=(ne+1))  # 网格Y坐标

    # 定义水动力变量字段
    h = ti.field(dtype=ti.f32, shape=(ne+1))      # 当前时刻水位
    hn = ti.field(dtype=ti.f32, shape=(ne + 1))   # 上一时刻水位
    u=ti.field(dtype=ti.f32, shape=(ne+1))       # X方向流速
    v=ti.field(dtype=ti.f32, shape=(ne+1))       # Y方向流速
    q_source=ti.field(dtype=ti.f32, shape=(ne+1))     # 源汇项流量
    q_source_flag=ti.field(dtype=ti.f32, shape=(ne+1)) # 源汇项标志
    node_types=ti.field(dtype=ti.i32, shape=(13000))   # 节点类型数组

    # 定义下渗相关字段
    under_suf=ti.field(dtype=ti.i32, shape=(ne+1))  # 下垫面类型
    # 定义不同下垫面类型的下渗率
    qf1 = ti.field(dtype=ti.f32, shape=())  # 下垫面类型1的下渗率
    qf2 = ti.field(dtype=ti.f32, shape=())  # 下垫面类型2的下渗率
    qf3 = ti.field(dtype=ti.f32, shape=())  # 下垫面类型3的下渗率
    qf4 = ti.field(dtype=ti.f32, shape=())  # 下垫面类型4的下渗率
    qf5 = ti.field(dtype=ti.f32, shape=())  # 下垫面类型5的下渗率
    qf6 = ti.field(dtype=ti.f32, shape=())  # 下垫面类型6的下渗率
    qf7 = ti.field(dtype=ti.f32, shape=())  # 下垫面类型7的下渗率

    # 定义网格拓扑连接信息
    nsl1=ti.field(dtype=ti.i32, shape=(ne+1))  # 左侧边界数量
    nsl2=ti.field(dtype=ti.i32, shape=(ne+1))  # 右侧边界数量
    nsl3=ti.field(dtype=ti.i32, shape=(ne+1))  # 下方边界数量
    nsl4=ti.field(dtype=ti.i32, shape=(ne+1))  # 上方边界数量
    grid_area0=ti.field(dtype=ti.f32, shape=(ne+1))  # 网格面积
    ibd_ie = ti.field(dtype=ti.i32, shape=(ne + 1))  # 边界网格索引
    # 定义网格四个方向的连接边界索引
    isl1=ti.field(dtype=ti.i32, shape=(ne+1,10))  # 左侧连接边界索引
    isl2=ti.field(dtype=ti.i32, shape=(ne+1,10))  # 右侧连接边界索引
    isl3=ti.field(dtype=ti.i32, shape=(ne+1,10))  # 下方连接边界索引
    isl4=ti.field(dtype=ti.i32, shape=(ne+1,10))  # 上方连接边界索引 

    # 定义闸门相关字段
    gate=ti.field(dtype=ti.i32, shape=(100,100))        # 闸门网格连接矩阵

    # 定义边界信息字段
    ise=ti.field(dtype=ti.i32, shape=(ns+1, 5))         # 边界单元连接信息
    dis=ti.field(dtype=ti.f32,shape=(ns+1))             # 边界距离
    
    q_x = ti.field(dtype=ti.f32, shape=(ns+1))          # X方向边界流量
    q_y = ti.field(dtype=ti.f32, shape=(ns+1))          # Y方向边界流量
    qn_x= ti.field(dtype=ti.f32, shape=(ns+1))          # X方向上一时刻边界流量
    qn_y= ti.field(dtype=ti.f32, shape=(ns+1))          # Y方向上一时刻边界流量

    id_dx=ti.field(dtype=ti.i32, shape=(ns+1))          # X方向边界控制标志
    id_dy=ti.field(dtype=ti.i32, shape=(ns+1))          # Y方向边界控制标志
    h_dike=ti.field(dtype=ti.i32, shape=(ns+1))         # 堤坝高度
    dt3=ti.field(dtype=ti.f32, shape=(ns+1))            # 边界时间步长

#----------------------------------------------------
    # 初始化时间和计数变量
    cumulative_time = 0      # 累计计算时间
    tide1 = 0               # 当前潮汐水位值
    tide2 = 0               # 下一个潮汐水位值
    tide2_time = 0          # 下一个潮汐时间点
    tide1_time = 0          # 当前潮汐时间点
    cumulative_time2 = 0    # 累计计算时间2（用于数据交换）

    # 初始化降雨和模拟参数
    rainq=0.0               # 当前降雨强度
    time_run = 0            # 模拟运行时间
    nt = 0                  # 时间步计数器
    tide_num1 = 0           # 潮汐数据索引1
    rain1_num = 0           # 降雨数据索引1
    rain2_num = 0           # 降雨数据索引2
    begin_time = 0          # 模拟开始时间
    ibd_ie_np = np.zeros((ne+1),dtype=np.int32)  # 边界网格索引数组

    # 调用坐标重构函数，读取网格和边界数据
    re_coo(inp_path, resource_path, ne_data, ns_data, re_coo_index)

    # 读取节点索引文件，获取节点名称列表
    node_name_file = resource_path+'/'+"node_index.txt"
    node_id_list = []
    with open(node_name_file, "r", encoding='utf-8') as file:
        for line in file:
            elements = line.strip().split(',')
            node_id_list.append(elements[1])  # 提取节点名称
    
    # 根据节点名称确定节点类型（0为出水口，1为其他类型）
    node_type = []
    for i in range(len(node_id_list)):
        node_type.append(0 if node_id_list[i][:7] == "Outfall" else 1)
    node_types_np = np.array(node_type).astype(np.int32)

    # 定义需要转换为二维数组的网格连接列表
    input_lists = {
        'isl1': ne_data.isl1_list,  # 左侧连接边界索引列表
        'isl2': ne_data.isl2_list,  # 右侧连接边界索引列表
        'isl3': ne_data.isl3_list,  # 下方连接边界索引列表
        'isl4': ne_data.isl4_list,  # 上方连接边界索引列表
    }

    # 将不规则的连接列表转换为规则的二维数组
    output_arrays = {}

    for name, i_list in input_lists.items():
        # 找到最大的子列表长度，用于确定数组的第二维大小
        max_len = max(len(i_node) for i_node in i_list)
        # 创建零填充的二维数组
        ilist_array = np.zeros((len(i_list), max_len), dtype=np.int32)

        # 将每个子列表的数据复制到二维数组中
        for i, sonlist in enumerate(i_list):
            ilist_array[i, :len(sonlist)] = sonlist
        output_arrays[name] = ilist_array

    # 提取转换后的网格连接数组
    isl1_np = output_arrays['isl1']  # 左侧连接边界索引数组
    isl2_np = output_arrays['isl2']  # 右侧连接边界索引数组
    isl3_np = output_arrays['isl3']  # 下方连接边界索引数组
    isl4_np = output_arrays['isl4']  # 上方连接边界索引数组
    
    # 将网格数据转换为NumPy数组
    xe_np = np.array(ne_data.xe_list, dtype=np.float32)      # 网格X坐标数组
    ye_np = np.array(ne_data.ye_list, dtype=np.float32)      # 网格Y坐标数组
    ze_np = np.array(ne_data.ze_list, dtype=np.float32)      # 网格底高程数组
    nsl1_np = np.array(ne_data.nsl1_list, dtype=np.int32)    # 左侧边界数量数组
    nsl2_np = np.array(ne_data.nsl2_list, dtype=np.int32)    # 右侧边界数量数组
    nsl3_np = np.array(ne_data.nsl3_list, dtype=np.int32)    # 下方边界数量数组
    nsl4_np = np.array(ne_data.nsl4_list, dtype=np.int32)    # 上方边界数量数组
    dis_np = np.array(ns_data.dis_list, dtype=np.float32)    # 边界距离数组
    under_suf_np = np.array(ne_data.under_suf_list, dtype=np.int32)  # 下垫面类型数组
    ise_np = np.array(ns_data.ise_list, dtype=np.int32)      # 边界单元连接信息数组
    
    # 构建闸门数据的二维数组（100x100的固定大小）
    gate_np_list = [[0 for _ in range(100)] for _ in range(100)]
    for i, gate_i in enumerate(gate_list):
        if i < 100:  # 限制闸门数量不超过100个
            gate_np_list[i][:len(gate_i)] = gate_i  
    gate_np = np.array(gate_np_list, dtype=np.int32)
    
    # 提取潮汐和降雨数据
    tide_date = tides_data.tide_date_list      # 潮汐日期列表
    tide_time = tides_data.tide_time_list      # 潮汐时间列表
    tide_value = tides_data.tide_value_list    # 潮汐水位值列表
    rainfall_date = rainfall_data.rainfall_date_list    # 降雨日期列表
    rainfall_value = rainfall_data.rainfall_value_list  # 降雨强度值列表

#--------------------------------------------------
    # 数据复制和初始化阶段
    # 将NumPy数组数据复制到Taichi字段的通用函数
    @ti.kernel
    @no_type_check
    def copy_test_to_taichi1(npy: ti.types.ndarray(),tai: ti.template()):
        """将一维NumPy数组复制到Taichi字段"""
        for i in tai:
            tai[i] =npy[i]

    copy_test_to_taichi1(xe_np,xe)
    copy_test_to_taichi1(ye_np,ye)
    copy_test_to_taichi1(nsl1_np,nsl1)
    copy_test_to_taichi1(nsl2_np,nsl2)
    copy_test_to_taichi1(nsl3_np,nsl3)
    copy_test_to_taichi1(nsl4_np,nsl4)
    copy_test_to_taichi1(dis_np,dis)
    copy_test_to_taichi1(ze_np,ze)
    copy_test_to_taichi1(under_suf_np,under_suf)
    copy_test_to_taichi1(node_types_np,node_types)


    @ti.kernel
    @no_type_check
    def copy_test_to_taichi2(npy: ti.types.ndarray(), tai: ti.template()):
        """将二维NumPy数组复制到Taichi字段"""
        for j,k in tai:
            tai[j, k] = npy[j, k]
    
    # 使用复制函数将二维NumPy数组数据复制到Taichi字段
    copy_test_to_taichi2(isl1_np,isl1)          # 复制左侧连接边界索引
    copy_test_to_taichi2(isl2_np,isl2)          # 复制右侧连接边界索引
    copy_test_to_taichi2(isl3_np,isl3)          # 复制下方连接边界索引
    copy_test_to_taichi2(isl4_np,isl4)          # 复制上方连接边界索引
    copy_test_to_taichi2(ise_np,ise)            # 复制边界单元连接信息
    copy_test_to_taichi2(gate_np,gate)          # 复制闸门数据（二维数组）

    @ti.kernel
    def find_bd():
        """识别边界网格并统计边界网格数量"""
        nbd_ie[None] = 0  # 初始化边界网格计数器
        ti.loop_config(serialize=True)  # 配置循环串行化执行
        for i in range(1, ns+1):
            # 检查边界类型：如果ise[i,0]==1，表示Y方向边界
            if ise[i,0]==1:
                # 检查是否有一个网格为0（边界条件）
                if min(ise[i,3],ise[i,4])==0:
                    ie=max(ise[i,3],ise[i,4])  # 获取非零的网格索引
                    # 检查网格是否在指定的坐标范围内
                    if xe[ie]<808411 and ye[ie] < 837066:
                        nbd_ie[None] = nbd_ie[None] + 1  # 增加边界网格计数
                        ibd_ie[nbd_ie[None]] = ie        # 记录边界网格索引
            else:
                # X方向边界处理
                if min(ise[i,1],ise[i,2])==0:
                    ie=max(ise[i,1],ise[i,2])  # 获取非零的网格索引
                    # 检查网格是否在指定的坐标范围内
                    if xe[ie] < 808411 and ye[ie] < 837066:
                        nbd_ie[None] = nbd_ie[None] + 1  # 增加边界网格计数
                        ibd_ie[nbd_ie[None]] = ie        # 记录边界网格索引
    find_bd()
    print(f"Boundary grid count: {nbd_ie[None]}")
    ibd_ie_np = ibd_ie.to_numpy()
    @ti.kernel
    def initial_taichi():
        for ie in range(1,ne+1):
            h[ie]=ze[ie]  
            if h[ie] < 0: h[ie] = 0
            hn[ie] = h[ie]
            q_source_flag[ie]=1

        for iss in range(1,ns+1):
            id_dx[iss]=1
            id_dy[iss]=1
            q_x[iss]=0
            q_y[iss]=0
            qn_x[iss]=0
            qn_y[iss]=0
            h_dike[iss]=-999

        total_time[None]=0
        dt[None]=0.1
        timesteps[None]=0
        tide[None]=0

    @ti.kernel
    def gate_control():
        for i in range(100):
            gate_up=gate[i,0]
            gate_down=gate[i,1]
            gate_level=gate[i,2]
            gate_on = h[gate_up] + 0.1 > h[gate_down]
            if gate_on:
                for j in range(3,100):
                    gateid = gate[i,j]
                    ze[gateid] = 0
            else:
                for j in range(3,100):
                    gateid = gate[i,j]
                    ze[gateid] = gate_level

    @ti.kernel
    @no_type_check
    def update_flow():
        """更新边界流量计算（基于浅水方程的有限差分法）"""
        for iss in range(1,ns+1):
            # 初始化X方向计算参数
            dt1: ti.f32 = 100  # X方向时间步长初值
            ie1:ti.i32=ise[iss,1]   # X方向左侧网格索引
            ie2:ti.i32=ise[iss,2]   # X方向右侧网格索引
            flag_x:ti.f32=ti.min(ie1,ie2)/abs(ti.min(ie1,ie2)+0.0001)  # X方向边界标志
            h1:ti.f32=h[ie1]  # 左侧网格水位
            h2:ti.f32=h[ie2]  # 右侧网格水位
            z1:ti.f32=ze[ie1] # 左侧网格底高程
            z2:ti.f32=ze[ie2] # 右侧网格底高程
            hf_x:ti.f32=ti.max(h2,h1)-ti.max(z2,z1,h_dike[iss])  # X方向有效水深
            dx:ti.f32=ti.max(xe[ie2]-xe[ie1],0.01)  # X方向网格间距
            # 计算X方向压力梯度项
            q1:ti.f32 =-g*ti.max(hf_x,0)*dt[None]*(hn[ie2]-hn[ie1])/dx
            # 计算X方向摩阻项（曼宁公式）
            q2:ti.f32=1+g*dt[None]*(n**2)*ti.abs(qn_x[iss]/(ti.max(hf_x,0.00001)**(7/3)))

            # 获取相邻边界索引用于数值扩散
            side_l:ti.i32=isl1[ie1,1]  # 左侧相邻边界
            side_r:ti.i32=isl2[ie2,1]  # 右侧相邻边界
            # 计算X方向流量（包含时间权重和空间扩散）
            q_x[iss]=(sita*qn_x[iss]+(1-sita)/2*(qn_x[side_l]+qn_x[side_r])+q1)/q2
            q_x[iss]=q_x[iss]*id_dx[iss]  # 应用边界控制标志
            # 应用干湿边界处理
            q_x[iss]=q_x[iss] *ti.max((hf_x-h_min)/(abs((hf_x-h_min))+0.00001),0.)
            q_x[iss]=q_x[iss]*flag_x  # 应用边界方向标志
            qn_x[iss]=q_x[iss]  # 更新上一时刻X方向流量
            # 计算X方向CFL时间步长
            dt1: ti.f32 = afa * dis[iss]/ (ti.sqrt(g * ti.max(hf_x,0.01)) + ti.abs(q_x[iss])
                                           / ti.max(hf_x,0.01))
            
            # 初始化Y方向计算参数
            ie4:ti.i32 = ise[iss,4]  # Y方向上侧网格索引
            ie3:ti.i32= ise[iss,3]   # Y方向下侧网格索引
            flag_y:ti.f32 =ti.min(ie3,ie4)/abs(ti.min(ie3,ie4)+0.0001)  # Y方向边界标志
            h4: ti.f32 = h[ie4]  # 上侧网格水位
            h3: ti.f32 = h[ie3]  # 下侧网格水位
            z4: ti.f32 = ze[ie4] # 上侧网格底高程
            z3: ti.f32 = ze[ie3] # 下侧网格底高程
            dt2: ti.f32 =100     # Y方向时间步长初值
            hf_y:ti.f32=ti.max(h4,h3)-ti.max(z4,z3,h_dike[iss])  # Y方向有效水深
            dy:ti.f32=ti.max(ye[ie4]-ye[ie3],0.01)  # Y方向网格间距
            # 计算Y方向压力梯度项
            q3:ti.f32 = -g * ti.max(hf_y,0) * dt[None] * (hn[ie4] - hn[ie3]) / dy
            # 计算Y方向摩阻项（曼宁公式）
            q4:ti.f32 = 1 + g * dt[None] * (n ** 2) * ti.abs(qn_y[iss] / (ti.max(hf_y,0.00001) ** (7/3)))

            # 获取相邻边界索引用于数值扩散
            side_u: ti.i32 = isl4[ie4, 1]  # 上侧相邻边界
            side_d: ti.i32 = isl3[ie3, 1]  # 下侧相邻边界

            # 计算Y方向流量（包含时间权重和空间扩散）
            q_y[iss]=(sita*qn_y[iss]+(1-sita)/2*(qn_y[side_u]+qn_y[side_d]) +q3) / q4
            q_y[iss] = q_y[iss] * id_dy[iss]  # 应用边界控制标志
            # 应用干湿边界处理
            q_y[iss] = q_y[iss]* ti.max((hf_y - h_min) / (abs((hf_y - h_min)) + 0.00001), 0.)
            q_y[iss]=q_y[iss]*flag_y  # 应用边界方向标志
            qn_y[iss]=q_y[iss]  # 更新上一时刻Y方向流量
            # 计算Y方向CFL时间步长
            dt2= afa * dis[iss] / (ti.sqrt(g * ti.max(hf_y,0.01))
                                   + ti.abs(q_y[iss])/ti.max( hf_y,0.01))

            # 选择最小时间步长以满足CFL条件
            flag_max=ti.max(flag_x*dt1,flag_y*dt2) 
            dt3[iss]=ti.max((0.001-flag_max)*100000,flag_max)

    @ti.kernel
    @no_type_check
    def update_h():
        """更新网格水位（基于连续性方程）"""
        for i in h:
            # 初始化各方向流量和距离变量
            q1:ti.f32=0  # 左侧边界总流量
            q2:ti.f32=0  # 右侧边界总流量
            q3:ti.f32=0  # 下方边界总流量
            q4:ti.f32=0  # 上方边界总流量
            dis1:ti.f32=0  # X方向总距离
            dis3:ti.f32=0  # Y方向总距离
            a:ti.f32=0     # 源汇项流量

            # 计算左侧边界流入的总流量
            k1_num=nsl1[i]  # 左侧边界数量
            for k1 in range(1,k1_num+1): 
                k1_line=isl1[i,k1]  # 左侧边界索引
                q1+=q_x[k1_line]*dis[k1_line]  # 累加流量×距离
                dis1 += dis[k1_line]  # 累加X方向距离
                # print(dis1)

            # 计算右侧边界流出的总流量
            k2_num=nsl2[i]  # 右侧边界数量
            for k2 in range(1, k2_num+1):
                k2_line=isl2[i,k2]  # 右侧边界索引
                q2 += q_x[k2_line]*dis[k2_line]  # 累加流量×距离

            # 计算下方边界流入的总流量
            k3_num=nsl3[i]  # 下方边界数量
            for k3 in range(1,k3_num+1): 
                k3_line=isl3[i,k3]  # 下方边界索引
                q3 += q_y[k3_line]*dis[k3_line]  # 累加流量×距离
                dis3 += dis[k3_line]  # 累加Y方向距离

            # 计算上方边界流出的总流量
            k4_num=nsl4[i]  # 上方边界数量
            for k4 in range(1,k4_num+1): 
                k4_line=isl4[i,k4]  # 上方边界索引
                q4 += q_y[k4_line]*dis[k4_line]  # 累加流量×距离

            # 获取源汇项流量（如泵站、排水口等）
            a = q_source[i]
            # 基于连续性方程更新水位：水位变化 = (流入-流出+源汇项)×时间步长÷网格面积
            h[i] = hn[i] + ((q1-q2+q3-q4+a)*dt[None])/(0.0001+dis1 * dis3)
            # 添加降雨贡献
            h[i]+=rainq_t[None]*dt[None]
            
            # 计算不同下垫面类型的标志函数（用于选择对应的下渗率）
            # f1对应下垫面类型1（当under_suf[i]==1时，f1=1，其他为0）
            f1 = ti.min(ti.abs(
                (under_suf[i] - 2) * (under_suf[i] - 3) * (under_suf[i] - 4) * (under_suf[i] - 5) * (under_suf[i] - 6) * (under_suf[i] - 7)),
                        1)
            # f2对应下垫面类型2
            f2 = ti.min(ti.abs(
                (under_suf[i] - 1) * (under_suf[i] - 3) * (under_suf[i] - 4) * (under_suf[i] - 5) * (under_suf[i] - 6)* (under_suf[i] - 7)),
                        1)
            # f3对应下垫面类型3
            f3 = ti.min(ti.abs(
                (under_suf[i] - 2) * (under_suf[i] - 1) * (under_suf[i] - 4) * (under_suf[i] - 5) * (under_suf[i] - 6)* (under_suf[i] - 7)),
                        1)
            # f4对应下垫面类型4
            f4 = ti.min(ti.abs(
                (under_suf[i] - 2) * (under_suf[i] - 3) * (under_suf[i] - 1) * (under_suf[i] - 5) * (under_suf[i] - 6)* (under_suf[i] - 7)),
                        1)
            # f5对应下垫面类型5
            f5 = ti.min(ti.abs(
                (under_suf[i] - 2) * (under_suf[i] - 3) * (under_suf[i] - 4) * (under_suf[i] - 1) * (under_suf[i] - 6)* (under_suf[i] - 7)),
                        1)
            # f6对应下垫面类型6
            f6 = ti.min(ti.abs(
                (under_suf[i] - 2) * (under_suf[i] - 3) * (under_suf[i] - 4) * (under_suf[i] - 5) * (under_suf[i] - 1)* (under_suf[i] - 7)),
                        1)
            # f7对应下垫面类型7
            f7 = ti.min(ti.abs(
                (under_suf[i] - 2) * (under_suf[i] - 3) * (under_suf[i] - 4) * (under_suf[i] - 5)* (under_suf[i] - 6) * (under_suf[i] - 1)),
                        1)

            # 根据下垫面类型计算下渗损失
            h[i]-= (qf1[None]*f1+qf2[None]*f2+qf3[None]*f3+qf4[None]*f4+qf5[None]*f5+qf6[None]*f6+qf7[None]*f7) * dt[None]
            # 确保水位不低于地面高程
            h[i]=ti.max(h[i],ze[i])

            # 计算网格面积（X方向距离×Y方向距离）
            grid_area0[i]=dis1*dis3
            # 更新上一时刻水位
            hn[i]=h[i]

            # 计算水深（确保最小值为0.01m以避免数值问题）
            depth=ti.max(h[i]-ze[i],0.01)
            # 计算X方向平均流速：(流入+流出流量)÷距离÷2÷水深
            u[i]=(q1+q2)/ti.max(dis1,0.01)/2./depth
            # 计算Y方向平均流速：(流入+流出流量)÷距离÷2÷水深
            v[i]=(q3+q4)/ti.max(dis3,0.01)/2./depth

            # 干湿边界处理：当水深很小时，流速设为0
            symbol=ti.max(depth/(abs(depth)+0.001),0)
            u[i]=u[i]*symbol  # 应用干湿边界标志到X方向流速
            v[i]=v[i]*symbol  # 应用干湿边界标志到Y方向流速

    total_time[None]=total_time[None]+dt[None]
    timesteps[None]=timesteps[None]+dt[None]
    @ti.kernel
    @no_type_check
    def boundary_h():
        """应用边界条件（设置潮汐边界水位）"""
        nbd_ie_val: ti.i32 = nbd_ie[None]  # 获取边界网格总数
        for i in range(1,nbd_ie_val):
            ie=ibd_ie[i]  # 获取边界网格索引
            h[ie]=tide[None]  # 设置边界网格水位为当前潮汐水位
            hn[ie]=h[ie]  # 更新上一时刻边界网格水位


    @ti.kernel
    def choose_dtmin():
        """计算全局最小时间步长（CFL条件）"""
        dt[None]=1000  # 初始化时间步长为较大值
        for iss in range(1,ns+1):
               ti.atomic_min(dt[None], dt3[iss])  # 原子操作选择最小时间步长         

    def read_from_1d(timeout=300):
        """从一维管网模型读取数据
        
        Args:
            timeout: 等待超时时间（秒）
            
        Returns:
            dict: 一维模型返回的数据字典，包含各节点的水位和流量信息
            
        Raises:
            TimeoutError: 当等待一维模型数据超时时抛出
        """
        if shared['1d_ready'].wait(timeout=timeout):
            with shared['lock']:
                data = shared['1d_data']  # 获取一维模型计算结果
                shared['1d_ready'].clear()  # 清除就绪标志
                return data
        else:
            raise TimeoutError("等待1D数据超时")

    def send_to_1d(data_dict):
        with shared['lock']:
            shared['2d_data'].clear()
            shared['2d_data'].update(data_dict)
            shared['2d_ready'].set()
            shared['1d_ready'].clear()

    def initial_cpu():
        nonlocal tide1_time,tide2_time,tide1,tide2,begin_time,tide,tide_data
        nonlocal rainq,rain_data,rain1_num,rain2_num,rain_data
        step_index = max((int(time_index * 5/ tide_step) - 1),0)
        begin_time = tide_data[step_index][0]
        tide[None] = tide_data[step_index][1]

    tide_data = []
    for date_str, time_str, value in zip(tide_date, tide_time, tide_value):
        datetime_str = f"{date_str} {time_str}"
        datetime_obj = datetime.strptime(datetime_str, '%d/%m/%Y %H:%M:%S')
        tide_time = float(datetime_obj.timestamp())
        tide_level = value
        tide_data.append([tide_time, tide_level])
    tide_step = round((tide_data[1][0]-tide_data[0][0])/60) 

    rain_data = []
    for datetime_str, value in zip(rainfall_date, rainfall_value):
        datetime_obj = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M')
        rainq_time = float(datetime_obj.timestamp())
        rainq = float(value)
        rain_data.append([rainq_time, rainq])

####################################################
    start_time = time.time()
    initial_taichi()
    initial_cpu()
    data_dict = {}

    # connection_grid = [50, 50]

    tuopu_list_M = []
    with open(resource_path+'/'+"topo_H&M.txt", 'r', encoding='utf-8') as file:
        for line in file:
            elements = line.strip().split(',')
            elements = [int(element) for element in elements]
            tuopu_list_M.append(elements)

    node_num=[]
    with open(resource_path+'/'+'node_num_per_grid.txt', 'r') as file:
        for line in file:
            node_num.append(int(line.strip()))
    print('初始化已完成')

    while rain_data[rain1_num][0] < begin_time:
         rain1_num +=1

    rain_active=0    
    last_outputtime = 0
    total_rain_time = 0
    ze_np = ze.to_numpy()

    if time_index != 0:
        h_npy = np.zeros(ne + 1, dtype=np.float32)
        u_npy = np.zeros(ne + 1, dtype=np.float32)
        v_npy = np.zeros(ne + 1, dtype=np.float32)

        with open(f"{resource_path}/result_{time_index}.dat", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                parts = line.strip().split()
                h_npy[i + 1] = float(parts[1])
                u_npy[i + 1] = float(parts[2])
                v_npy[i + 1] = float(parts[3])
        print("已使用热启动文件")

        h.from_numpy(h_npy)
        hn.from_numpy(h_npy)
        u.from_numpy(u_npy)
        v.from_numpy(v_npy)

    # 主时间循环：模拟洪水演进过程
    last_output_count = 0
    last_output_time = begin_time
    while True:
        if last_output_count >= 144:    # 12 hours simulation
            break
        q_source_np = np.zeros(ne + 1, dtype=np.float32)  # 初始化源汇项数组
        nt=nt+1  # 增加时间步计数器

        # 计算当前绝对时间（模拟开始时间+运行时间）
        time2 = time_run + begin_time
        
        # 潮汐数据插值处理
        if time2 >= tide_data[tide_num1][0]:
            try:
                # 获取当前和下一个潮汐数据点
                tide1_time = tide_data[tide_num1][0]
                tide1 = tide_data[tide_num1][1]
                tide_num2 = tide_num1 + 1
                tide2_time = tide_data[tide_num2][0]
                tide2 = tide_data[tide_num1][1]
                # 线性插值计算当前时刻潮汐水位
                tide[None] = tide1 + (tide2 - tide1) / (tide2_time - tide1_time) * (
                        time2 - tide1_time)
            except StopIteration: 
                break  # 潮汐数据结束，退出循环
            except ValueError as e:
                print(f"出现空值：{e}")
                continue
        else:
            # 基于时间步长更新潮汐水位
            tide[None] = (tide2 - tide1) / (tide2_time - tide1_time) * dt[None] + tide[None]

        # 降雨数据处理和插值
        if rain2_num < len(rain_data) and time2 >= rain_data[rain1_num][0]:
            rain2_num =rain1_num+1
            if time2<=rain_data[rain2_num][0]: 
                # 计算降雨强度（转换为m/s单位）
                rainq=rain_data[rain2_num][1]/(rain_data[rain2_num][0]-rain_data[rain1_num][0])*0.001
            elif time2>rain_data[rain2_num][0]:
                rain1_num +=1  # 移动到下一个降雨数据点
                rain2_num=rain1_num+1
                if rain2_num>=len(rain_data): 
                    rainq=0  # 降雨数据结束
                else:
                     rainq=rain_data[rain1_num+1][1]/(rain_data[rain1_num+1][0]-rain_data[rain1_num][0])*0.001
        else: 
            rainq=0  # 无降雨
        rainq_t[None]=rainq  # 更新Taichi字段中的降雨强度
        
        # 设置降雨活跃标志（用于下渗计算）
        if rainq > 0:
            rain_active = 1
        else:
            rain_active = 0

        # 执行水动力计算的核心步骤
        boundary_h()     # 应用边界条件（设置潮汐边界水位）
        gate_control()   # 闸门控制（调整闸门高度）
        update_flow()    # 更新边界流量（基于浅水方程）
        
        # 在 update_h() 前添加自定义水量转移，当前时间步立即生效
        if watergroups:  # 如果有水量转移组
            for watergroup in watergroups:
                if watergroup.enabled:
                    source_grid = watergroup.source_grid
                    target_grid = watergroup.target_grid
                    transfer_flow = watergroup.transfer_flow
                    q_source_np = q_source.to_numpy()
                    q_source_np[source_grid] += -transfer_flow  # 源网格减水
                    q_source_np[target_grid] += transfer_flow   # 目标网格加水
                    copy_test_to_taichi1(q_source_np, q_source)
                    
        update_h()       # 更新网格水位（基于连续性方程）
        
        # 更新模拟时间
        time_run=dt[None]+time_run
        
        # 降雨期间的下渗参数动态更新（基于Horton下渗模型）
        if rain_active == 1: 
            total_rain_time=total_rain_time+dt[None]  # 累计降雨时间

            # 绿地/草地下渗率（类型3,5,7）：初始3英寸/小时，最终0.1英寸/小时
            qf3[None]= ((0.1 + (3 - 0.1) * math.exp(-2 * total_rain_time / 3600))
                         * 0.0254 / 3600)  
            qf5[None] = ((0.1 + (3 - 0.1) * math.exp(-2 * total_rain_time / 3600))
                         * 0.0254 / 3600)
            qf7[None] = ((0.1 + (3 - 0.1) * math.exp(-2 * total_rain_time / 3600))
                         * 0.0254 / 3600)
            # 土壤下渗率（类型1,2）：初始0.8英寸/小时，最终0.02英寸/小时
            qf1[None] = ((0.02 + (0.8 - 0.02) * math.exp(-10 * total_rain_time / 3600))
                         * 0.0254 / 3600) 
            qf2[None] = ((0.02 + (0.8 - 0.02) * math.exp(-10 * total_rain_time / 3600))
                         * 0.0254 / 3600)
            # 不透水面下渗率（类型4,6）：设为0
            qf4[None]=0
            qf6[None]=0

        choose_dtmin()   # 计算全局最小时间步长（满足CFL条件）
        
        # Output cumulative time every 5 minutes
        if time2 - last_output_time >= 300.0:
            last_output_time += 300.0
            print(f'Cumulative simulation time: {time_run} seconds, current dt: {dt[None]} seconds')
            
            last_output_count += 1
    print(f'Time profiling results: {time.time() - start_time} seconds')

    print(" OVER ")