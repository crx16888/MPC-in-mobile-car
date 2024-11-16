# opti.variable用于定义优化变量，这些值在优化过程中是可变的；
#opti.parameter用于定义参数，这些值在优化过程中是不变的

#在使用opti库求解mpc问题的过程中分为如下步骤：
# 1.定义系统动态模型，它描述了系统状态如何随时间变化。它通常是一个微分方程或差分方程，表示当前状和控制输入如何影响下一个状态。
#     # 单轨车模型的运动学方程
#     dx = v * ca.cos(yaw)
#     dy = v * ca.sin(yaw)
#     dyaw = v * ca.tan(delta) / (L_f + L_r)
#     dv = a
# 2.定义需要优化问题的优化变量和参数 
# opti.variable,opti.parameter
# 3.给优化变量的初始值、终态和参数赋值 
# opti.set_value
# 4.定义需要优化的函数，即代价函数，设置优化目标（即代价函数最小）
# 5.为优化问题添加约束条件
# 6.选择求解器并求解
# #总结：变量(包含初始值和终态)和参数，约束条件，需要优化的函数，要给它优化成什么样，采用什么方法优化
import numpy as np
import casadi as ca
# import matplotlib.pyplot as plt

class MPCCController:
    def __init__(self):
        # 定义车辆参数
        self.L_f = 1.0  # 前轴到质心的距离（m）
        self.L_r = 1.0  # 后轴到质心的距离（m）

        # 定义MPC参数
        self.N = 10  # 预测步数
        self.dt = 0.1  # 时间步长
        self.q_e = 1.0  # 横向误差的权重
        self.q_v = 1.0  # 速度误差的权重
        self.r_delta = 0.1  # 控制输入变化量的权重

        # 创建优化变量
        self.opti = ca.Opti()  # 创建优化问题对象
        self.X = self.opti.variable(self.N + 1, 4)  # 状态变量 (x, y, yaw, v) 的预测序列
        self.U = self.opti.variable(self.N, 2)  # 控制变量 (delta, a) 的预测序列
        self.X0 = self.opti.parameter(1, 4)  # 初始状态
        self.ref_path = self.opti.parameter(self.N + 1, 4)  # 参考路径的预测序列

        # 初始化代价函数
        self.cost = 0
        self.setup_optimization()

    def setup_optimization(self):
        # 迭代构建优化问题
        for k in range(self.N):
            # 代价函数：包括横向误差、速度误差和控制变化量
            state_error = self.ref_path[k,:] - self.X[k,:]
            self.cost += self.q_e * (state_error[0] ** 2 + state_error[1] ** 2) + self.q_v * (state_error[3]) ** 2
            if k < self.N - 1:
                self.cost += self.r_delta * (self.U[k + 1, 0] - self.U[k, 0]) ** 2 + self.r_delta * (self.U[k + 1, 1] - self.U[k, 1]) ** 2

            # 状态更新方程（离散化）
            next_state = self.X[k, :] + self.dt * self.vehicle_dynamics(self.X[k, :], self.U[k, :]) #切片实际上传的是一个一维数组，而1x4表示的是二维数组，4，才是一维数组
            self.opti.subject_to(self.X[k + 1, :] == next_state) #更新下一个时间步长的值

        # 设定边界条件
        self.opti.subject_to(self.X[0, :] == self.X0)  # 初始状态等于X0
        self.opti.subject_to(self.opti.bounded(-0.5, self.U[:, 0], 0.5))  # 转向角限制（弧度）
        self.opti.subject_to(self.opti.bounded(-3.0, self.U[:, 1], 3.0))  # 加速度限制

        # 设置代价函数
        self.opti.minimize(self.cost)

        # 选择求解器
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-3}
        self.opti.solver('ipopt', opts)

    def vehicle_dynamics(self, state, control):
        #传递进来的应该是某一时刻的所有状态和输入，返回状态的微分量；输入的是第一个最优控制序列 所有输入进来的量应该是一维数组
        x, y, yaw, v = state[0], state[1], state[2], state[3]
        delta, a = control[0], control[1]

        # 单轨车模型的运动学方程
        dx = v * ca.cos(yaw)
        dy = v * ca.sin(yaw)
        dyaw = v * ca.tan(delta) / (self.L_f + self.L_r)
        dv = a
        column_vector = ca.vertcat(dx, dy, dyaw, dv)
        row_vector = ca.transpose(column_vector) #转换为行向量
        return row_vector

    def mpcc_control(self, initial_state, reference_path):
        #输入变量是当前状态和参考轨迹，分别是1x4和11x4
        # 设置参数值
        self.opti.set_value(self.X0, initial_state)
        self.opti.set_value(self.ref_path, reference_path)

        # 求解优化问题
        sol = self.opti.solve()
        
        # 返回最优控制输入
        optimal_U = sol.value(self.U) #10x2，这是优化后的控制序列，后面我们只取其第一个序列
        optimal_X = sol.value(self.X) #11x4，这是求解优化后的状态信息，包含xk到xk+N，我们目的是为了使得X靠近Xr
        return optimal_U, optimal_X

# 示例：运行控制器
if __name__ == "__main__":
    mpc = MPCCController()  # 创建控制器实例
    current_state = np.zeros((1, 4)) #创建了一个1x4的二维数组
    current_state_1dim = current_state.reshape(-1)
    # current_state_1dim = np.zero(4)
    # current_state = current_state.reshape(1, -1)
    # 定义总的参考轨迹
    total_steps = 20  # 总仿真步数
    full_trajectory = np.array([[i*10, i*10, 0, 1] for i in range(total_steps)])
    # 存储历史数据
    state_history = [current_state.copy()]
    control_history = []   
    # 初始化控制序列
    u_sequence = np.zeros((mpc.N, 2))  # 存储完整的N步控制序列
    # 模拟total_steps个时间步
    for step in range(total_steps):
        # print(f"\n步骤 {step + 1}/{total_steps}")
        # print(f"当前状态: x = {current_state[0]:.2f}, y = {current_state[1]:.2f}, yaw = {current_state[2]:.2f}, v = {current_state[3]:.2f}")    
        # 获取当前时刻开始的N+1步参考轨迹
        ref_trajectory = full_trajectory[step:step + mpc.N + 1]
        if len(ref_trajectory) < mpc.N + 1:
            last_point = full_trajectory[-1] 
            ref_trajectory = np.vstack([ref_trajectory] + [last_point] * (mpc.N + 1 - len(ref_trajectory)))  # 补足到 N + 1 个点
        
        # 使用上一步的控制序列平移作为初值（去掉第一个，末尾补零） 上一步的控制序列上一步已经用掉了，这次需要用新的控制序列作为输入
        u_init = np.vstack([u_sequence[1:], np.zeros((1, 2))]) #始终让每一轮需要提取的控制量位于第一行，便于optimal_control = u_sequence[0]取出
        
        # 求解MPC获取控制序列
        optimal_U, optimal_X = mpc.mpcc_control(current_state, ref_trajectory)
        # print(f"控制输入: 转向角 = {optimal_U[0]:.2f} m/s, 加速度 = {optimal_U[1]:.2f} rad")
        # print(f"当前参考点: x = {ref_trajectory[0][0]:.2f}, y = {ref_trajectory[0][1]:.2f}, yaw = {ref_trajectory[0][2]:.2f}, v = {ref_trajectory[0][3]:.2f}")
        u_sequence = optimal_U
        # # 打印未来N步的控制序列
        # print("\n预测的控制序列:")
        # for i, u in enumerate(u_sequence):
        #     print(f"预测步骤 {i+1}: 转向角 = {u[0]:.2f} m/s, 加速度 = {u[1]:.2f} rad")              
        optimal_U0 = optimal_U[0, :] #取第一个控制序列
        # 使用运动学模型更新状态
        current_state = current_state + mpc.dt * mpc.vehicle_dynamics(current_state_1dim, optimal_U0) #current_state_1dim这才是传递一个一维数组进去
        current_state_1dim = current_state.T #current_state是1x4的二维数组，应该将其转化为1维数组
        state_history.append(current_state.copy())        

    # reference_path = np.ones((4, 11))  # 假设参考路径为...，参考路径在整个代码中无改变
    # control_action, X_value = controller.mpcc_control(initial_state, reference_path)  # 调用控制方法
    # print(control_action)
    # print(X_value)

#需要反思的地方：
# 1.切片实际上传的是一个一维数组，而1x4表示的是二维数组，4，才是一维数组 对二维数组X，X[0]访问的是第一行的所有元素，所以State[0]不能精确到一个值，而对切片得到的一维可以
# 2.使用casadi库就不要用numpy库，二者转换数据维度等处理数据的的方法都不一样 我们要基于同一种库创建数据，然后在这个库的基础上去进行处理
# 3.casadi库并不只是一个简单的管理数据的库，它集成了无人驾驶领域的全部，其还有各种针对这些数据在无人驾驶领域做优化的方法等   所以不像numpy创建数据处理数据再调比如scipy.optimize库优化计算数据
# 4.后面考虑只写一个含有casadi库的代码 因为：用ca库创建数据然后处理数据调用ca库的优化方法计算数据，效率值比基于numpy库调用scipy.optimize方法要快很多