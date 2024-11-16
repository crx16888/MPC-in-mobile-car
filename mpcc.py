import numpy as np
import casadi as ca

# 定义车辆参数
L_f = 1.0  # 前轴到质心的距离（m）
L_r = 1.0  # 后轴到质心的距离（m）

# 定义MPC参数
N = 10  # 预测步数
dt = 0.1  # 时间步长
q_e = 1.0  # 横向误差的权重
q_v = 1.0  # 速度误差的权重
r_delta = 0.1  # 控制输入变化量的权重

# 创建优化变量
x = ca.MX.sym('x')   # 位置x
y = ca.MX.sym('y')   # 位置y
yaw = ca.MX.sym('yaw')  # 航向角
v = ca.MX.sym('v')   # 速度
state = ca.vertcat(x, y, yaw, v)

delta = ca.MX.sym('delta')  # 前轮转角
a = ca.MX.sym('a')  # 加速度
control = ca.vertcat(delta, a)

# 定义车辆模型，基于当前状态和控制量返回变化状态量
def vehicle_dynamics(state, control):
    x, y, yaw, v = state[0], state[1], state[2], state[3]
    delta, a = control[0], control[1]

    # 单轨车模型的运动学方程
    dx = v * ca.cos(yaw)
    dy = v * ca.sin(yaw)
    dyaw = v * ca.tan(delta) / (L_f + L_r)
    dv = a

    return ca.vertcat(dx, dy, dyaw, dv)

# 设置优化器
opti = ca.Opti()  # 创建优化问题对象

# 定义优化变量和参数
X = opti.variable(4, N + 1)  # 状态变量 (x, y, yaw, v) 的预测序列
U = opti.variable(2, N)  # 控制变量 (delta, a) 的预测序列

# 初始状态和目标轨迹为参数
X0 = opti.parameter(4)  # 初始状态
ref_path = opti.parameter(4, N + 1)  # 参考路径的预测序列

# 初始化代价函数
cost = 0

# 迭代构建优化问题，每一轮都更新第k时刻的cost和状态
for k in range(N):
    # 代价函数：包括横向误差、速度误差和控制变化量
    state_error = ref_path[:, k] - X[:, k]
    cost += q_e * (state_error[0] ** 2 + state_error[1] ** 2) + q_v * (state_error[3] - X[3, k]) ** 2
    if k < N - 1:
        cost += r_delta * (U[0, k + 1] - U[0, k]) ** 2 + r_delta * (U[1, k + 1] - U[1, k]) ** 2

    # 状态更新方程（离散化）
    next_state = X[:, k] + dt * vehicle_dynamics(X[:, k], U[:, k])
    opti.subject_to(X[:, k + 1] == next_state)

# 设定边界条件
opti.subject_to(X[:, 0] == X0)  # 初始状态等于X0
opti.subject_to(opti.bounded(-0.5, U[0, :], 0.5))  # 转向角限制（弧度）
opti.subject_to(opti.bounded(-3.0, U[1, :], 3.0))  # 加速度限制

# 设置代价函数
opti.minimize(cost)

# 选择求解器
opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-3}
opti.solver('ipopt', opts)

# MPCC控制函数 需要不断地去调控制函数去求解
def mpcc_control(initial_state, reference_path):
    # 设置参数值
    opti.set_value(X0, initial_state)
    opti.set_value(ref_path, reference_path)

    # 求解优化问题
    sol = opti.solve() #这是一个自动驾驶领域解决MPC问题专用的求解器。在此处传入了初始状态和终态，前面调opti给了变量、目标函数，约束条件然后就会自动开始求解。
    
    # 返回最优控制输入
    optimal_U = sol.value(U[:, 0])
    optimal_X = sol.value(X)
    return optimal_U,optimal_X

# 示例：运行控制器
initial_state = np.array([0, 0, 0, 1])  # 初始状态：x=0, y=0, yaw=0, v=1
reference_path = np.ones((4, N + 1))  # 假设参考路径为...，参考路径在整个代码中无改变

# 填入期望的参考路径数据
# 可以使用适当的轨迹生成函数来设定 reference_path，示例中简化为零轨迹
for i in range(N):control_action,X_value = mpcc_control(initial_state, reference_path)

