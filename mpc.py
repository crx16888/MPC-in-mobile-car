import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
class VehicleMPC:
    """
    车辆模型预测控制器类
    实现了基于简化运动学模型的MPC控制算法
    """
    def __init__(self):
        # MPC 控制器参数
        self.dt = 0.1  # 时间步长(秒)
        self.N = 10    # 预测时域长度(步数)
        
        # 车辆物理参数
        self.L = 2.7   # 轴距(米)
        
        # 状态和输入约束
        self.v_max = 20.0  # 最大速度(米/秒)
        self.v_min = 0.0   # 最小速度(米/秒)
        self.theta_max = np.pi/4  # 最大转向角(弧度)
        self.theta_min = -np.pi/4  # 最小转向角(弧度)
        
        # 代价函数权重矩阵
        self.Q = np.diag([1.0, 1.0, 0.5])  # 状态误差权重矩阵 [x, y, delta]
        self.R = np.diag([0.1, 0.1])       # 控制输入权重矩阵 [v, theta]
        
    def vehicle_dynamics(self, state, control):
        """
        车辆运动学模型
        参数:
            state: [x, y, delta] - 位置x, 位置y, 航向角delta
            control: [v, theta] - 速度v, 转向角theta
        返回:
            状态导数 [dx/dt, dy/dt, d(delta)/dt]
        """
        x, y, delta = state
        v, theta = control
        
        # 运动学方程
        dx = v * np.cos(delta)  # x方向速度
        dy = v * np.sin(delta)  # y方向速度
        ddelta = v * np.tan(theta) / self.L  # 航向角变化率,角速度=垂直于车辆方向的速度/轴距
        
        return np.array([dx, dy, ddelta]) # 返回x方向速度,y方向速度,角速度
        
    def objective(self, u, current_state, ref_trajectory):
        """
        MPC优化目标函数
        参数:
            u: 待优化的控制序列
            current_state: 当前状态
            ref_trajectory: 参考轨迹
        返回:
            总代价值
        """
        total_cost = 0
        state = current_state
        
        # 将控制输入重塑为Nx2矩阵
        u = u.reshape(-1, 2)
        
        for i in range(self.N): #这里只是将预测的状态序列和参考的状态序列做差算J，优化预测的状态序列还在后面调优化器实现：传入控制序列、当前状态和参考预测状态序列
            # 计算状态误差代价
            state_error = state - ref_trajectory[i] #当前时刻的状态-当前时刻的参考状态，优化下一个时刻的状态，只能让下一个时刻的状态和当前时刻的参考状态靠近，然而下一个时刻的状态是线性时变的
            total_cost += state_error.T @ self.Q @ state_error #对应状态误差代价，即(x_k+1-x_ref)^T*Q*(x_k+1-x_ref) 
            
            # 计算控制输入代价
            total_cost += u[i].T @ self.R @ u[i] #对应控制输入代价，即u_k^T*R*u_k   
            
            # 使用运动学模型预测下一状态
            state = state + self.dt * self.vehicle_dynamics(state, u[i]) # 状态更新，即x_k+1 = x_k + [dx, dy, ddelta] * dt    
            
        return total_cost
        
    def solve_mpc(self, current_state, ref_trajectory):
        """
        求解MPC优化问题
        参数:
            current_state: 当前状态
            ref_trajectory: 参考轨迹
        返回:
            最优控制输入的第一组 [v, theta]
        """
        # 控制序列初值
        u0 = np.zeros(2 * self.N)
        
        # 设置控制输入约束
        bounds = [(self.v_min, self.v_max), (self.theta_min, self.theta_max)] * self.N
        
        # 使用SLSQP算法求解优化问题
        result = minimize(
            self.objective,
            u0,
            args=(current_state, ref_trajectory),
            method='SLSQP',
            bounds=bounds
        )
        
        # 返回最优控制序列的第一组控制量
        return result.x.reshape(-1, 2)[0]
    
if __name__ == "__main__":
    mpc = VehicleMPC()
    
    # 初始状态
    current_state = np.array([0, 0, 0])
    
    # 定义总的参考轨迹
    total_steps = 20  # 总仿真步数
    full_trajectory = np.array([[i*10, i*10, 0] for i in range(total_steps)])
    # 存储历史数据
    state_history = [current_state.copy()]
    control_history = []
    
    # 初始化控制序列
    u_sequence = np.zeros((mpc.N, 2))  # 存储完整的N步控制序列
    
    # 模拟total_steps个时间步
    for step in range(total_steps):
        print(f"\n步骤 {step + 1}/{total_steps}")
        print(f"当前状态: x = {current_state[0]:.2f}, y = {current_state[1]:.2f}, delta = {current_state[2]:.2f}")
        
        # 获取当前时刻开始的N步参考轨迹
        ref_trajectory = full_trajectory[step:step + mpc.N] #我觉得这里应该是N+1，未来的预测应该多一个
        if len(ref_trajectory) < mpc.N:
            last_point = full_trajectory[-1] 
            ref_trajectory = np.vstack([ref_trajectory] + [last_point] * (mpc.N - len(ref_trajectory))) #如果ref_trajectory不足10个向量，就让后面都以最后一行的值自动补足
        
        # 使用上一步的控制序列平移作为初值（去掉第一个，末尾补零） 上一步的控制序列上一步已经用掉了，这次需要用新的控制序列作为输入
        u_init = np.vstack([u_sequence[1:], np.zeros((1, 2))]) #始终让每一轮需要提取的控制量位于第一行，便于optimal_control = u_sequence[0]取出
        
        # 求解MPC获取控制序列
        result = minimize(
            mpc.objective,
            u_init.flatten(),  # 展平作为初始值
            args=(current_state, ref_trajectory),
            method='SLSQP',
            bounds=[(mpc.v_min, mpc.v_max), (mpc.theta_min, mpc.theta_max)] * mpc.N
        ) #给点要优化的函数，要得到的量的初始值、其他参数（这些都会传入到优化函数里面），优化算法，约束条件
        
        # 更新完整控制序列
        u_sequence = result.x.reshape(-1, 2)
        
        # 取出当前时刻的控制输入
        optimal_control = u_sequence[0] 
        control_history.append(optimal_control)
        
        print(f"控制输入: 速度 = {optimal_control[0]:.2f} m/s, 转向角 = {optimal_control[1]:.2f} rad")
        print(f"当前参考点: x = {ref_trajectory[0][0]:.2f}, y = {ref_trajectory[0][1]:.2f}")
        
        # 打印未来N步的控制序列
        print("\n预测的控制序列:")
        for i, u in enumerate(u_sequence):
            print(f"预测步骤 {i+1}: 速度 = {u[0]:.2f} m/s, 转向角 = {u[1]:.2f} rad")
        
        # 使用运动学模型更新状态
        current_state = current_state + mpc.dt * mpc.vehicle_dynamics(current_state, optimal_control) #这里是真的要优化下一个状态了
        state_history.append(current_state.copy())

    # 打印最终结果
    print("\n模拟完成!")
    print(f"最终状态: x = {current_state[0]:.2f}, y = {current_state[1]:.2f}, delta = {current_state[2]:.2f}")

    # 可视化结果
    state_history = np.array(state_history)
    control_history = np.array(control_history)
    
    plt.figure(figsize=(15, 5))
    
    # 轨迹图
    plt.subplot(121)
    plt.plot(full_trajectory[:, 0], full_trajectory[:, 1], 'r--', label='参考轨迹')
    plt.plot(state_history[:, 0], state_history[:, 1], 'b-', label='实际轨迹')
    plt.scatter(state_history[0, 0], state_history[0, 1], c='g', marker='o', label='起点')
    plt.scatter(state_history[-1, 0], state_history[-1, 1], c='r', marker='o', label='终点')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.title('车辆轨迹跟踪结果')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    # 控制输入历史
    plt.subplot(122)
    time_steps = np.arange(len(control_history)) * mpc.dt
    plt.plot(time_steps, control_history[:, 0], 'b-', label='速度')
    plt.plot(time_steps, control_history[:, 1], 'r-', label='转向角')
    plt.grid(True)
    plt.legend()
    plt.title('控制输入历史')
    plt.xlabel('时间 (s)')
    plt.ylabel('控制量')
    
    plt.tight_layout()
    plt.show()

