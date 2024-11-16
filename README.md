本仓库包含MPC、MPCC代码和待测试的test代码,MPC代码仍在更新中

本仓库旨在探索不同的MPC系列代码跟踪车辆轨迹的能力,探索MPC算法中哪些量会对算法性能有较大的影响

MPCC代码相较于MPC代码在X、U和J的构造上均有差异,此外也可以采取不同的求解器。当前我们旨在探索dt和N对于MPC算法性能的影响:state = state + self.dt * self.vehicle_dynamics(state, u[i]),dt会强影响state更新的斜率