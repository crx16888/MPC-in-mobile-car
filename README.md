mpc.py is for mpc code in mobile car,the code is still under development

当前存在的问题：在MPC算法中，状态误差=当前时刻的状态-当前时刻的参考状态，然后通过最小化代价函数的方式得到最优控制，从而优化下一个时刻的状态；然而这一系列过程都只是让下一个时刻的状态和当前时刻的参考状态靠近，如果参考状态的值是线性时变的，如参考状态ref_trajectory = 10t，那么我们优化得到的下一个时刻的状态仅仅只是和当前参考状态接近而不是和下一个参考状态接近。

mpcc.py is for mpcc

test is error code
