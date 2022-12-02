# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

""" Run CPG """
import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv


ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states

xs_cpg = np.zeros((TEST_STEPS,4))
zs_cpg = np.zeros((TEST_STEPS,4))

# First task, data structure initialization
r_list = np.zeros([TEST_STEPS,4])
theta_list = np.zeros([TEST_STEPS,4])
dr_list = np.zeros([TEST_STEPS,4])
dtheta_list = np.zeros([TEST_STEPS,4])

# Second task, data structure initialization
pos_leg_0 = np.zeros([TEST_STEPS,3])
des_pos_leg_0 = np.zeros([TEST_STEPS,3])


############## Sample Gains
# joint PD gains
kp=1*np.array([100,100,100])
kd=1*np.array([2,2,2])
# Cartesian PD gains
kpCartesian = 1* np.diag([500]*3)
kdCartesian = 1* np.diag([20]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 
  # get desired foot positions from CPG 
  xs,zs = cpg.update()

  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)
    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz)
    # Add joint PD contribution to tau for leg i
    q_i = q[3*i:3*(i+1)]
    dq_i = dq[3*i:3*(i+1)]
    tau += kp*(leg_q-q_i) + kd*(-dq_i)

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      J,pos = env.robot.ComputeJacobianAndPosition(i)
      # Get current foot velocity in leg frame (Equation 2)
      v = np.matmul(J,dq_i)
      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += np.matmul(np.transpose(J),(np.matmul(kpCartesian,(leg_xyz-pos))+np.matmul(kdCartesian,-v)))

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action)

  xs_cpg[j,:] = xs
  zs_cpg[j,:] = zs
  # [TODO] save any CPG or robot states
  r_list[j,:] = cpg.get_r()
  theta_list[j,:] = cpg.get_theta()
  dr_list[j,:] = cpg.get_dr()
  dtheta_list[j,:] = cpg.get_dtheta()



##################################################### 
# PLOTS
#####################################################
# example
#fig = plt.figure()
#plt.plot(t,joint_pos[1,:], label='FR thigh')
#plt.legend()
#plt.show()
fig2 = plt.figure()
plt.plot(range(TEST_STEPS),xs_cpg[:,1])
plt.legend("x range for FR foot")
plt.show()

fig3 = plt.figure()
plt.plot(range(TEST_STEPS),zs_cpg[:,1])
plt.legend("z range for FR foot")
plt.show()