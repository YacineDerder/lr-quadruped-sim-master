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


#best results obtained with cartesian only!!
PD_CARTESIAN_ONLY = True #if this is true the next one has to be true as well
ADD_CARTESIAN_PD = True #change both of these variables to have PD only in the joint space
BASIC_TRACKING_PLOT = False
CPG_STATES_PLOT = False
EXT_TRACKING_PLOT = True
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
                    record_video=False #didnt work for me, you have to install ffmpeg
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states

xs_cpg = np.zeros((TEST_STEPS,4))
ys_cpg = np.zeros((TEST_STEPS,4))
zs_cpg = np.zeros((TEST_STEPS,4))
xs_robot = np.zeros((TEST_STEPS,4))
ys_robot = np.zeros(TEST_STEPS)
zs_robot = np.zeros((TEST_STEPS,4))
distance_run = np.zeros(3)
robot_v = np.zeros((TEST_STEPS,3))
robot_p = np.zeros(TEST_STEPS)

# First task, data structure initialization
r_list = np.zeros([TEST_STEPS,4])
theta_list = np.zeros([TEST_STEPS,4])
dr_list = np.zeros([TEST_STEPS,4])
dtheta_list = np.zeros([TEST_STEPS,4])

# Second task, data structure initialization
pos_leg_0 = np.zeros([TEST_STEPS,3])
des_pos_leg_0 = np.zeros([TEST_STEPS,3])
joint_leg_3 = np.zeros([TEST_STEPS,3])
des_joint_leg_3 = np.zeros([TEST_STEPS,3])


############## Sample Gains
# joint PD gains
kp=1*np.array([100,100,100])
kd=0.5*np.array([2,2,2])
if not PD_CARTESIAN_ONLY and not ADD_CARTESIAN_PD : #same as if "PD_JOINT_ONLY"
  kp = 10*np.array([100,100,100])
  kd = 5 * np.array([2, 2, 2])
# Cartesian PD gains
kpCartesian = 5* np.diag([500]*3)
kdCartesian = 1* np.diag([20]*3)
if PD_CARTESIAN_ONLY:
  kpCartesian = 5 * np.diag([500] * 3)
  kdCartesian = 2 * np.diag([20] * 3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 
  # get desired foot positions from CPG 
  xs,zs = cpg.update()

  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()
  robot_v[j,:] = env.robot.GetBaseLinearVelocity()

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
    if PD_CARTESIAN_ONLY:
      tau = np.zeros(3)

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
  #we take the value for the joints for the last leg, i = 3 out of the loop
  joint_leg_3[j,:] = q_i
  des_joint_leg_3[j,:] = leg_q

  raw_power = np.multiply(env.robot.GetMotorVelocities(), env.robot.GetMotorTorques())
  robot_p[j] =np.sum(np.maximum(raw_power, np.zeros(12))) #we also assume here that the robot cannot use regenerative braking

  xs_cpg[j,:] = xs
  zs_cpg[j,:] = zs
  if not ADD_CARTESIAN_PD:
    J, pos = env.robot.ComputeJacobianAndPosition(i)
  xs_robot[j,:] = pos[0]
  zs_robot[j,:] = pos[2]
  ys_cpg[j,:] = pos[1]
  ys_robot[j] = leg_xyz[1]
  # [TODO] save any CPG or robot states
  r_list[j,:] = cpg.get_r()
  theta_list[j,:] = cpg.get_theta()
  dr_list[j,:] = cpg.get_dr()
  dtheta_list[j,:] = cpg.get_dtheta()

distance_run = np.array(env.robot.GetBasePosition())-np.array(env.robot._GetDefaultInitPosition())
print(distance_run)

##################################################### 
# PLOTS
#####################################################

short_t = int(np.ceil(TEST_STEPS/25))
#print(short_t)

if CPG_STATES_PLOT:
  fig, axis = plt.subplots(4,4)
  axis[0,0].plot(range(short_t),r_list[:short_t,0])
  axis[0,0].set_title("r0")
  axis[0,1].plot(range(short_t),r_list[:short_t,1])
  axis[0,1].set_title("r1")
  axis[0,2].plot(range(short_t),r_list[:short_t,2])
  axis[0,2].set_title("r2")
  axis[0,3].plot(range(short_t),r_list[:short_t,3])
  axis[0,3].set_title("r3")

  axis[1,0].plot(range(short_t),dr_list[:short_t,0])
  axis[1,0].set_title("dr0")
  axis[1,1].plot(range(short_t),dr_list[:short_t,1])
  axis[1,1].set_title("dr1")
  axis[1,2].plot(range(short_t),dr_list[:short_t,2])
  axis[1,2].set_title("dr2")
  axis[1,3].plot(range(short_t),dr_list[:short_t,3])
  axis[1,3].set_title("dr3")

  axis[2,0].plot(range(short_t),theta_list[:short_t,0])
  axis[2,0].set_title("theta0")
  axis[2,1].plot(range(short_t),theta_list[:short_t,1])
  axis[2,1].set_title("theta1")
  axis[2,2].plot(range(short_t),theta_list[:short_t,2])
  axis[2,2].set_title("theta2")
  axis[2,3].plot(range(short_t),theta_list[:short_t,3])
  axis[2,3].set_title("theta3")

  axis[3,0].plot(range(short_t),dtheta_list[:short_t,0])
  axis[3,0].set_title("dtheta0")
  axis[3,1].plot(range(short_t),dtheta_list[:short_t,1])
  axis[3,1].set_title("dtheta1")
  axis[3,2].plot(range(short_t),dtheta_list[:short_t,2])
  axis[3,2].set_title("dtheta2")
  axis[3,3].plot(range(short_t),dtheta_list[:short_t,3])
  axis[3,3].set_title("dtheta3")
  fig.tight_layout(pad=2.0)
  plt.show()


if EXT_TRACKING_PLOT:
  #print(np.size(xs_cpg))
  fig, axis = plt.subplots(1,3)
  axis[0].plot(range(short_t*3),xs_cpg[:short_t*3,3],c="red")
  axis[0].plot(range(short_t*3), xs_robot[:short_t*3, 1], c="blue")
  axis[0].set_title("x")
  axis[1].plot(range(short_t * 3), ys_cpg[:short_t * 3,3], c="red")
  axis[1].plot(range(short_t * 3), ys_robot[:short_t * 3], c="blue")
  axis[1].set_ylim([-0.25,0.25])
  axis[1].set_title("y")
  axis[2].plot(range(short_t*3), zs_cpg[:short_t*3,3], c="red")
  axis[2].plot(range(short_t*3), zs_robot[:short_t*3, 1], c="blue")
  axis[2].set_title("z")
  #if not PD_CARTESIAN_ONLY and not ADD_CARTESIAN_PD:#same as if "PD_JOINT_ONLY"
  #  plt.title("tracking cartesian command for leg 4, joint PD only")
  #if PD_CARTESIAN_ONLY:
  #  plt.title("tracking cartesian command for leg 4, cartesian PD only")
  plt.show()

  fig, axis = plt.subplots(1,3)
  axis[0].plot(range(short_t*3), des_joint_leg_3[:short_t*3, 0], c="red")
  axis[0].plot(range(short_t*3), joint_leg_3[:short_t*3, 0], c="blue")
  axis[0].set_ylim([-1.5,1.5])
  axis[0].set_title("hip angle")
  axis[1].plot(range(short_t*3), des_joint_leg_3[:short_t*3, 1], c="red")
  axis[1].plot(range(short_t*3), joint_leg_3[:short_t*3, 1], c="blue")
  axis[1].set_title("thigh angle")
  axis[2].plot(range(short_t*3), des_joint_leg_3[:short_t*3, 2], c="red")
  axis[2].plot(range(short_t*3), joint_leg_3[:short_t*3, 2], c="blue")
  axis[2].set_title("calf angle")
  #if not PD_CARTESIAN_ONLY and not ADD_CARTESIAN_PD:
  #  plt.title("tracking joint command for leg 4, joint PD only")
  #if PD_CARTESIAN_ONLY:
  #  plt.title("tracking joint command for leg 4, cartesian PD only")
  plt.show()


# plot the speed
fig1 = plt.figure()
plt.plot(range(TEST_STEPS),np.sqrt(robot_v[:,0]**2+robot_v[:,1]**2))
plt.show()
average_speed = np.sum(np.sqrt(robot_v[:,0]**2+robot_v[:,1]**2))/TEST_STEPS
average_power = np.sum(robot_p)/TEST_STEPS
print("average speed: ")
print(average_speed)
print("average power: ")
print(average_power)
mass = np.sum(env.robot.GetTotalMassFromURDF())
CoT = average_power/(mass*9.81*average_speed)
print("CoT: ")
print(CoT)

if BASIC_TRACKING_PLOT:
  #print(np.size(xs_cpg))
  fig2 = plt.figure()
  plt.plot(range(short_t*3),xs_cpg[:short_t*3,3],c="red")
  plt.plot(range(short_t*3),xs_robot[:short_t*3,1],c="blue")
  plt.title("tracking x cartesian command for leg 4")
  plt.show()

  fig3 = plt.figure()
  plt.plot(range(short_t*3),zs_cpg[:short_t*3],c="red")
  plt.plot(range(short_t*3),zs_robot[:short_t*3,1],c="blue")
  plt.title("tracking z cartesian command for leg 4")
  plt.show()