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

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"   #ADDED LINE TO FIX A RBUG AT RUNNING
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "SAC"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '121321105810'
log_dir = interm_dir + '120222153155'     # SAC 0.5m/s CPG
# log_dir = interm_dir + '120322105510'     # SAC 0.8m/s CPG
# log_dir = interm_dir + '120522142306'      # SAC 1.5m/s CPG
# log_dir = interm_dir + '120622144757'      # SAC 0.5m/s PD 
# log_dir = interm_dir + '121022162956'      # SAC 0.5m/s CARTESIAN_PD
# log_dir = interm_dir + '121422221342'      # SAC 0.5m/s CARTESIAN_PD special reward 1
# log_dir = interm_dir + '121522113448'      # SAC 0.5m/s CARTESIAN_PD special reward 2
# log_dir = interm_dir + '122422172715'     # SAC 0.5m/s CPG special reward

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 
env_config['motor_control_mode'] = "CPG" 
# env_config['motor_control_mode'] = "PD" 
# env_config['motor_control_mode'] = "CARTESIAN_PD"
env_config['observation_space_mode'] = "LR_COURSE_OBS"
env_config['task_env'] =  "LR_COURSE_TASK"
# env_config['competition_env'] = True

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation 
#

compute_CoT = True
save_data = False
des_speed = "_05ms"  # For figure naming
# des_speed = "_08ms"
# des_speed = "_15ms"

if compute_CoT:
    power_hist = np.zeros(1000)
    velocity_hist = np.zeros((1000,3))

if save_data:
    if (env_config['motor_control_mode'] == "CPG"):
        CPG_r_history_FL = np.zeros(1000)
        CPG_r_history_FR = np.zeros(1000)
        CPG_r_history_RR = np.zeros(1000)
        CPG_r_history_RL = np.zeros(1000)
        CPG_theta_history_FL = np.zeros(1000)
        CPG_theta_history_FR = np.zeros(1000)
        CPG_theta_history_RR = np.zeros(1000)
        CPG_theta_history_RL = np.zeros(1000)
    base_x_pos_hist = np.zeros(1000)
    base_y_pos_hist = np.zeros(1000)
    base_z_pos_hist = np.zeros(1000)
    FL_foot_hist = np.zeros(1000)
    FR_foot_hist = np.zeros(1000)
    RR_foot_hist = np.zeros(1000)
    RL_foot_hist = np.zeros(1000)


for i in range(1000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    #

    if compute_CoT:
        if i < 1000:
            raw_power = np.multiply(env.envs[0].env.robot.GetMotorVelocities(), env.envs[0].env.robot.GetMotorTorques())
            power_hist[i] = np.sum(np.maximum(raw_power, np.zeros(12))) #we also assume here that the robot cannot use regenerative braking
            velocity_hist[i] = env.envs[0].env.robot.GetBaseLinearVelocity()

    if save_data:
        if i < 1000: # Only first run 
            base_x_pos_hist[i] = env.envs[0].env.robot.GetBasePosition()[0]
            base_y_pos_hist[i] = env.envs[0].env.robot.GetBasePosition()[1]
            base_z_pos_hist[i] = env.envs[0].env.robot.GetBasePosition()[2]

            _, _, _, feetInContactBool = env.envs[0].env.robot.GetContactInfo()
            FL_foot_hist[i] = int(feetInContactBool[0])
            FR_foot_hist[i] = int(feetInContactBool[1])
            RR_foot_hist[i] = int(feetInContactBool[2])
            RL_foot_hist[i] = int(feetInContactBool[3])

            if (env_config['motor_control_mode'] == "CPG"):
                CPG_r_history_FL[i] = env.envs[0].env._cpg.get_r()[0]
                CPG_theta_history_FL[i] = env.envs[0].env._cpg.get_theta()[0]
                CPG_r_history_FR[i] = env.envs[0].env._cpg.get_r()[1]
                CPG_theta_history_FR[i] = env.envs[0].env._cpg.get_theta()[1]
                CPG_r_history_RR[i] = env.envs[0].env._cpg.get_r()[2]
                CPG_theta_history_RR[i] = env.envs[0].env._cpg.get_theta()[2]
                CPG_r_history_RL[i] = env.envs[0].env._cpg.get_r()[3]
                CPG_theta_history_RL[i] = env.envs[0].env._cpg.get_theta()[3]
    
# [TODO] make plots:
if compute_CoT:
    average_speed = np.sum(np.sqrt(velocity_hist[:,0]**2+velocity_hist[:,1]**2))/1000
    average_power = np.sum(power_hist)/1000
    print("average speed: ")
    print(average_speed)
    print("average power: ")
    print(average_power)
    mass = np.sum(env.envs[0].env.robot.GetTotalMassFromURDF())
    CoT = average_power/(mass*9.81*average_speed)
    print("CoT: ")
    print(CoT)

if save_data:
    print("Saving plots")
    x_axis = np.arange(1000)

    f1 = plt.figure()
    plt.plot(base_x_pos_hist, base_y_pos_hist)
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.xlabel('x position')
    plt.ylabel('y position')
    fig_name = env_config['motor_control_mode'] + des_speed + "_xy_Pos.png"
    plt.savefig(fig_name)

    f2 = plt.figure()
    # plt.plot(x_axis, FL_foot_hist, label='FL')
    # plt.plot(x_axis, FR_foot_hist, label='FR')
    plt.plot(x_axis, RR_foot_hist, label='RR')
    plt.plot(x_axis, RL_foot_hist, label='RL')
    plt.xlabel('timesteps')
    plt.ylabel('Foot contact boolean')
    plt.legend()
    fig_name = env_config['motor_control_mode'] + des_speed + "_feet_bool.png"
    plt.savefig(fig_name)

    if (env_config['motor_control_mode'] == "CPG"):
        # f3 = plt.figure()
        # plt.plot(x_axis, CPG_r_history_FL, label='FL')
        # plt.plot(x_axis, CPG_r_history_FR, label='FR')
        # # plt.plot(x_axis, CPG_r_history_RR, label='RR')
        # # plt.plot(x_axis, CPG_r_history_RL, label='RL')
        # plt.xlabel('timesteps')
        # plt.ylabel('CPG r value')
        # plt.legend()
        # fig_name = env_config['motor_control_mode'] + des_speed + "_r.png"
        # plt.savefig(fig_name)

        # f4 = plt.figure()
        # plt.plot(x_axis, CPG_theta_history_FL, label='FL')
        # plt.plot(x_axis, CPG_theta_history_FR, label='FR')
        # # plt.plot(x_axis, CPG_theta_history_RR, label='RR')
        # # plt.plot(x_axis, CPG_theta_history_RL, label='RL')
        # plt.xlabel('timesteps')
        # plt.ylabel('CPG theta value')
        # plt.legend()
        # fig_name = env_config['motor_control_mode'] + des_speed + "_theta.png"
        # plt.savefig(fig_name)

        f5 = plt.figure()
        # plt.plot(x_axis, CPG_r_history_FL*np.sin(CPG_theta_history_FL), label='front left')
        # plt.plot(x_axis, CPG_r_history_FR*np.sin(CPG_theta_history_FR), label='front right')
        plt.plot(x_axis, CPG_r_history_RL*np.sin(CPG_theta_history_FL), label='rear right')
        plt.plot(x_axis, CPG_r_history_RL*np.sin(CPG_theta_history_FR), label='rear left')
        plt.xlabel('timesteps')
        plt.ylabel('CPG projected on the y axis')
        plt.legend()
        fig_name = env_config['motor_control_mode'] + des_speed + "_front.png"
        plt.savefig(fig_name)

    plt.show()