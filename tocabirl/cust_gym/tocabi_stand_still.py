import numpy as np
from gym.envs.mujoco import mujoco_env
from gym.envs.robotics.rotations import quat2euler
from gym import utils
from math import exp, sin, cos, pi
import time
from pyquaternion import Quaternion
from . import tocabi_stand_still_env
from .utils.cubic import cubic
from .utils.lpf import lpf
import os

GroundCollisionCheckBodyList = ["base_link",\
            "R_HipRoll_Link", "R_HipCenter_Link", "R_Thigh_Link", "R_Knee_Link",\
            "L_HipRoll_Link", "L_HipCenter_Link", "L_Thigh_Link", "L_Knee_Link",\
            "Waist1_Link", "Waist2_Link", "Upperbody_Link", \
            "R_Shoulder1_Link", "R_Shoulder2_Link", "R_Shoulder3_Link", "R_Armlink_Link", "R_Elbow_Link", "R_Forearm_Link", "R_Wrist1_Link", "R_Wrist2_Link",\
            "L_Shoulder1_Link", "L_Shoulder2_Link", "L_Shoulder3_Link", "L_Armlink_Link", "L_Elbow_Link", "L_Forearm_Link", "L_Wrist1_Link","L_Wrist2_Link"]

SelfCollisionCheckBodyList = GroundCollisionCheckBodyList + ["L_AnkleCenter_Link", "L_AnkleRoll_Link", "L_Foot_Link", "R_AnkleCenter_Link", "R_AnkleRoll_Link", "R_Foot_Link"]

ObstacleList = ["obstacle1", "obstacle2", "obstacle3", "obstacle4", "obstacle5", "obstacle6", "obstacle7", "obstacle8", "obstacle9"]


class DYROSTocabiEnv(tocabi_stand_still_env.TocabiEnv):
    def __init__(self, frameskip=8):
        super(DYROSTocabiEnv, self).__init__('dyros_tocabi.xml', frameskip)
        # utils.EzPickle.__init__(self)
        for id in GroundCollisionCheckBodyList:
            self.ground_collision_check_id.append(self.model.body_name2id(id))
        for id in SelfCollisionCheckBodyList:
            self.self_collision_check_id.append(self.model.body_name2id(id))
        self.ground_id.append(0)
        # for id in ObstacleList:
        #     self.ground_id.append(self.model.body_name2id(id))
        print("Collision Check ID", self.ground_collision_check_id)
        print("Self Collision Check ID", self.self_collision_check_id)
        print("Ground ID", self.ground_id)
        print("R Foot ID",self.model.body_name2id("R_Foot_Link"))
        print("L Foot ID",self.model.body_name2id("L_Foot_Link"))

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        orientation = Quaternion(array=qpos[3:7])
        
        orientation_noise = np.random.uniform(-0.034, 0.034,3)
        orientation = orientation * Quaternion(axis=(1.0, 0.0, 0.0), radians=orientation_noise[0]) * \
                        Quaternion(axis=(0.0, 1.0, 0.0), radians=orientation_noise[1]) * Quaternion(axis=(0.0, 0.0, 1.0), radians=orientation_noise[2])
        euler_angle = quat2euler(orientation.elements)

        mocap_cycle_period = self.mocap_data_num* self.mocap_cycle_dt
        phase = np.array((self.init_mocap_data_idx + self.time % mocap_cycle_period / self.mocap_cycle_dt) % self.mocap_data_num / self.mocap_data_num)
        sin_phase = np.array(sin(2*pi*phase))
        cos_phase = np.array(cos(2*pi*phase))
        
        return np.concatenate([[euler_angle[0], euler_angle[1]],
                    (self.qpos_noise).flatten(),
                    # (self.qvel_noise).flatten(),
                    self.qvel_lpf.flatten(),
                    sin_phase.flatten(),
                    cos_phase.flatten()])


    def step(self, a):        
        done_by_early_stop = False
        self.action_log.append(a)
        # print("Action: ", a)
        # a[:] = 0.0

        # Simulation
        for _ in range(self.frame_skip):
            if (len(self.action_log) < 2):
                a_idx = -len(self.action_log)
            else:
                a_idx = -2
            self.do_simulation(self.action_log[a_idx],1) 
            qpos = self.sim.data.qpos[7:]
            self.qpos_noise = qpos + np.random.uniform(-0.00001, 0.00001, len(qpos))
            self.qvel_noise = (self.qpos_noise - self.qpos_pre) / self.model.opt.timestep
            self.qpos_pre = np.copy(self.qpos_noise)
            self.qvel_lpf = lpf(self.qvel_noise, self.qvel_lpf, 1/self.model.opt.timestep, 4.0)

        self.time += self.dt
 
        # Collision Check
        for i in range(self.sim.data.ncon):
            if (any(self.model.geom_bodyid[self.sim.data.contact[i].geom1] == ground_id for ground_id in self.ground_id) and \
                    any(self.model.geom_bodyid[self.sim.data.contact[i].geom2] == collisioncheckid for collisioncheckid in self.ground_collision_check_id)) or \
                (any(self.model.geom_bodyid[self.sim.data.contact[i].geom2] == ground_id for ground_id in self.ground_id) and \
                    any(self.model.geom_bodyid[self.sim.data.contact[i].geom1] == collisioncheckid for collisioncheckid in self.ground_collision_check_id)):
                done_by_early_stop = True # Ground-Body contact
            if (any(self.model.geom_bodyid[self.sim.data.contact[i].geom1] == self_col_id for self_col_id in self.self_collision_check_id) and \
                    any(self.model.geom_bodyid[self.sim.data.contact[i].geom2] == self_col_id for self_col_id in self.self_collision_check_id)):
                done_by_early_stop = True # Self Collision contact

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        basequat = self.sim.data.get_body_xquat("Neck_Link")
        quat_desired = Quaternion(array=[1,0,0,0])  
        baseQuatError = (quat_desired.conjugate * Quaternion(array=basequat)).angle

        mimic_body_orientation_reward =  0.1  * exp(-13.2*abs(baseQuatError)) 
        mimic_qpos_reward =  0.7*exp(-2.0*(np.linalg.norm(self.init_q_desired[7:] - qpos[7:])**2))
        mimic_qvel_reward =  0.2*exp(-0.01*(np.linalg.norm(self.init_qvel[6:] - qvel[6:])**2))
        mimic_base_pose_reward = 0.1*exp(-9.2*np.linalg.norm(self.init_q_desired[0:3] - qpos[0:3]))

        reward = mimic_body_orientation_reward + mimic_qpos_reward + mimic_qvel_reward + mimic_base_pose_reward
        
        if not done_by_early_stop:
            self.epi_len += 1
            self.epi_reward += reward
            if (self.spec is not None and self.epi_len == self.spec.max_episode_steps):
                print("Epi len: ", self.epi_len)
                # np.savetxt("./result/"+"data_log"+".txt", self.data_log,delimiter='\t')
                # np.savetxt("./result/"+"action_log"+".txt", self.action_log,delimiter='\t')

                return self._get_obs(), reward, done_by_early_stop, dict(episode=dict(r=self.epi_reward, l=self.epi_len), specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward, mimic_qpos_reward=mimic_qpos_reward, mimic_qvel_reward=mimic_qvel_reward, mimic_base_pose_reward=mimic_base_pose_reward))

            return self._get_obs(), reward, done_by_early_stop, dict(specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward, mimic_qpos_reward=mimic_qpos_reward, mimic_qvel_reward=mimic_qvel_reward, mimic_base_pose_reward=mimic_base_pose_reward))
        else:
            mimic_body_orientation_reward = 0.0
            mimic_qpos_reward = 0.0
            mimic_qvel_reward = 0.0
            mimic_base_pose_reward = 0.0
            reward = 0.0

            print("Epi len: ", self.epi_len)
            # try: os.mkdir('./result')
            # except: pass
            # np.savetxt("./result/"+"data_log"+".txt", self.data_log,delimiter='\t')
            # np.savetxt("./result/"+"action_log"+".txt", self.action_log,delimiter='\t')

            return self._get_obs(), reward, done_by_early_stop, dict(episode=dict(r=self.epi_reward, l=self.epi_len), specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward, mimic_qpos_reward=mimic_qpos_reward, mimic_qvel_reward=mimic_qvel_reward, mimic_base_pose_reward=mimic_base_pose_reward))


    def reset_model(self):
        self.time = 0.0
        self.epi_len = 0
        self.epi_reward = 0

        # Dynamics Randomization
        body_mass = np.array(self.nominal_body_mass)
        body_mass_noise = np.random.uniform(0.8, 1.2, len(body_mass))
        body_mass = body_mass * body_mass_noise
        self.model.body_mass[:]  = body_mass

        body_inertia = np.array(self.nominal_body_inertia)
        body_inertia_noise = np.random.uniform(0.8, 1.2, len(body_inertia))
        body_inertia = np.multiply(body_inertia, body_inertia_noise[:, np.newaxis])
        self.model.body_inertia[:]  = body_inertia

        body_ipos = np.array(self.nominal_body_ipos)
        body_ipos_noise = np.random.uniform(0.8, 1.2, len(body_ipos))
        body_ipos = np.multiply(body_ipos, body_ipos_noise[:, np.newaxis])
        self.model.body_ipos[:]  = body_ipos
        
        dof_damping = np.array(self.nominal_dof_damping)
        noise_scale = 2.0
        dof_damping_noise = np.random.uniform(1/noise_scale, noise_scale, len(dof_damping))
        dof_damping = dof_damping * dof_damping_noise
        self.model.dof_damping[:]  = dof_damping

        dof_frictionloss = np.array(self.nominal_dof_frictionloss)
        dof_frictionloss_noise = np.random.uniform(1/noise_scale, noise_scale, len(dof_frictionloss))
        dof_frictionloss = dof_frictionloss * dof_frictionloss_noise
        self.model.dof_frictionloss[:]  = dof_frictionloss
        
        self.set_state(self.init_q_desired, self.init_qvel)  

        self.init_mocap_data_idx = np.random.randint(low=0, high=self.mocap_data_num)

        self.qpos_noise = self.init_q_desired[7:]
        self.qpos_pre = self.init_q_desired[7:]
        self.qvel_noise.fill(0)
        self.qvel_lpf.fill(0)

        self.action_log = []
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
