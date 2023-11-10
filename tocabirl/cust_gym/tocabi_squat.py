import numpy as np
from gym.envs.mujoco import mujoco_env
from gym.envs.robotics.rotations import quat2euler
from gym import utils
from math import exp, sin, cos, pi
import time
from pyquaternion import Quaternion
from . import tocabi_squat_env
from .utils.cubic import cubic
from .utils.lpf import lpf
from .utils.rotation import quat2fixedXYZ

GroundCollisionCheckBodyList = ["base_link",\
            "R_HipRoll_Link", "R_HipCenter_Link", "R_Thigh_Link", "R_Knee_Link",\
            "L_HipRoll_Link", "L_HipCenter_Link", "L_Thigh_Link", "L_Knee_Link",\
            "Waist1_Link", "Waist2_Link", "Upperbody_Link", \
            "R_Shoulder1_Link", "R_Shoulder2_Link", "R_Shoulder3_Link", "R_Armlink_Link", "R_Elbow_Link", "R_Forearm_Link", "R_Wrist1_Link", "R_Wrist2_Link",\
            "L_Shoulder1_Link", "L_Shoulder2_Link", "L_Shoulder3_Link", "L_Armlink_Link", "L_Elbow_Link", "L_Forearm_Link", "L_Wrist1_Link","L_Wrist2_Link"]

SelfCollisionCheckBodyList = GroundCollisionCheckBodyList + ["L_AnkleCenter_Link", "L_AnkleRoll_Link", "L_Foot_Link", "R_AnkleCenter_Link", "R_AnkleRoll_Link", "R_Foot_Link"]

ObstacleList = ["obstacle1", "obstacle2", "obstacle3", "obstacle4", "obstacle5", "obstacle6", "obstacle7", "obstacle8", "obstacle9"]

Kp = np.array([2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
     2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
     6000.0, 10000.0, 10000.0,
     400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0,
     100.0, 100.0,
     400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0])

Kv = np.array([15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
     15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
     200.0, 100.0, 100.0,
     10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
     2.0, 2.0,
     10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0])

class DYROSTocabiEnv(tocabi_squat_env.TocabiEnv):
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
        
        # orientation_noise = np.random.uniform(-0.034, 0.034,3)
        # orientation = orientation * Quaternion(axis=(1.0, 0.0, 0.0), radians=orientation_noise[0]) * \
        #                 Quaternion(axis=(0.0, 1.0, 0.0), radians=orientation_noise[1]) * Quaternion(axis=(0.0, 0.0, 1.0), radians=orientation_noise[2])

        fixed_angle = quat2fixedXYZ(orientation.elements)

        mocap_cycle_period = self.mocap_data_num* self.mocap_cycle_dt
        phase = np.array((self.init_mocap_data_idx + self.time % mocap_cycle_period / self.mocap_cycle_dt) % self.mocap_data_num / self.mocap_data_num)
        sin_phase = np.array(sin(2*pi*phase))
        cos_phase = np.array(cos(2*pi*phase))     

        # cur_obs = np.concatenate([[euler_angle[0], euler_angle[1], euler_angle[2]],
        #     (self.qpos_noise).flatten(),
        #     self.qvel_lpf.flatten(),
        #     qvel[3:6].flatten(),
        #     sin_phase.flatten(),
        #     cos_phase.flatten(),
        #     [self.target_vel]])

        cur_obs = np.concatenate([[fixed_angle[0], fixed_angle[1], fixed_angle[2]],
                    (self.qpos_noise[0:12]).flatten(),
                    # (self.qvel_noise).flatten(),
                    self.qvel_lpf[0:12].flatten(),
                    # qvel[3:6].flatten(),
                    sin_phase.flatten(),
                    cos_phase.flatten()])

        self.action_last = np.copy(self.action_cur)     
        self.qvel_pre = np.copy(qvel[6:])
        
        return (cur_obs - self.obs_mean[:-1]) / np.sqrt(self.obs_var[0:-1] + 1e-8)


    def step(self, a):
        a = a * self.action_high
        done_by_early_stop = False
        self.action_cur = a * self.motor_constant_scale
        # print("Action: ", a)
        # a[:] = 0.0

        mocap_cycle_period = self.mocap_data_num* self.mocap_cycle_dt

        local_time = self.time % mocap_cycle_period
        local_time_plus_init = (local_time + self.init_mocap_data_idx*self.mocap_cycle_dt) % mocap_cycle_period
        self.mocap_data_idx = (self.init_mocap_data_idx + int(local_time / self.mocap_cycle_dt)) % self.mocap_data_num
        next_idx = self.mocap_data_idx + 1 
        
        target_data_qpos = np.zeros_like(a)    
        target_data_qpos = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,1:34], self.mocap_data[next_idx,1:34], 0.0, 0.0)

        # Simulation
        for _ in range(self.frame_skip):
            upper_torque = Kp[12:]*(target_data_qpos[12:] - self.qpos_noise[12:]) + Kv[12:]*(-self.qvel_noise[12:])
            self.action_log.append(np.concatenate([a,upper_torque]))
            # self.action_log.append(a)
            if (len(self.action_log) < self.action_delay):
                a_idx = -len(self.action_log)
            else:
                a_idx = -self.action_delay
            self.do_simulation(self.action_log[a_idx],1) 

            qpos = self.sim.data.qpos[7:]
            #self.qpos_noise = qpos + np.random.uniform(-0.00001, 0.00001, len(qpos))
            self.qpos_noise = qpos + np.clip(np.random.normal(0, 0.00001 / 3.0, len(qpos)), -0.00001, 0.00001)
            self.qvel_noise = (self.qpos_noise - self.qpos_pre) / self.model.opt.timestep
            self.qpos_pre = np.copy(self.qpos_noise)
            self.qvel_lpf = lpf(self.qvel_noise, self.qvel_lpf, 1/self.model.opt.timestep, 4.0)
        
            # self.render()
        self.time += self.dt

        self.read_sensor_data()

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        
        basequat = self.sim.data.get_body_xquat("Neck_Link")
        quat_desired = Quaternion(array=[1,0,0,0])  
        baseQuatError = (quat_desired.conjugate * Quaternion(array=basequat)).angle
        
        orientation = Quaternion(array=qpos[3:7])
        fixed_angle = quat2fixedXYZ(orientation.elements)

        # Collision Check
        geom1 = np.zeros(self.sim.data.ncon)
        geom2 = np.zeros(self.sim.data.ncon)
        for i in range(self.sim.data.ncon):
            geom1[i] = self.model.geom_bodyid[self.sim.data.contact[i].geom1]
            geom2[i] = self.model.geom_bodyid[self.sim.data.contact[i].geom2]
            
        if (np.in1d(geom1, self.ground_id) * \
                np.in1d(geom2, self.ground_collision_check_id)).any() or \
            (np.in1d(geom2, self.ground_id) * \
                np.in1d(geom1, self.ground_collision_check_id)).any():
            done_by_early_stop = True # Ground-Body contact
        if (np.in1d(geom1, self.self_collision_check_id) * \
                np.in1d(geom2, self.self_collision_check_id)).any():
            done_by_early_stop = True # Self Collision contact

        if ((np.abs(fixed_angle) > 30*3.14/180).any()):
            done_by_early_stop = True


        mimic_body_orientation_reward =  0.3 * exp(-13.2*abs(baseQuatError)) 
        qpos_regulation = 0.35*exp(-2.0*(np.linalg.norm(target_data_qpos - qpos[7:])**2))
        qvel_regulation = 0.05*exp(-0.01*(np.linalg.norm(self.init_qvel[6:] - qvel[6:])**2))
        torque_diff_regulation = 0.3*(exp(-0.01*(np.linalg.norm(a - self.action_last))))
        qacc_regulation = 0.3*exp(-20.0*(np.linalg.norm(self.qvel_pre - qvel[6:])**2))
        torque_regulation = 0.3*exp(-0.01*(np.linalg.norm(self.action_cur)))
        contact_force_symmetric_reward = 0.15*exp(-0.01*(np.linalg.norm(self.ft_left_foot - self.ft_right_foot)))
        contact_force_diff_regulation = 0.1*exp(-0.01*(np.linalg.norm(self.ft_left_foot - self.ft_left_foot_pre) + np.linalg.norm(self.ft_right_foot - self.ft_right_foot_pre)))

        
        reward = mimic_body_orientation_reward + qpos_regulation + qvel_regulation + torque_diff_regulation + qacc_regulation + torque_regulation + contact_force_symmetric_reward + contact_force_diff_regulation

        self.ft_left_foot_pre = np.copy(self.ft_left_foot)
        self.ft_right_foot_pre = np.copy(self.ft_right_foot)
        
        if not done_by_early_stop:
            self.epi_len += 1
            self.epi_reward += reward
            if (self.spec is not None and self.epi_len == self.spec.max_episode_steps):
                print("Epi len: ", self.epi_len)
                # np.savetxt("./result/"+"action_log"+".txt", self.action_log,delimiter='\t')
                # np.savetxt("./result/noise/"+"data_log_w_r_wo_noise_w_regulation_2"+".txt", self.data_log,delimiter='\t')
                # np.savetxt("./result/noise/"+"data_log_w_r_lpf_10_w_regulation_2"+".txt", self.data_log,delimiter='\t')

                return self._get_obs(), reward, done_by_early_stop, dict(episode=dict(r=self.epi_reward, l=self.epi_len), \
                    specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward,\
                                        qpos_regulation=qpos_regulation,\
                                        qvel_regulation=qvel_regulation,\
                                        torque_diff_regulation=torque_diff_regulation,
                                        qacc_regulation=qacc_regulation,
                                        torque_regulation=torque_regulation,
                                        contact_force_symmetric_reward=contact_force_symmetric_reward,
                                        contact_force_diff_regulation=contact_force_diff_regulation))

            return self._get_obs(), reward, done_by_early_stop, \
                dict(specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward, \
                                        qpos_regulation=qpos_regulation,\
                                        qvel_regulation=qvel_regulation,\
                                        torque_diff_regulation=torque_diff_regulation,
                                        qacc_regulation=qacc_regulation,
                                        torque_regulation=torque_regulation,
                                        contact_force_symmetric_reward=contact_force_symmetric_reward,
                                        contact_force_diff_regulation=contact_force_diff_regulation))
        else:
            mimic_body_orientation_reward = 0.0
            qpos_regulation = 0.0
            qvel_regulation = 0.0
            torque_diff_regulation = 0.0
            qacc_regulation = 0.0
            torque_regulation = 0.0
            contact_force_symmetric_reward = 0.0
            contact_force_diff_regulation = 0.0
            reward = 0.0

            print("Epi len: ", self.epi_len)            
            # try: os.mkdir('./result')
            # except: pass
            # np.savetxt("./result/"+"data_log"+".txt", self.data_log,delimiter='\t')
            # np.savetxt("./result/"+"action_log"+".txt", self.action_log,delimiter='\t')

            return self._get_obs(), reward, done_by_early_stop, dict(episode=dict(r=self.epi_reward, l=self.epi_len),\
                 specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward, \
                                    qpos_regulation=qpos_regulation,\
                                    qvel_regulation=qvel_regulation,\
                                    torque_diff_regulation=torque_diff_regulation,
                                    qacc_regulation=qacc_regulation,
                                    torque_regulation=torque_regulation,
                                    contact_force_symmetric_reward=contact_force_symmetric_reward,
                                    contact_force_diff_regulation=contact_force_diff_regulation))

    def reset_model(self):
        self.time = 0.0
        self.epi_len = 0
        self.epi_reward = 0

        # Dynamics Randomization
        body_mass = np.array(self.nominal_body_mass)
        body_mass_noise = np.random.uniform(0.6, 1.4, len(body_mass))
        body_mass = body_mass * \
                    np.array([body_mass_noise[0],
                            body_mass_noise[1],
                            body_mass_noise[2], body_mass_noise[3], body_mass_noise[4], body_mass_noise[5], body_mass_noise[6], body_mass_noise[7], body_mass_noise[8],
                            body_mass_noise[2], body_mass_noise[3], body_mass_noise[4], body_mass_noise[5], body_mass_noise[6], body_mass_noise[7], body_mass_noise[8],
                            body_mass_noise[16], body_mass_noise[17], body_mass_noise[18], 
                            body_mass_noise[19], body_mass_noise[20], body_mass_noise[21], body_mass_noise[22], body_mass_noise[23], body_mass_noise[24], body_mass_noise[25], body_mass_noise[26], 
                            body_mass_noise[27], body_mass_noise[28],
                            body_mass_noise[19], body_mass_noise[20], body_mass_noise[21], body_mass_noise[22], body_mass_noise[23], body_mass_noise[24], body_mass_noise[25], body_mass_noise[26]])
        self.model.body_mass[:]  = body_mass

        body_inertia = np.array(self.nominal_body_inertia)
        body_inertia_noise = np.random.uniform(0.6, 1.4, len(body_inertia))
        body_inertia = np.multiply(body_inertia, body_inertia_noise[:, np.newaxis])
        self.model.body_inertia[:]  = body_inertia
        

        body_ipos = np.array(self.nominal_body_ipos)
        body_ipos_noise = np.random.uniform(0.6, 1.4, len(body_ipos))
        body_ipos = np.multiply(body_ipos, body_ipos_noise[:, np.newaxis])
        self.model.body_ipos[:]  = body_ipos
        
        dof_damping = np.array(self.nominal_dof_damping)
        dof_damping_noise = np.random.uniform(0.6, 1.4, len(dof_damping))#np.random.uniform(1/noise_scale, noise_scale, len(dof_damping))
        dof_damping = dof_damping * dof_damping_noise
        self.model.dof_damping[:]  = dof_damping

        dof_frictionloss = np.array(self.nominal_dof_frictionloss)
        dof_frictionloss_noise = np.random.uniform(0.6, 1.4, len(dof_frictionloss))#np.random.uniform(1/noise_scale, noise_scale, len(dof_frictionloss))
        dof_frictionloss = dof_frictionloss * dof_frictionloss_noise
        self.model.dof_frictionloss[:]  = dof_frictionloss

        # Motor Constant Randomization
        motor_constant_scale = np.random.uniform(0.90, 1.10, 6)
        self.motor_constant_scale = np.tile(motor_constant_scale, 2)

        # Delay Randomization
        self.action_delay = np.random.randint(low=5, high=12)
        self.init_mocap_data_idx = np.random.randint(low=0, high=self.mocap_data_num)
        init_q_pos = np.copy(self.init_q_desired)
        init_q_pos[7:] = self.mocap_data[self.init_mocap_data_idx,1:]
        
        self.set_state(init_q_pos, self.init_qvel)  
        init_q_pos[2] = init_q_pos[2] - (self.sim.data.get_body_xpos("R_Foot_Link")[2] - 0.15811) # To offset so that feet are on ground
        self.set_state(init_q_pos, self.init_qvel)  

        self.qpos_noise = init_q_pos[7:] + np.clip(np.random.normal(0, 0.00001 / 3.0, len(init_q_pos[7:])), -0.00001, 0.00001)
        self.qpos_pre = init_q_pos[7:]
        self.qvel_noise.fill(0)
        self.qvel_lpf.fill(0)

        self.action_log = []
        self.data_log = []

        self.obs_buff = []

        self.action_cur = self.action_last = Kp[0:12]*(init_q_pos[7:19] - self.qpos_noise[0:12])
        self.action_cur.fill(0)
        # self.action_last = Kp*(init_q_pos[7:] - self.qpos_noise[0:])
        self.qvel_pre.fill(0)

        self.read_sensor_data()
        self.ft_left_foot_pre = np.copy(self.ft_left_foot)
        self.ft_right_foot_pre = np.copy(self.ft_right_foot)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

    def read_sensor_data(self):
        self.ft_left_foot = self.data.sensordata[self.ft_left_foot_adr:self.ft_left_foot_adr+3]
        self.ft_right_foot = self.data.sensordata[self.ft_right_foot_adr:self.ft_right_foot_adr+3]

    def perturbation_start(self):
        True