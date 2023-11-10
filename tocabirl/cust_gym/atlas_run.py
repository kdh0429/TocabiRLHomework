import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from math import exp, sin, cos, pi
import time
from pyquaternion import Quaternion
from . import atlas_run_env
from .utils.cubic import cubic
from .utils.lpf import lpf
from .utils.rotation import quat2fixedXYZ

GroundCollisionCheckBodyList = ["pelvis",\
            "ltorso", "mtorso", "utorso",\
            "l_clav", "l_scap", "l_uarm", "l_larm", "l_farm", "l_hand",\
            "head",\
            "r_clav", "r_scap", "r_uarm", "r_larm", "r_farm", "r_hand",\
            "l_uglut", "l_lglut", "l_uleg", "l_lleg", "l_talus",\
            "r_uglut", "r_lglut", "r_uleg", "r_lleg", "r_talus"]

SelfCollisionCheckBodyList = GroundCollisionCheckBodyList + ["l_foot", "r_foot"]
LeftLegList = ["l_uglut", "l_lglut", "l_uleg", "l_lleg", "l_talus", "l_foot"]
RightLegList = ["r_uglut", "r_lglut", "r_uleg", "r_lleg", "r_talus", "r_foot"]

ObstacleList = ["obstacle1", "obstacle2", "obstacle3", "obstacle4", "obstacle5", "obstacle6", "obstacle7", "obstacle8", "obstacle9"]

Kp = np.array([600.0, 1000.0, 1000.0,
     400.0, 400.0, 400.0, 400.0, 100.0, 100.0,
     100.0,
     400.0, 400.0, 400.0, 400.0, 100.0, 100.0,
     2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
     2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,]) / 8.0
# Kp[16:28] /= 10.0

Kv = np.array([200.0, 100.0, 100.0,
     10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
     2.0, 
     10.0,  10.0, 10.0, 10.0, 3.0, 3.0,
     15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
     15.0, 50.0, 20.0, 25.0, 24.0, 24.0,]) / 4.0

control_freq_scale = 1

class AtlasEnv(atlas_run_env.AtlasEnv):
    def __init__(self, frameskip=int(8/control_freq_scale)):
        super(AtlasEnv, self).__init__('atlas.xml', frameskip)
        # utils.EzPickle.__init__(self)
        for id in GroundCollisionCheckBodyList:
            self.ground_collision_check_id.append(self.model.body_name2id(id))
        for id in SelfCollisionCheckBodyList:
            self.self_collision_check_id.append(self.model.body_name2id(id))
        for id in LeftLegList:
            self.l_leg_id.append(self.model.body_name2id(id))
        for id in RightLegList:
            self.r_leg_id.append(self.model.body_name2id(id))
        self.ground_id.append(0)
        self.right_foot_id.append(self.model.body_name2id("r_foot"))
        self.left_foot_id.append(self.model.body_name2id("l_foot"))
        # for id in ObstacleList:
        #     self.ground_id.append(self.model.body_name2id(id))
        print("Collision Check ID", self.ground_collision_check_id)
        print("Self Collision Check ID", self.self_collision_check_id)
        print("Ground ID", self.ground_id)
        print("R Foot ID",self.model.body_name2id("r_foot"))
        print("L Foot ID",self.model.body_name2id("l_foot"))

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
                    (self.qpos_noise[16:28]).flatten(),
                    # (self.qvel_noise).flatten(),
                    self.qvel_lpf[16:28].flatten(),
                    # qvel[3:6].flatten(),
                    sin_phase.flatten(),
                    cos_phase.flatten(),
                    [self.target_vel[0]]])

        self.action_last = np.copy(self.action_cur)     
        self.qvel_pre = np.copy(qvel[6:])
        
        return cur_obs


    def step(self, a):
        a = a * self.action_high
        done_by_early_stop = False
        self.action_cur = a[0:-1]
        # print("Action: ", a)
        # a[:] = 0.0

        mocap_cycle_period = self.mocap_data_num* self.mocap_cycle_dt

        local_time = self.time % mocap_cycle_period
        local_time_plus_init = (local_time + self.init_mocap_data_idx*self.mocap_cycle_dt) % mocap_cycle_period
        self.mocap_data_idx = (self.init_mocap_data_idx + int(local_time / self.mocap_cycle_dt)) % self.mocap_data_num
        next_idx = self.mocap_data_idx + 1 
        
        target_data_qpos = np.zeros_like(a)    
        target_data_qpos = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,8:40], self.mocap_data[next_idx,8:40], 0.0, 0.0)

        if (self.perturbation_on):
            if (self.epi_len % control_freq_scale*2000 == self.perturb_timing):
                self.magnitude = np.random.uniform(0, 250)
                theta = np.random.uniform(0, 2*pi)
                self.new_xfrc[1,0] = self.magnitude * cos(theta)
                self.new_xfrc[1,1] = self.magnitude * sin(theta)
                self.pert_duration = control_freq_scale*np.random.randint(1, 50)
                self.cur_pert_duration = 0
            
            if (self.cur_pert_duration < self.pert_duration):
                self.sim.data.xfrc_applied[:] = self.new_xfrc
                self.cur_pert_duration = self.cur_pert_duration + 1
            else:
                self.sim.data.xfrc_applied[:] = np.zeros_like(self.sim.data.xfrc_applied)

        # Simulation
        for _ in range(self.frame_skip):
            upper_torque = Kp[0:16]*(self.init_q_desired[7:23] - self.qpos_noise[0:16]) + Kv[0:16]*(-self.qvel_noise[0:16])

            self.action_log.append(np.concatenate([upper_torque, self.action_cur]))
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

            # init_q_pos = np.copy(self.init_q_desired)
            # init_qvel = np.copy(self.init_qvel)
            # init_q_pos[7:23] = self.init_q_desired[7:23]
            # init_q_pos[23:35] = target_data_qpos[0:12]
            # self.set_state(init_q_pos, init_qvel) 
            
        self.time += self.dt
        self.time += a[-1]
 
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        
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
        # if (np.in1d(geom1, self.self_collision_check_id) * \
        #         np.in1d(geom2, self.self_collision_check_id)).any():
        #     done_by_early_stop = True # Self Collision contact 
        if (np.in1d(geom1, self.l_leg_id) * \
                np.in1d(geom2, self.r_leg_id)).any() or \
            (np.in1d(geom2, self.r_leg_id) * \
                np.in1d(geom1, self.l_leg_id)).any():
            done_by_early_stop = True # Left leg and right leg cross

        left_foot_contact = False
        if (np.in1d(geom1, self.ground_id) * \
                np.in1d(geom2, self.left_foot_id)).any() or \
            (np.in1d(geom2, self.left_foot_id) * \
                np.in1d(geom1, self.ground_id)).any():
            left_foot_contact = True 
        right_foot_contact = False
        if (np.in1d(geom1, self.ground_id) * \
                np.in1d(geom2, self.right_foot_id)).any() or \
            (np.in1d(geom2, self.right_foot_id) * \
                np.in1d(geom1, self.ground_id)).any():
            right_foot_contact = True
            
        if (qpos[2] < 0.7):
            done_by_early_stop = True

        # self.read_sensor_data()
        basequat = self.sim.data.get_body_xquat("head")
        quat_desired = Quaternion(array=[1,0,0,0])  
        baseQuatError = (quat_desired.conjugate * Quaternion(array=basequat)).angle

        pelvis_quat = Quaternion(array=qpos[3:7])
        pelvis_vel_local = pelvis_quat.conjugate.rotate(qvel[0:3])

        mimic_body_orientation_reward =  0.3 * exp(-13.2*abs(baseQuatError)) 
        qpos_regulation = 0.35*exp(-4.0*(np.linalg.norm(target_data_qpos[0:12] - qpos[23:])**2))
        qvel_regulation = 0.05*exp(-0.01*(np.linalg.norm(self.init_qvel[6:] - qvel[6:])**2))
        body_vel_reward = 0.3*exp(-3.0*(np.linalg.norm(pelvis_vel_local[0:2] - self.target_vel)**2))
        contact_force_penalty = 0.05*(exp(-0.01*(np.linalg.norm(self.ft_left_foot) + np.linalg.norm(self.ft_right_foot))))
        torque_regulation = 0.05*exp(-0.01*(np.linalg.norm(self.action_cur)))
        torque_diff_regulation = 0.2*(exp(-0.01*(np.linalg.norm(self.action_cur - self.action_last))))
        qacc_regulation = 0.05*exp(-20.0*(np.linalg.norm(self.qvel_pre - qvel[6:])**2))
        if ((self.mocap_data_idx < 9) or (22 <= self.mocap_data_idx)): # Left support
            if (left_foot_contact and not right_foot_contact):
                foot_contact_reward = 0.2
            else:
                foot_contact_reward = 0.0
        elif (11 <= self.mocap_data_idx and self.mocap_data_idx < 20): # Right Support
            if (right_foot_contact and not left_foot_contact):
                foot_contact_reward = 0.2
            else:
                foot_contact_reward = 0.0
        else:
            if (not right_foot_contact and not left_foot_contact):
                foot_contact_reward = 0.2
            else:
                foot_contact_reward = 0.0
        contact_force_diff_regulation = 0.1*exp(-0.01*(np.linalg.norm(self.ft_left_foot - self.ft_left_foot_pre) + np.linalg.norm(self.ft_right_foot - self.ft_right_foot_pre)))

        reward = mimic_body_orientation_reward + qpos_regulation + qvel_regulation + contact_force_penalty + torque_regulation + torque_diff_regulation + qacc_regulation + body_vel_reward + foot_contact_reward + contact_force_diff_regulation

        self.ft_left_foot_pre = np.copy(self.ft_left_foot)
        self.ft_right_foot_pre = np.copy(self.ft_right_foot)
        
        if not done_by_early_stop:
            self.epi_len += 1
            self.epi_reward += reward
            if (self.spec is not None and self.epi_len == self.spec.max_episode_steps):
                print("Epi len: ", self.epi_len)
                # np.savetxt("./result/"+"action_log"+".txt", self.action_log,delimiter='\t')
                # np.savetxt("./result/"+"data_log"+".txt", self.data_log,delimiter='\t')

                return self._get_obs(), reward, done_by_early_stop, dict(episode=dict(r=self.epi_reward, l=self.epi_len), \
                    specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward,\
                                        qpos_regulation=qpos_regulation,\
                                        qvel_regulation=qvel_regulation,\
                                        contact_force_penalty=contact_force_penalty,
                                        torque_regulation=torque_regulation,
                                        torque_diff_regulation=torque_diff_regulation,
                                        qacc_regulation=qacc_regulation,
                                        body_vel_reward=body_vel_reward,
                                        foot_contact_reward=foot_contact_reward,
                                        contact_force_diff_regulation=contact_force_diff_regulation))

            return self._get_obs(), reward, done_by_early_stop, \
                dict(specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward, \
                                        qpos_regulation=qpos_regulation,\
                                        qvel_regulation=qvel_regulation,\
                                        contact_force_penalty=contact_force_penalty,
                                        torque_regulation=torque_regulation,
                                        torque_diff_regulation=torque_diff_regulation,
                                        qacc_regulation=qacc_regulation,
                                        body_vel_reward=body_vel_reward,
                                        foot_contact_reward=foot_contact_reward,
                                        contact_force_diff_regulation=contact_force_diff_regulation))
        else:
            mimic_body_orientation_reward = 0.0
            qpos_regulation = 0.0
            qvel_regulation = 0.0
            contact_force_penalty = 0.0
            torque_regulation = 0.0
            torque_diff_regulation = 0.0
            qacc_regulation = 0.0
            body_vel_reward = 0.0
            foot_contact_reward = 0.0
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
                                    contact_force_penalty=contact_force_penalty,
                                    torque_regulation=torque_regulation,
                                    torque_diff_regulation=torque_diff_regulation,
                                    qacc_regulation=qacc_regulation,
                                    body_vel_reward=body_vel_reward,
                                    foot_contact_reward=foot_contact_reward,
                                    contact_force_diff_regulation=contact_force_diff_regulation))

    def reset_model(self):
        self.time = 0.0
        self.epi_len = 0
        self.epi_reward = 0

        # Dynamics Randomization
        # body_mass = np.array(self.nominal_body_mass)
        # body_mass_noise = np.random.uniform(0.6, 1.4, len(body_mass))
        # body_mass = body_mass * \
        #             np.array([body_mass_noise[0],
        #                     body_mass_noise[1],
        #                     body_mass_noise[2], body_mass_noise[3], body_mass_noise[4], body_mass_noise[5], body_mass_noise[6], body_mass_noise[7], body_mass_noise[8],
        #                     body_mass_noise[2], body_mass_noise[3], body_mass_noise[4], body_mass_noise[5], body_mass_noise[6], body_mass_noise[7], body_mass_noise[8],
        #                     body_mass_noise[16], body_mass_noise[17], body_mass_noise[18], 
        #                     body_mass_noise[19], body_mass_noise[20], body_mass_noise[21], body_mass_noise[22], body_mass_noise[23], body_mass_noise[24], body_mass_noise[25], body_mass_noise[26], 
        #                     body_mass_noise[27], body_mass_noise[28],
        #                     body_mass_noise[19], body_mass_noise[20], body_mass_noise[21], body_mass_noise[22], body_mass_noise[23], body_mass_noise[24], body_mass_noise[25], body_mass_noise[26]])
        # self.model.body_mass[:]  = body_mass

        # body_inertia = np.array(self.nominal_body_inertia)
        # body_inertia_noise = np.random.uniform(0.6, 1.4, len(body_inertia))
        # body_inertia = np.multiply(body_inertia, body_inertia_noise[:, np.newaxis])
        # self.model.body_inertia[:]  = body_inertia

        # body_ipos = np.array(self.nominal_body_ipos)
        # body_ipos_noise = np.random.uniform(0.6, 1.4, len(body_ipos))
        # body_ipos = np.multiply(body_ipos, body_ipos_noise[:, np.newaxis])
        # self.model.body_ipos[:]  = body_ipos
        
        # dof_damping = np.array(self.nominal_dof_damping)
        # dof_damping_noise = np.random.uniform(0.6, 1.4, len(dof_damping))#np.random.uniform(1/noise_scale, noise_scale, len(dof_damping))
        # dof_damping = dof_damping * dof_damping_noise
        # self.model.dof_damping[:]  = dof_damping

        # dof_frictionloss = np.array(self.nominal_dof_frictionloss)
        # dof_frictionloss_noise = np.random.uniform(0.6, 1.4, len(dof_frictionloss))#np.random.uniform(1/noise_scale, noise_scale, len(dof_frictionloss))
        # dof_frictionloss = dof_frictionloss * dof_frictionloss_noise
        # self.model.dof_frictionloss[:]  = dof_frictionloss

        # Delay Randomization
        self.action_delay = 5 #np.random.randint(low=3, high=7)

        if (np.random.rand(1) < 0.5):
            self.init_mocap_data_idx = 3#np.random.randint(low=0, high=self.mocap_data_num)
        else:
            self.init_mocap_data_idx = 15
        init_q_pos = np.copy(self.init_q_desired)
        # init_q_pos[7:] = self.mocap_data[self.init_mocap_data_idx,1:]
        
        init_qvel = np.copy(self.init_qvel)
        # init_qvel[0] = np.random.uniform(0.0, 0.3)
        self.set_state(init_q_pos, init_qvel)  

        # init_q_pos[2] = init_q_pos[2] - (self.sim.data.get_body_xpos("R_Foot_Link")[2] - 0.15811) # To offset so that feet are on ground
        # self.set_state(init_q_pos, self.init_qvel)  

        self.qpos_noise = init_q_pos[7:]
        self.qpos_pre = init_q_pos[7:]
        self.qvel_noise.fill(0)
        self.qvel_lpf.fill(0)

        self.read_sensor_data()
        self.ft_left_foot_pre = np.copy(self.ft_left_foot)
        self.ft_right_foot_pre = np.copy(self.ft_right_foot)

        self.target_vel = np.array([np.random.uniform(-0.2, 2.0), 0.0])

        self.action_log = []
        self.data_log = []

        self.action_last.fill(0)
        self.qvel_pre.fill(0)

        self.perturb_timing = np.random.randint(1,control_freq_scale*2000)
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
        self.perturbation_on = True