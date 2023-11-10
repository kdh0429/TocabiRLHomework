import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from math import exp, sin, cos, pi
import time
from pyquaternion import Quaternion
from . import tocabi_walk_env
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

control_freq_scale = 1

class DYROSTocabiEnv(tocabi_walk_env.TocabiEnv):
    def __init__(self, frameskip=int(8/control_freq_scale)):
        super(DYROSTocabiEnv, self).__init__('dyros_tocabi.xml', frameskip)
        # utils.EzPickle.__init__(self)
        for id in GroundCollisionCheckBodyList:
            self.ground_collision_check_id.append(self.model.body_name2id(id))
        for id in SelfCollisionCheckBodyList:
            self.self_collision_check_id.append(self.model.body_name2id(id))
        self.ground_id.append(0)
        self.right_foot_id.append(self.model.body_name2id("R_Foot_Link"))
        self.left_foot_id.append(self.model.body_name2id("L_Foot_Link"))
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
        
        fixed_angle = quat2fixedXYZ(orientation.elements)
        fixed_angle[:] = fixed_angle[:] + self.quat_bias

        mocap_cycle_period = self.mocap_data_num* self.mocap_cycle_dt
        phase = np.array((self.init_mocap_data_idx + self.time % mocap_cycle_period / self.mocap_cycle_dt) % self.mocap_data_num / self.mocap_data_num)
        sin_phase = np.array(sin(2*pi*phase))
        cos_phase = np.array(cos(2*pi*phase))     

        cur_obs = np.concatenate([[fixed_angle[0], fixed_angle[1], fixed_angle[2]],
                    (self.qpos_noise[0:12] + self.q_bias).flatten(),
                    (self.qvel_noise[0:12]).flatten(),
                    sin_phase.flatten(),
                    cos_phase.flatten(),
                    [self.target_vel[0]],[self.target_vel[1]],
                    [self.ft_left_foot[2] + self.ft_bias[0]], [self.ft_right_foot[2] + self.ft_bias[1]],
                    [self.torque_left_foot[0] + self.ft_mx_bias[0]], [self.torque_right_foot[0] + self.ft_mx_bias[1]],
                    [self.torque_left_foot[1] + self.ft_my_bias[0]], [self.torque_right_foot[1] + self.ft_my_bias[1]]])

        self.action_last = np.copy(self.action_cur)     
        self.qvel_pre = np.copy(qvel[6:])
        self.action_raw_pre = np.copy(self.action_raw)

        cur_obs = (cur_obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)

        if (self.epi_len == 0 or self.obs_buf == []):
            for _ in range(self.num_obs_hist):
                for _ in range(self.num_obs_skip):
                    self.obs_buf.append(cur_obs)
                    self.action_buf.append(np.array(self.action_raw, dtype=np.float64))
        
        self.obs_buf[0:self.num_obs_skip*self.num_obs_hist-1] = self.obs_buf[1:self.num_obs_skip*self.num_obs_hist]
        self.obs_buf[-1] = cur_obs
        self.action_buf[0:self.num_obs_skip*self.num_obs_hist-1] = self.action_buf[1:self.num_obs_skip*self.num_obs_hist]
        self.action_buf[-1] = np.array(self.action_raw, dtype=np.float64)

        obs = []
        for i in range(self.num_obs_hist):
            obs.append(self.obs_buf[self.num_obs_skip*(i+1)-1])
        
        act = []
        for i in range(self.num_obs_hist-1):
            act.append(self.action_buf[self.num_obs_skip*(i+1)])

        return np.concatenate([np.array(obs).flatten(), np.array(act).flatten()])



    def step(self, a):
        self.action_raw = np.copy(a)
        a = a * self.action_high
        done_by_early_stop = False
        self.action_cur = a[0:-1] * self.motor_constant_scale
        # print("Action: ", a)
        # a[:] = 0.0

        mocap_cycle_period = self.mocap_data_num* self.mocap_cycle_dt

        local_time = self.time % mocap_cycle_period
        local_time_plus_init = (local_time + self.init_mocap_data_idx*self.mocap_cycle_dt) % mocap_cycle_period
        self.mocap_data_idx = (self.init_mocap_data_idx + int(local_time / self.mocap_cycle_dt)) % self.mocap_data_num
        next_idx = self.mocap_data_idx + 1 
        
        target_data_qpos = np.zeros_like(a)    
        target_data_qpos = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,1:34], self.mocap_data[next_idx,1:34], 0.0, 0.0)

        if (self.perturbation_on):
            if (self.epi_len % (control_freq_scale*2000) == self.perturb_timing):
                impulse = np.random.uniform(50, 120)
                self.pert_duration = control_freq_scale*np.random.randint(25, 250)
                self.magnitude = impulse/(self.pert_duration*self.dt)
                theta = np.random.uniform(0, 2*pi)
                self.new_xfrc[1,0] = self.magnitude * cos(theta)
                self.new_xfrc[1,1] = self.magnitude * sin(theta)
                self.cur_pert_duration = 0
            
            if (self.cur_pert_duration < self.pert_duration):
                self.sim.data.xfrc_applied[:] = self.new_xfrc
                self.cur_pert_duration = self.cur_pert_duration + 1
            else:
                self.sim.data.xfrc_applied[:] = np.zeros_like(self.sim.data.xfrc_applied)

        if (self.spec is not None):
            if (self.epi_len % int(self.spec.max_episode_steps/4) == int(self.spec.max_episode_steps/4)-1):
                self.vel_change_duration = np.random.randint(1, 250)
                self.cur_vel_change_duration = 0  
                self.start_target_vel = np.copy(self.target_vel)
                self.final_target_vel = np.array([np.random.uniform(-0.2, 0.8), np.random.uniform(-0.2, 0.2)])         
            if (self.cur_vel_change_duration < self.vel_change_duration):
                self.target_vel = self.start_target_vel + (self.final_target_vel-self.start_target_vel) * self.cur_vel_change_duration / self.vel_change_duration
                self.cur_vel_change_duration = self.cur_vel_change_duration + 1
            else:
                self.target_vel = np.copy(self.target_vel)

        # self.target_vel[0] = 0.0
        # self.target_vel[1] = 0.0

        # Simulation
        for _ in range(self.frame_skip):
            upper_torque = Kp[12:]*(target_data_qpos[12:] - self.qpos_noise[12:]) + Kv[12:]*(-self.qvel_noise[12:])
            self.action_log.append(self.action_cur)
            if (len(self.action_log) < self.action_delay):
                a_idx = -len(self.action_log)
            else:
                a_idx = -self.action_delay
            self.do_simulation(np.concatenate([self.action_log[a_idx], upper_torque]),1) 
            qpos = self.sim.data.qpos[7:]
            # self.qpos_noise = qpos + np.random.uniform(-0.00001, 0.00001, len(qpos))
            self.qpos_noise = qpos + np.clip(np.random.normal(0, 0.00008 / 3.0, len(qpos)), -0.00008, 0.00008)
            self.qvel_noise = (self.qpos_noise - self.qpos_pre) / self.model.opt.timestep
            self.qpos_pre = np.copy(self.qpos_noise)
            # self.qvel_lpf = lpf(self.qvel_noise, self.qvel_lpf, 1/self.model.opt.timestep, 4.0)

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
        if (np.in1d(geom1, self.self_collision_check_id) * \
                np.in1d(geom2, self.self_collision_check_id)).any():
            done_by_early_stop = True # Self Collision contact

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

        if (qpos[2] < 0.6):
            done_by_early_stop = True        
        if (self.sim.data.get_body_xpos("Neck_Link")[2] < 0.8):
            done_by_early_stop = True

        # self.read_sensor_data()

        basequat = self.sim.data.get_body_xquat("Neck_Link")
        quat_desired = Quaternion(array=[1,0,0,0])  
        baseQuatError = (quat_desired.conjugate * Quaternion(array=basequat)).angle

        if (baseQuatError > 1.0):
            done_by_early_stop = True

        pelvis_quat = Quaternion(array=qpos[3:7])
        pelvis_vel_local = pelvis_quat.conjugate.rotate(qvel[0:3])

        mimic_body_orientation_reward =  0.3 * exp(-13.2*abs(baseQuatError)) 
        qpos_regulation = 0.35*exp(-4.0*(np.linalg.norm(target_data_qpos - qpos[7:])**2))
        qvel_regulation = 0.05*exp(-0.01*(np.linalg.norm(self.init_qvel[6:] - qvel[6:])**2))
        body_vel_reward = 0.3*exp(-3.0*(np.linalg.norm(pelvis_vel_local[0:2] - self.target_vel)**2))
        contact_force_penalty = 0.1*(exp(-0.0005*(np.linalg.norm(self.ft_left_foot) + np.linalg.norm(self.ft_right_foot))))
        torque_regulation = 0.05*exp(-0.01*(np.linalg.norm(self.action_raw[0:-1] * self.action_high[0])))
        torque_diff_regulation = 0.6*(exp(-0.01*(np.linalg.norm((self.action_raw[0:-1] - self.action_raw_pre[0:-1])* self.action_high[0]))))
        qacc_regulation = 0.05*exp(-20.0*(np.linalg.norm(self.qvel_pre - qvel[6:])**2))
        weight_scale = sum(self.model.body_mass[:]) / sum(self.nominal_body_mass)
        force_ref_reward = 0.1*exp(-0.001*(np.linalg.norm(self.ft_left_foot[2] - weight_scale*self.mocap_data[self.mocap_data_idx,34]))) \
                        + 0.1*exp(-0.001*(np.linalg.norm(self.ft_right_foot[2] - weight_scale*self.mocap_data[self.mocap_data_idx,35])))

        if ((self.mocap_data_idx < 300) or \
            (3300 < self.mocap_data_idx and self.mocap_data_idx < 3600) or \
            (1500 < self.mocap_data_idx and self.mocap_data_idx < 2100)): # Double support
            if (right_foot_contact and left_foot_contact):
                foot_contact_reward = 0.2
            else:
                foot_contact_reward = 0.0
        elif (300 < self.mocap_data_idx and self.mocap_data_idx < 1500):
            if (right_foot_contact and not left_foot_contact):
                foot_contact_reward = 0.2
            else:
                foot_contact_reward = 0.0
        else:
            if (not right_foot_contact and left_foot_contact):
                foot_contact_reward = 0.2
            else:
                foot_contact_reward = 0.0
        contact_force_diff_regulation = 0.2*exp(-0.01*(np.linalg.norm(self.ft_left_foot - self.ft_left_foot_pre) + np.linalg.norm(self.ft_right_foot - self.ft_right_foot_pre)))

        force_thres_penalty = 0.0
        if ((abs(self.ft_left_foot[2]) > 1.4 * 9.81 * sum(self.model.body_mass)) or (abs(self.ft_right_foot[2]) > 1.4 * 9.81 * sum(self.model.body_mass))):
            force_thres_penalty = -0.08
        
        force_diff_thres_penalty = 0.0
        if ((abs(self.ft_left_foot[2] - self.ft_left_foot_pre[2]) >  0.2 * 9.81 * sum(self.model.body_mass)) or (abs(self.ft_right_foot[2] - self.ft_right_foot_pre[2]) > 0.2 * 9.81 * sum(self.model.body_mass))):
            force_diff_thres_penalty = -0.05

        reward = mimic_body_orientation_reward + qpos_regulation + qvel_regulation + contact_force_penalty + torque_regulation + torque_diff_regulation + qacc_regulation + body_vel_reward + foot_contact_reward + contact_force_diff_regulation + double_support_force_diff_regulation + force_thres_penalty + force_diff_thres_penalty + force_ref_reward
        
        self.ft_left_foot_pre = np.copy(self.ft_left_foot)
        self.ft_right_foot_pre = np.copy(self.ft_right_foot)

        # self.data_log.append(np.concatenate([self.ft_left_foot, self.ft_right_foot]))
        
        if not done_by_early_stop:
            self.epi_len += 1
            self.epi_reward += reward
            if (self.spec is not None and self.epi_len == self.spec.max_episode_steps):
                print("Epi len: ", self.epi_len)
                # np.savetxt("./result/"+"action_log"+".txt", self.action_log,delimiter='\t')
                # np.savetxt("./result/"+"data_log_torque_250Hz"+".txt", self.data_log,delimiter='\t')

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
                                        contact_force_diff_regulation=contact_force_diff_regulation,
                                        force_thres_penalty=force_thres_penalty,
                                        force_diff_thres_penalty=force_diff_thres_penalty,
                                        force_ref_reward=force_ref_reward))

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
                                        contact_force_diff_regulation=contact_force_diff_regulation,
                                        force_thres_penalty=force_thres_penalty,
                                        force_diff_thres_penalty=force_diff_thres_penalty,
                                        force_ref_reward=force_ref_reward))
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
            force_thres_penalty = 0.0
            force_diff_thres_penalty = 0.0
            force_ref_reward = 0.0
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
                                    contact_force_diff_regulation=contact_force_diff_regulation,
                                    force_thres_penalty=force_thres_penalty,
                                    force_diff_thres_penalty=force_diff_thres_penalty,
                                    force_ref_reward=force_ref_reward))

    def reset_model(self):
        self.time = 0.0
        self.epi_len = 0
        self.epi_reward = 0

        # Dynamics Randomization
        body_mass = np.array(self.nominal_body_mass)
        self.body_mass_noise = np.random.uniform(0.8, 1.2, len(body_mass))
        body_mass = body_mass * self.body_mass_noise
        self.model.body_mass[:]  = body_mass

        body_inertia = np.array(self.nominal_body_inertia)
        body_inertia_noise = np.random.uniform(0.8, 1.2, len(body_inertia))
        body_inertia = np.multiply(body_inertia, body_inertia_noise[:, np.newaxis])
        self.model.body_inertia[:]  = body_inertia
        

        body_ipos = np.array(self.nominal_body_ipos)
        body_ipos_noise = np.random.uniform(0.8, 1.2, len(body_ipos))
        body_ipos = np.multiply(body_ipos, body_ipos_noise[:, np.newaxis])
        self.model.body_ipos[:]  = body_ipos
        
        self.dof_damping = np.array(self.nominal_dof_damping)
        dof_damping_noise = np.random.uniform(0.1, 20.0, len(self.dof_damping))
        self.dof_damping = np.random.uniform(0.1, 3.0, len(self.dof_damping))
        self.model.dof_damping[:]  = self.dof_damping

        self.dof_frictionloss = np.array(self.nominal_dof_frictionloss)
        dof_frictionloss_noise = np.random.uniform(0.8, 1.2, len(self.dof_frictionloss))
        self.dof_frictionloss = self.dof_frictionloss * dof_frictionloss_noise 
        self.model.dof_frictionloss[:]  = self.dof_frictionloss

        # Motor Constant Randomization
        self.motor_constant_scale = np.random.uniform(0.80, 1.2, 12)

        # Delay Randomization
        self.action_delay = np.random.randint(low=5, high=15)

        if (np.random.rand(1) < 0.5):
            self.init_mocap_data_idx = 0
        else:
            self.init_mocap_data_idx = 1800
        init_q_pos = np.copy(self.init_q_desired)
        
        init_qvel = np.copy(self.init_qvel)
        self.set_state(init_q_pos, init_qvel)  

        self.qpos_noise = init_q_pos[7:]
        self.qpos_pre = init_q_pos[7:]
        self.qvel_noise.fill(0)
        self.qvel_lpf.fill(0)

        self.read_sensor_data()
        self.ft_left_foot_pre = np.copy(self.ft_left_foot)
        self.ft_right_foot_pre = np.copy(self.ft_right_foot)

        self.target_vel = np.array([np.random.uniform(-0.2, 0.8), np.random.uniform(-0.2, 0.2)])
        
        self.q_bias = np.random.uniform(-3.14/100.0, 3.14/100.0, 12)
        self.quat_bias = np.random.uniform(-3.14/150.0, 3.14/150.0, 3)
        self.ft_bias = np.random.uniform(-100.0, 100.0, 2)
        self.ft_mx_bias = np.random.uniform(-10.0, 10.0, 2)
        self.ft_my_bias = np.random.uniform(-10.0, 10.0, 2)

        self.action_log = []
        self.data_log = []

        self.action_last.fill(0)
        self.qvel_pre.fill(0)

        self.perturb_timing = np.random.randint(1,control_freq_scale*2000)
        self.obs_buf = []
        self.action_buf = []
        self.action_raw.fill(0.0)
        self.action_raw_pre.fill(0.0)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

    def read_sensor_data(self):
        self.ft_left_foot = self.data.sensordata[self.ft_left_foot_adr:self.ft_left_foot_adr+3]
        self.ft_right_foot = self.data.sensordata[self.ft_right_foot_adr:self.ft_right_foot_adr+3]
        self.torque_left_foot = self.data.sensordata[self.torque_left_foot_adr:self.torque_left_foot_adr+3]
        self.torque_right_foot = self.data.sensordata[self.torque_right_foot_adr:self.torque_right_foot_adr+3]

    def perturbation_start(self):
        self.perturbation_on = True