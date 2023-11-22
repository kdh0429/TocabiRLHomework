import os 
import gym

# Use our custom PPO
from tocabirl.ppo import PPO # from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from tocabirl.vec_normalize_clipped_var import VecNormalizeClippedVar

# Import our custom gym environment, which in the init will generate the gym environment
from tocabirl.cust_gym import tocabi_squat, terrain_generator

import numpy as np
import datetime

from typing import Callable
from dataclasses import dataclass


@dataclass 
class args:
   run_type = "train" # "train", "fine_tune", "enjoy",
   n_cpu = 1 if run_type == "enjoy" else 8
   task = 'Walk' # "Stand Still", "Squat", "Walk", "AtlasWalk" 
   render = False # Available when n_cpu = 1 or run_type = "enjoy"
   n_steps = int(2*8192/n_cpu)
   batch_size = 128 
   total_timesteps = 80000000
   initial_lr = 1e-5
   final_lr = 2e-7
   env = None
   play_model = "ppo2_DYROSTocabi_2023-11-21 10:55:55.288059" # tranined model name

def linear_schedule(initial_lr: float, final_lr:float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining*initial_lr + (1-progress_remaining)*final_lr

    return func

def setup_env():
   ''' TODO: Add argument to load specific config '''
       
   def make_env(env_id, rank, seed=0):
      def _init():
         env = gym.make(env_id)
         env.seed(seed + rank)
         return env
      set_random_seed(seed)
      return _init

   if args.task == "Stand Still": env_id = "tocabi-stand-still-v0"
   elif args.task=="Squat":       env_id = "tocabi-squat-v0"
   elif args.task=="Walk":       env_id = "tocabi-walk-v0"
   elif args.task=="Run":       env_id = "tocabi-run-v0"
   elif args.task=="AtlasWalk":       env_id = "atlas-walk-v0"
   elif args.task=="AtlasRun":       env_id = "atlas-run-v0"
   else:                          env_id = "tocabi-walk-v0"

   # Use single environment if only using 1 core
   if args.n_cpu == 1:
      env = gym.make(env_id)
      env = DummyVecEnv([lambda: env])
   else:
      env = SubprocVecEnv([make_env(env_id, i) for i in range(args.n_cpu)])

   args.env = env

def train():
   _dir = os.path.join("trained_model",args.task)
   file_name = os.path.join(_dir,f"ppo2_DYROSTocabi_{str(datetime.datetime.now())}")
   
   try: os.mkdir(_dir)
   except: pass 

   model = PPO('FixedStddevMlpPolicy', args.env, n_steps=args.n_steps, 
               batch_size=args.batch_size, render=args.render, 
               learning_rate=linear_schedule(initial_lr=args.initial_lr, final_lr=args.final_lr), task=args.task)

   model.learn(total_timesteps=args.total_timesteps)
   save_model(model, file_name=file_name)

def enjoy():
   file_name = os.path.join("trained_model", args.task, args.play_model)

   model = PPO.load(file_name, env=args.env)

   try: os.mkdir("./result")
   except: pass 

   model.policy.to("cpu")
   for name, param in model.policy.state_dict().items():
      name= name.replace(".","_")
      weight_file_name = "./result/" + name + ".txt"
      np.savetxt(weight_file_name, param.data)

   #Enjoy trained agent
   obs =  np.copy(args.env.reset())
   epi_reward = 0
   while True:
      action, _states = model.predict(obs, deterministic=True)
      obs, rewards, dones, info = args.env.step(action)
      args.env.render()
      epi_reward += rewards      
      
      if dones:
         print("Episode Reward: ", epi_reward)
         epi_reward = 0

def save_model(model, file_name):
   model.save(file_name)
   del model
   del args.env


def main():
   setup_env()

   # Run a new training
   if (args.run_type is "train"):
      train()
   # Fine-tune a trained model
   elif (args.run_type is "fine_tune"):
      fine_tune()
   # Load and enjoy
   else:
      enjoy()
      
if __name__ == '__main__':
    main()