# TOCABI Reinforcement Learning
This package includes MuJoCo environment for humanoid TOCABI reinforcement learning. TOCABI is a human-sized humanoid developed in Seoul National University [DYROS LAB](http://dyros.snu.ac.kr/). The `main` branch includes an end-to-end torque control RL. 

# **Installation**
## **Prerequisites**
This package requires Python 3.6+ and uses MuJoCo as a simulator (We assume that your graphic driver is installed).
1. Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz).
2. Extract the downloaded file into ~/.mujoco/mujoco210.
    mkdir ~/.mujoco/
    tar -xzvf ~/Downloads/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/


You may need to add environment variables such as to your '~/.bashrc' file:

    export LD_LIBRARY_PATH=/home/user_name/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so

## **Install using pip**

It is recommended to create a new [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/) environment and use it. Refer to [this website](https://sdc-james.gitbook.io/onebook/2./2.2./2.2.1.) to install Anaconda and create a python environment with Anaconda. 

When the python environment is activated, go to the project directory and run
`pip install .`

This should install the `tocabirl` package in site-packages of your python environment. 

# **How to run**
    python run_new.py
You can change a task or robot to be trained by changing parameters of `class args:` in `run_new.py`.

# ** Brief Directory Description ** #
'data': Mean and variance of states
'motions': Reference motion files used during training
'tocabirl': Training-related codes and models. For example, you should modify 'tocabirl/cust_gym/tocabi_walk.py' and 'tocabirl/cust_gym/tocabi_walk_env.py' file to set a environment for the walking task.
'trained model': Trained models. After training is completed, a file such as 'ppo2_DYROSTocabi_2023-11-13 16:28:29.323250' is saved to this directory.

TODO
- Add explaination of arguments
