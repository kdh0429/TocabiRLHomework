# TOCABI Reinforcement Learning
This package includes MuJoCo environment for humanoid TOCABI reinforcement learning. TOCABI is a human-sized humanoid developed in Seoul National University [DYROS LAB](http://dyros.snu.ac.kr/). The `master` branch includes an end-to-end torque control RL, and `position` branch includes a position control RL with a PD controller. 

# **Installation**
## **Prerequisites**
This package requires Python 3.6+ and uses MuJoCo as a simulator.
1. Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz).
2. Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.

You may need to add environment variables such as:

    export LD_LIBRARY_PATH=/home/user_name/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so

## **Install using pip**

Go to the project directory with the python environment activated.
Then run
`pip install .`

This should install the `tocabirl` package in site-packages. 

# **How to run**
    python run_new.py
You can change a task or robot to be trained by changing parameters of `class args:` in `run_new.py`.


TODO: 
- Detailed documentation (directory, etc...)
- Add explaination of arguments
