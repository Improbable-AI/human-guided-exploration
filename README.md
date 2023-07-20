# Human-Guided Exploration (HuGE)

This repository provides the official implementation of the Human Guided Exploration (HuGE) algorithm, as proposed in *Breadcrumbs to the Goal: Goal-Conditioned Exploration from Human-in-the-loop feedback*
The manuscript is available on [arXiv](TODO). See the [project page](https://human-guided-exploration.github.io/HuGE/)

If you use this codebase, please cite

    Marcel Torne, Max Balsells, Zihan Wang, Samedh Desai, Tao Chen, Pulkit Agrawal, Abhishek Gupta. Breadcrumbs to the goal: Goal-Conditioned Exploration from Human-in-the-loop feedback.

## Citation
```
@inproceedings{torne2023huge,
  title={Breadcrumbs to the goal: Goal-Conditioned Exploration from Human-in-the-loop feedback},
  author={Torne, Marcel and Balsells, Max and Wang, Zihan and Desai, Samedh and and Chen, Tao and Agrawal, Pulkit and Gupta, Abhishek},
  booktitle={arxiv},
  year={2023},
  organization={PMLR}
}
```


## Installation Setup
### Install MuJoCo 2.0.0
Download the MuJoCo binaries for [Linux](https://www.roboti.us/download/mujoco200_linux.zip)

Extract the downloaded `mujoco200` directory into `~/.mujoco/mujoco200`.

If you want to specify a nonstandard location for the package, use the env variable `MUJOCO_PY_MUJOCO_PATH`.

### Clone repository

```
git clone git@github.com:Improbable-AI/human-guided-exploration.git
cd human-guided-exploration
```

### Conda Environment

```
conda env create -f environment.yml
conda activate huge
conda develop dependencies
conda develop dependencies/lexa_benchmark
conda develop dependencies/ravens
```

See the [Troubleshooting](https://github.com/Improbable-AI/human-guided-exploration/blob/main/README.md#troubleshooting) section if you are having any issues


## HuGE

```
python launch_main.py --env_name pointmass_rooms --method huge
```

#### Methods available:
- **huge**: official implementation using synthetic human feedback (see section TODO for running HuGE from real human feedback), the synthetic human feedback is generated from reward functions (useful for analysis).
- **oracle**: same algorithm as HuGE but directly querying the reward function for selecting the closest goal instead of learning a goal selector from human feedback.
- **gcsl**: implementation of Goal-Conditioned Supervised Learning (GCSL) Baseline. [1]

#### Benchmarks available:
![alt text](https://github.com/Improbable-AI/human-guided-exploration/blob/main/materials/inline_tasks.png?raw=true)

- **bandu**: Object assembly task, using a Ur5 with a suction gripper it needs to assemble a very specific castle-like structure. Simulated using pybullet and code inspired from [ravens benchmark](https://github.com/google-research/ravens) [2].
- **block_stacking**:  Object assembly task, using a Ur5 with a suction gripper it needs to stack three blocks. Simulated using pybullet and code inspired from [ravens benchmark](https://github.com/google-research/ravens) [2].
- **kitchenSeq**: long-horizon arm manipulation task, Sawyer arm needs to open the slider, microwave and cabinet sequentially to succeed. Simulated using MuJoCo and code inspired from [lexa-benchmark](https://github.com/orybkin/lexa-benchmark) [3].
- **pusher_hard**: object manipulation task, moving puck around walls to reach a goal using a Sawyer arm, simulated using MuJoCo and code inspired from [GCSL](https://github.com/dibyaghosh/gcsl) [1].
- **complex_maze**: long-horizon 2D navigation task, simulated using MuJoCo and code inspired from [GCSL](https://github.com/dibyaghosh/gcsl) [1].
- **pointmass_rooms**: simple 2D navigation task, simulated using MuJoCo and code inspired from [GCSL](https://github.com/dibyaghosh/gcsl) [1].



## Running HuGE from human feedback with our interface

We designed an interface (see below) to collect labels from humans and integrated it with our HuGE algorithm. Next, we provide the instructions to launch the interface and train policies from human feedback using HuGE. 

![alt text](https://github.com/Improbable-AI/human-guided-exploration/blob/main/materials/crowdsourcing_interface.png?raw=true)

First, launch the backend. HuGE will be running on this thread and listening for Human Feedback coming from our interface. This backend is using [FastAPI](https://fastapi.tiangolo.com).

```
ENV_NAME=${env_name} uvicorn launch_huge_human:app --host 0.0.0.0
```

Second, launch the frontend. We designed an interface using [ReactJS](https://react.dev). This will keep presenting the user with two images of achieved states during training and will ask the user to select which one of the two is closer to achieving the target goal. This interface will keep sending the answers to the backend, which will asynchronously train the goal selector as more labels are received. We prepared a docker container to hold and run the interface. Proceed, to launch the frontend:

```
cd interface/frontend
make
make run
```

You should be able to see the interface on port 80 of the machine you are running the interface at. For example, `http://localhost:80` 

### Crowdsourcing experiments

By default, we are running everything in the localhost. However, if you want to run crowdsourcing experiments with annotators from all over the world without needing direct access to your physical machine, we allow you to do that and next we show you how to do it. 

First, change the url of your backend in interface/frontend/src/App.js line 129

You should substitute:
```
const base = "http://localhost:8000"
```
for the public IP adress corresponding to the machine you are running your code at.

Then do as before,

```
cd interface/frontend
make
make run
```

You should be able to see the interface on port 80 of the machine you are running the interface at: `http://${IP_ADDRESS_INTERFACE}:80` 


## Adding your custom environments

### 1. Wrap your custom environment under the `GymGoalEnvWrapper`

The GymGoalEnvWrapper class is defined at `huge/envs/gymenv_wrapper.py`.

We provide an example of a simple environment wrapped under this class in `huge/envs/simple_example.py`

### 2. Add your environment in __init__.py file
Next, you must name and add your environment on the `creat_env` function in `huge/envs/__init__.py`

### 3. Optional: setting hyperparameters
Add an entry corresponding to your new environment on the `config.yaml` file for specifying custom parameters that you want to change different from the default ones.

## Troubleshooting

#### GLIBCXX error
If you get any errors like the following:

```
ImportError: $CONDA_PATH/lib/python3.6/site-packages/torch/lib/../../../../libstdc++.so.6: version `GLIBCXX_3.4
.29' not found (required by /lib/x86_64-linux-gnu/libOSMesa.so.8)
```

delete the `libstdc++.so.6` file:

```
rm $CONDA_PATH/lib/python3.6/site-packages/torch/lib/../../../../libstdc++.so.6
```
#### ParamSpec error

If you get the following error:
```
ImportError: cannot import name 'ParamSpec'
```

do the following:

```
pip uninstall typing_extensions
pip uninstall fastapi
pip install --no-cache fastapi
```

## Development Notes

The directory structure currently looks like this:

- huge (Contains all code)
    - envs (Contains all environment files and wrappers)
    - algo (Contains all HuGE code)
        - huge.py (implements high-level algorithm logic, e.g. data collection, policy update, evaluate, save data)
        - buffer.py (The replay buffer used to *relabel* and *sample* (s,g,a,h) tuples
        - networks.py (Implements neural network policies.)
        - variants.py (Contains relevant hyperparameters for HuGE)
    - baselines (Contains implementations of the baselines presented in the paper)
- doodad (We require this old version of doodad)
- dependencies (Contains other libraries like rlkit, rlutil, room_world, multiworld, etc.)

Please file an issue if you have trouble running this code.

## References
[1] D. Ghosh, A. Gupta, J. Fu, A. Reddy, C. Devin, B. Eysenbach, and S. Levine. Learning to reach
goals without reinforcement learning. CoRR, abs/1912.06088, 2019

[2] A. Zeng, P. Florence, J. Tompson, S. Welker, J. Chien, M. Attarian, T. Armstrong, I. Krasin, D. Duong, V. Sindhwani, and J. Lee. Transporter networks: Rearranging the visual world for robotic manipulation. Conference on Robot Learning (CoRL), 2020.

[3] R. Mendonca, O. Rybkin, K. Daniilidis, D. Hafner, and D. Pathak. Discovering and achieving goals via world models. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 24379–24391, 2021
