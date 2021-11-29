# Project2 : Continuous Control

___

## Problem definition

In this project, the goal is to make sure the double-jointed arm(s) can move toward the target locations. Depending on
the problem setting, there might be one arm or twenty arms moving at the same time in the environment.

### Number of states

There are overall `33` states, which involve the positions, rotations, velocities, and angular velocities of the arm(s).

### Number of actions

There is a vector `4` actions, which are corresponding to torque applicable to the two joints. Notice that each value in
the vector should be a number between `-1` and `+1`.

## Getting Started

One should follow
the [instructions from Udacity DRL ND](https://github.com/udacity/deep-reinforcement-learning#dependencies) to figure
out all the dependencies.

In order to execute the Jupyter notebook correctly, one of the prebuilt simulators is required to be installed, based on
the operating system and desired simulating environment.

### Version 1: One (1) Agent

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (
  64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

### Version 2: Twenty (20) Agents

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

The file needs to placed in the root directory of the repository and unzipped. Notice that the unzipped name might need
to slightly modified in order to execute the Jupyter notebook successfully.

## Folder layout

- p2_continuous_control_submission
    - Continuous_Control.ipynb
    - README.md
    - Report.pdf
    - agent_manager.py
    - ddpg_actor_local.pt
    - ddpg_critic_local.pt
    - model.py

## Contents

### Continuous_Control.ipynb

The main jupyter notebook that provides the train functionality. One can execute it without additional dependency.

Notice that there is a caching mechanism for the model file `ddpg_actor_local.pt` and `ddpg_critic_local.pt`.
If `OVERWRITE` is set to `False` and that model file does exist, model `state_dict` will be loaded into the agent
manager instance.

### README.md

A readme file that provides high-level overview of the submission folder and how to execute the code.

### Report.pdf

A report that contains the detailed logic breakdown, including how the learning works, the differences between using one
and multiple agents, and the final performance report.

### agent.py

A Python module that contains the class `AgentManager`, which holds the action decision logic and the learning logic. **
Its logic is mostly copied from the two DDPQ practices `ddpg-bipedal` and `ddpg-pendulum`, except for some code
reorganizations and typing highlights.** Before I submitted my work, I also referred to
hortovanyi's [implementation](https://github.com/hortovanyi/DRLND-Continuous-Control) on GitHub, which enlightens me to
train the networks with twenty agents instead of one.

### ddpg_actor_local.pt

A snapshot of `actor_local` model values using `torch.save()`.

### ddpg_critic_local.pt

A snapshot of `critic_local` model values using `torch.save()`.

### model.py

A Python module that contains two extended `nn.Module` classes, `Actor` and `Critic`. The former network takes in a
state in order to figure out the _best_ action; whereas the latter network takes both the state and action and outputs
the proper Q-value of that combination.
