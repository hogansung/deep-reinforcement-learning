# Project1 : Navigation
___

## Problem definition
In this project, the goal is to retrieve as many _good_ bananas (+1), at the same time, avoiding the bad bananas (-1)
and walls, in a large, square world.

### Number of states
There are overall `37` states, which can be further broken down into the detection results from `7` rays, each with `5` 
states, as well as `2` velocity states, i.e. (`37 = 7 * 5 + 2`). For more details, please refer to 
[the source of the information](https://github.com/Unity-Technologies/ml-agents/issues/1134#issuecomment-417497502).

#### Ray states
There are overall `7` rays, which cover the angles of `[20, 90, 160, 45, 135, 70, 110]`, correspondingly. Notice that 
the angle of `90` locates directly in front of the agent.

Each ray contains `5` states. The first `4` states indicate the existence of good bananas, walls, bad bananas, and 
other agents, respectively. The last state indicates the distance to the object, as a fraction of the ray length.

#### Velocity states
There are two velocity states. The first one indicates the velocity to the left or to the right, ranging from `-20` to 
`+20`. The second one indicates the forward or backward velocity, ranging from `-20` to `+20`.

### Number of actions
There are overall `4` actions, where `0` indicates moving forward, `1` indicates moving backward, `2` indicates turning 
left, and `3` indicates turning right.

## Folder layout
- p1_navigation_submission
  - Navigation.ipynb
  - README.md
  - Report.pdf
  - agent.py
  - model.py
  - qnetwork_local.pt
  - replay_buffer.py
  - scaler.py

## Contents
### Navigation.ipynb
The main jupyter notebook that provides the train and inference functionality. One can execute it without additional
dependency.

Notice that there is a caching mechanism for the model file `qnetwork_local.pt`. If `OVERWRITE` is set to `False` and
that model file does exist, model state_dict will be loaded into the agent instance.

### README.md
A readme file that provides high-level overview of the submission folder and how to execute the code.

### Report.pdf
A report that contains the detailed logic breakdown, including how the model states are determined, the selected
hyper-parameters, and the final performance report. 

### agent.py
A Python module that contains the class `Agent`, which holds the action decision logic and the learning logic. **Its 
logic is mostly copied from the DQN practice, except for some code reorganizations and typing highlights.**

### model.py
A Python module that contains an extended nn.Module class, `Model`, which forwards the provided states in order to 
inference the actions. **Its logic is mostly copied from the DQN practice, except for some code reorganizations and 
typing highlights.**

### qnetwork_local.pt
A snapshot of qnetwork_local model values using `torch.save()`.

### replay_buffer.py
A Python module that contains the class `ReplayBuffer`, which keeps the history of `S-A-R-A` tuples that can later be 
sampled for further training. **Its logic is mostly copied from the DQN practice, except for some code reorganizations 
and typing highlights.**

### scaler.py
A Python module that contains the class `Scaler`, which is used to _smartly_ convert a given value into the correct 
bucket index. This is mostly used for the encoding of distance and velocity features.
