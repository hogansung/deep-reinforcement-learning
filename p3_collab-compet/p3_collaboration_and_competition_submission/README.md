# Project3 : Collaboration and Competition

___

## Problem definition

In this project, two agents control rackets to bounce a ball over a net, and the goal is to have them play against each
other as many rounds as possible. In other words, both agents should make sure the other one can catch the ball, instead
of letting a ball hit the ground or out of bounds.

### Number of agents

In this multi-agent project, there are only `2` agents. One is controlling the blue racket, and the other one is
controlling the red racket.

### Number of states

There are overall `24` states for each agent, which are formed of a stack of `3` continuous frames. In each frame, `8`
variables are corresponding to the _position_ and _velocity_ of the racket in a 2D space.

### Number of actions

There is a vector of `2` actions for each agent, which are corresponding to _moving toward (or away from) the net_ as
well as _jumping_. Notice that each value in the vector should be a number between `-1` and `+1`.

## Getting Started

One should follow
the [instructions from Udacity DRL ND](https://github.com/udacity/deep-reinforcement-learning#dependencies) to figure
out all the dependencies.

### Additional dependencies

Given that the DRL ND environment is pretty old and is still on Python 3.6, one needs to explicitly install
package `nptyping` in order to annotate numpy array properly.

```bash
pip install nptyping
```

If one is using Python 3.8+ versions, numpy array annotation is already included as part of the numpy library. For more
details, please refer to [Typing (numpy.typing)](https://numpy.org/devdocs/reference/typing.html).

### Modified version of Tennis environment

In order to execute the Jupyter notebook correctly, one of the prebuilt simulators is required to be installed, based on
the operating system.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (
  32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (
  64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

The file needs to placed in the root directory of the repository and unzipped.

## Folder layout

- p3_collaboration_and_competition_submission
    - README.md
    - Report.pdf
    - Tennis.ipynb
    - agent_manager.py
    - agent.py
    - maddpg_actor_local_0.pt
    - maddpg_actor_local_1.pt
    - maddpg_critic_local_0.pt
    - maddpg_critic_local_1.pt
    - model.py
    - ounoise.py
    - replay_buffer.py

## Contents

### README.md

A readme file that provides high-level overview of the submission folder and how to execute the code.

### Report.pdf

A report that contains the detailed logic breakdown, including how the algorithm works, how I debug my implementations,
the final performance reports, and some possible future works.

### Tennis.ipynb

The main jupyter notebook that provides the _training_ functionality. One can execute it without additional dependency.

Notice that there is a caching mechanism for the model file `maddpg_actor_local_{agent_idx}.pt`
and `maddpg_critic_local_{agent_idx}.pt`. If `OVERWRITE` is set to `False` and all the model files do exist,
`state_dict` for either the actor or critic model will be properly loaded from cache files.

### agent_manager.py

A Python module that contains the class `AgentManager`, which holds the action decision logic and the learning logic. I
referred partially to the previous _MADDPQ_ practice for my implementation. However, given the fact that there is no
such a third-party observation, as the one provided in the _MADDPQ_ practice, I have to modify the algorithm a bit to
fit the use case in this project.

### agent.py

A Python module that contains the class `Agent`, which wraps the action decision logic (without gradient) and the
soft-model update logic. The class methods can be arguably moved into the `AgentManager` class.

### maddpg_actor_local_{agent_idx}.pt

A snapshot of `actor_local` model values for agent `agent_idx` using `torch.save()`.

### maddpg_critic_local_{agent_idx}.pt

A snapshot of `critic_local` model values for agent `agent_idx` using `torch.save()`.

### model.py

A Python module that contains two extended `nn.Module` classes, `Actor` and `Critic`. The former network takes in a
state in order to figure out the _best_ action; whereas the latter network takes both the concatenated state and action
vectors across multiple agents and outputs the proper Q-value of that combination.

Notice that even if the critic network could theoretically be shared across agents, I decide to learn two models
separately as my first trial.

### ounoise.py

A Python module that contains the class `OUNoise`, whose goal is to introduce some variances to the action decisions.

### replay_buffer.py

A Python model that contains the class `ReplayBuffer`, which is used to store limited amount of historical tuples of
(S, A, R, S') in a deque structure. With this, the sampled data could be reused and the overall sampling cost is hugely
reduced.