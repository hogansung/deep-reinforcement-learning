import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.02
        self.eps = 1.0
        self.eps_mul = 0.9995
        self.eps_min = 0.01
        self.gamma = 0.99
        
    def get_probability(self, state):
        return [
            1  - self.eps + self.eps / self.nA if action_id == np.argmax(self.Q[state]) else self.eps / self.nA
            for action_id in range(self.nA)
        ]

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA, p=self.get_probability(state))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Implementation of Q-Learning
        next_action = np.argmax(self.Q[next_state])
        self.Q[state][action] += (1 - self.alpha) * self.Q[state][action] + self.alpha * (self.gamma * reward + self.Q[next_state][next_action])
        self.eps = max(self.eps * self.eps_mul, self.eps_min)