import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, eps = 0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 1
        self.alpha = 0.1
        self.gamma = 1
        self.nEpisode = 1


    def update_Q_sarsamax(self, state, action, reward, next_state=None):
        """Returns updated Q-value for the most recent experience."""
        current = self.Q[state][action]  # estimate in Q-table (for current state, action pair)
        # get value of state, action pair at next time step
        # Qsa_next = Q[next_state][next_action] if next_state is not None else 0    
        Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0  # value of next state
        target = reward + (self.gamma * Qsa_next)               # construct TD target
        new_value = current + (self.alpha * (target - current)) # get updated value
        return new_value
    
    def epsilon_greedy(self, state):
        """Selects epsilon-greedy action for supplied state.
        
        Params
        ======
            Q (dictionary): action-value function
            state (int): current state
            nA (int): number actions in the environment
            eps (float): epsilon
        """
        if random.random() > self.eps: # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:                     # otherwise, select an action randomly
            return np.random.choice(self.nA)

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.epsilon_greedy(state)

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
        if done :
            self.nEpisode += 1
            self.eps = 1.0 / self.nEpisode + 0.0001

        self.Q[state][action] = self.update_Q_sarsamax(state, action, reward, next_state)