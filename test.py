from collections import defaultdict

import matplotlib.pyplot as plt #drawing plots
from matplotlib.patches import Patch #draw shapes
import numpy as np #data and array manipulation
import seaborn as sns

from tqdm import tqdm #progress bar
import gymnasium as gym

env = gym.make('Blackhack-v1', sab=True, render_mode="rgb_array") 

#reset the environment to get the first observation
done = False
observation,info = env.reset()

#observation = (16,9,False)
#my hand, dealers hand, do I have a usable ace

#sample a random action from all valid actions
action = env.action_space.sample()
#action = 1

#execute the action in our environment and recieve info after taking the step
observation,reward,terminated,truncated,info = env.step(action)

#observation = (24,10,False)
#reward = -1.0
#terminated = true
#truncated = false

class BlackjackAgent:
    def __init__(
            self,
            learning_rate:float,
            initial_epsilon:float,
            epsilon_decay:float,
            final_epsilon:float,
            discount_factor:float = 0.95
    ):
        '''
        Initialize the agent with an empty dictionary of state-action values (q_values), a learning rate, and an epsilon
        '''
        self.q_values = defaultdict(lambda:np.zeros(env.action_space.n))
        
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        '''
        Returns the best action with a probability of (1-epsilon)
        Otherwise returns a random action to ensure exploration
        '''
        #retuns random action
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        #returns best action (acts greedily)
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool]
    ):
        '''Updates the Q-values after an action'''
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]

        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * temporal_difference 