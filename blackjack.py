from collections import defaultdict

import matplotlib.pyplot as plt #drawing plots
from matplotlib.patches import Patch #draw shapes
import numpy as np #data and array manipulation
import seaborn as sns

from tqdm import tqdm #progress bar
import gymnasium as gym

env = gym.make('Blackjack-v1', sab=True, render_mode="rgb_array") 

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
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)

# Hyparameters
learning_rate = 0.01 #0.001
n_episodes = 100_000 #10_000_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

agent = BlackjackAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon
)

# Training
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    #play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated
        obs = next_obs
    
    agent.decay_epsilon()

# Visualizing Training
rolling_length = 500
fig, axs = plt.subplots(ncols = 3, figsize = (12,5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode Length")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    /rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()

# Visualising the policy
def create_grids(agent, usable_ace = False):
    '''Create value and policy grid given an agent'''
    # convert state-action values to state values and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up cards
        np.arange(12, 22),
        np.arange(1, 11),
    ) 

    # create value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid

def create_plots(value_grid, policy_grid, title: str):
    '''creates a plot using value and polcy grid'''
    # create new figure with 2 subplots (left:state vales, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize = 16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=2,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidths=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig

# state values & policy with usable ace (ace counts as 11)
value_grid, policy_grid = create_grids(agent, usable_ace=True)
fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
plt.show()

# state values & policy without usable ace (ace counts as 1)
value_grid, policy_grid = create_grids(agent, usable_ace=False)
fig1 = create_plots(value_grid, policy_grid, title="Without usable ace")
plt.show()

env.close()