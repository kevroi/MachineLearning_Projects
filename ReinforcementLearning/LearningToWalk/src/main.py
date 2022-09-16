import gym
from gym.envs import box2d
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
# from pyvirtualdisplay import Display
# from IPython import display as disp
import copy

from td3_fork import Agent


PLOT_INTERVAL = 10 # update the plot every N episodes
VIDEO_EVERY = 50 # videos can take a very long time to render so only do it every N episodes

# select gpu if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Prepare environment and wrap it to capture video
env = gym.make("BipedalWalker-v3")
# env = gym.wrappers.Monitor(env, "./video", video_callable=lambda ep_id: ep_id%VIDEO_EVERY == 0, force=True)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

print('The environment has {} observations and the agent can take {} actions'.format(obs_dim, act_dim))
print(env.observation_space.shape)
print('The device is: {}'.format(device))

# Training Loop
# training loop for FORK

seed = 42
torch.manual_seed(seed)
# env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)


agent = Agent(alpha=0.001, beta=0.001, delta=0.001, input_shape=env.observation_space.shape,
				n_actions=env.action_space.shape[0], sys_weight=0.5, sys_weight2=0.4,
                sys_threshold=0.02, tau=0.005,  env=env,
				name="BipedalWalker-v3")

best_score = env.reward_range[0]
score_history = []
plot_data = []
log_f = open("agent-log.txt","w+")
n_games = 1010


device = agent.get_device()
print("Device is : ", device)

for episode in range(n_games):
    obs = env.reset()
    done = False
    ep_reward = 0

    while not done:
        action = agent.choose_action(obs)
        obs_, reward, _, done, info = env.step(action)
        agent.store_memory(obs, obs_, action, reward, done)
        agent.learn()
        obs = obs_
        ep_reward += reward

        agent.obs_lower_bound = np.amin(obs) if agent.obs_lower_bound > np.amin(obs) else agent.obs_lower_bound
        agent.obs_upper_bound = np.amax(obs) if agent.obs_lower_bound < np.amax(obs) else agent.obs_upper_bound
        agent.rew_lower_bound = (reward) if agent.rew_lower_bound > reward else agent.rew_lower_bound
        agent.rew_upper_bound = (reward) if agent.rew_upper_bound < reward else agent.rew_upper_bound
        # env.render()

		# env.render()

    score_history.append(ep_reward)
    avg_score = np.mean(score_history[-100:])

    log_f.write('episode: {}, reward: {}\n'.format(episode, ep_reward))
    log_f.flush()


    if episode % PLOT_INTERVAL == 0:
        plot_data.append([episode, np.array(score_history).mean(), np.array(score_history).std()])
        reward_list = []
        # plt.rcParams['figure.dpi'] = 100
        plt.plot([x[0] for x in plot_data], [x[1] for x in plot_data], '-', color='tab:grey')
        plt.fill_between([x[0] for x in plot_data], [x[1]-x[2] for x in plot_data], [x[1]+x[2] for x in plot_data], alpha=0.2, color='tab:grey')
        plt.xlabel('Episode number')
        plt.ylabel('Episode reward')
        plt.title(f'Episode: {episode}, Reward: {ep_reward}')
        # plt.show()
        plt.savefig('Reward.png', dpi=300)
        # files.download('Reward.png')
        # files.download('agent-log.txt')

    ep_reward = 0