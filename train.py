from Environment import Factory
from agent import PPO_agent
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from tqdm import tqdm
import tensorflow as tf

epochs = 10
steps_per_epoch = 1000
n_jobs = 8
n_machines = 3
n_actions = 6
n_types = 3

env = Factory()
agent = PPO_agent(steps_per_epoch, n_jobs, n_machines, n_types, n_actions)
obs, episode_return = env.reset(), 0


for epoch in range(epochs):
    sum_return = 0
    num_episodes = 0

    for t in range(steps_per_epoch):
        obs_m1 = obs[0].reshape(1, n_jobs, 6, 1)
        obs_m2 = obs[1].reshape(1, n_types, n_types, 1)
        obs_m3 = obs[2].reshape(1, n_machines, 4, 1)
        
        value_t = agent.critic([obs_m1, obs_m2, obs_m3])
        logits, action = agent.sample_action(obs_m1, obs_m2, obs_m3)
        obs_new, reward, done, _ = env.step(action[0].numpy())
        episode_return += reward
        

        # Get the value and log-probability of the action
        value_t = agent.critic([obs_m1, obs_m2, obs_m3])
        logprobability_t = agent.logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        agent.store(obs_m1, obs_m2, obs_m3, action, reward, value_t, logprobability_t)

        # Update the observation
        obs = obs_new

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if done else agent.critic(
                [obs_m1.reshape(1, n_jobs, 6, 1), obs_m2.reshape(1, n_types, n_types, 1), obs_m3.reshape(1, n_machines, 4, 1)]
            )
            agent.finish_trajectory(last_value)
            sum_return += episode_return
            num_episodes += 1
            observation, episode_return, episode_length = env.reset(), 0, 0
            
    agent.train()
    # Print mean return and length for each epoch
        
    print(
        f" Epoch: {epoch + 1}. Mean Return: {round(sum_return/num_episodes,2)}"
    )
    


obs = env.reset()

while True:
    obs_m1 = obs[0].reshape(1, n_jobs, 6, 1)
    obs_m2 = obs[1].reshape(1, n_types, n_types, 1)
    obs_m3 = obs[2].reshape(1, n_machines, 4, 1)
    
    action = agent.choose_action(obs_m1, obs_m2, obs_m3)
    next_obs, reward, done, info = env.step(action)
    obs = next_obs
    if done:
        print('Number of late submissions:',info)
        break
  

