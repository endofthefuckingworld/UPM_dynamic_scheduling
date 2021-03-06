from Environment import Factory
from agent import DQL_agent
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import numpy as np
import pandas as pd 

# problem size
n_jobs = 15
n_machines = 4
n_actions = 6
n_types = 15

# high parameters
update_per_actions = 4
max_memory_length = 30000
max_steps_per_episode = 1000
update_target_network = 12000
max_episode = 10000

performance = np.zeros((30,4))

for n in tqdm(range(30)):
    #agent and enviroment
    agent = DQL_agent(n_jobs, n_machines, n_actions, n_types)
    env = Factory(n)

    episode = 0
    episode_reward_buffer = []
    episode_epsilon_buffer = []
    rewards_epoch = []
    while True:
        episode += 1
        obs = env.reset()
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            agent.frame_count += 1

            # reshape to CNN acceptable
            obs_m1 = obs[0].reshape(1, n_jobs, 5+n_machines, 1)
            obs_m2 = obs[1].reshape(1, n_types, n_types, 1)
            obs_m3 = obs[2].reshape(1, n_machines, 4, 1)

            # choose action
            action = agent.choose_action(
                obs_m1, obs_m2, obs_m3
            )

            # decay prob of exploration
            agent.decay_epsilon()

            next_obs, reward, done, _ = env.step(action)

            episode_reward += reward

            next_obs_m1 = next_obs[0].reshape(1, n_jobs, 5+n_machines, 1)
            next_obs_m2 = next_obs[1].reshape(1, n_types, n_types, 1)
            next_obs_m3 = next_obs[2].reshape(1, n_machines, 4, 1)

            # store training data
            agent.store(
                obs_m1[0], obs_m2[0], obs_m3[0], action, next_obs_m1[0], 
                next_obs_m2[0], next_obs_m3[0], reward, done
            )

            obs = next_obs
            if agent.frame_count % update_per_actions == 0 and len(agent.done_buffer) >= agent.batch_size:
                agent.train_q_network()

            if agent.frame_count % update_target_network == 0:
                agent.update_target_network()

            if len(agent.done_buffer) > max_memory_length:
                agent.remove_old_buffer()

            if done:
                break

        episode_reward_buffer.append(episode_reward)
        if len(episode_reward_buffer) > 40:
            if episode % 100 == 0:
                rewards_epoch.append(np.mean(episode_reward_buffer[-40:]))
                episode_epsilon_buffer.append(agent.epsilon)
                #print('Epoch:{:4d}, mean reward :{}'.format(episode, np.mean(episode_reward_buffer[-20:])))
            # solve condition
            if episode >= max_episode:
                #print('Solve at epoch:{:4d}, reward:{}'.format(episode, np.mean(episode_reward_buffer[-20:])))
                break

            del episode_reward_buffer[0]

    obs = env.reset()
    f= open("Experiment_test/test"+str(n)+".txt","w+")
    while True:
        obs_m1 = obs[0].reshape(1, n_jobs, 5+n_machines, 1)
        obs_m2 = obs[1].reshape(1, n_types, n_types, 1)
        obs_m3 = obs[2].reshape(1, n_machines, 4, 1)

        action = np.argmax(agent.q_network.predict([obs_m1, obs_m2, obs_m3])[0])
        f.writelines('Time :{} | take action:{}'.format(env.env.now, action)+'\r\n')
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        if done:
            f.writelines('weighted delay:'+str(info[0])+'\r\n')
            performance[n] = info
            break
    f.close()
    
    
    
    import matplotlib.pyplot as plt
    plt.style.use("seaborn")

    plt.plot(np.arange(100,len(rewards_epoch)*100+1, 100), rewards_epoch)
    plt.title('Average sum of rewards per episode')
    plt.ylabel('sum of rewards')
    plt.xlabel('epoch')
    plt.savefig('Experiment_train/train'+str(n)+'.png')
    plt.close()
    
p_df = pd.DataFrame(performance)
p_df.to_csv('DQN_performance.csv', index=False)