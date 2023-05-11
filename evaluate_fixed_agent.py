import gymnasium as gym 
from agent import Agent 
from torch.utils.tensorboard import SummaryWriter
from monkeyqmazediscrete_v3 import MonkeyQmazeDiscreteV3
from monkeyqmazediscrete_v0 import MonkeyQmazeDiscreteV0
import numpy as np 

def evaluate_fixed_agent(episode_num, agent=Agent, writer=SummaryWriter, goal_location_idx = 0, game_ver = 0):
    if game_ver == 0:
        env = MonkeyQmazeDiscreteV0()
    elif game_ver == 3:
        env = MonkeyQmazeDiscreteV3()

    state, info = env.reset(reward_idx=goal_location_idx) 

    test_episode = 100

    reward_vector = np.zeros(test_episode)
    length_vector = np.zeros(test_episode)

    for i_episode in range(test_episode):

      state, info = env.reset(reward_idx=goal_location_idx) 
        
      if agent.agent_memory_based:
          agent.small_reward_memory_reset() 

      done = False
      truncated = False 
      total_length = 1 
      total_reward = 0 

      while not (done or truncated):
          action = agent.act(state) 
          next_state, reward, done, truncated, info = env.step(action)

          total_reward += reward 
          total_length += 1 

          state = next_state 
          
          if agent.agent_memory_based:
              agent.record_small_reward_memory(reward) 


      reward_vector[i_episode] = total_reward 
      length_vector[i_episode] = total_length 
        
    writer.add_scalar("simulated agent reward", np.mean(reward_vector), episode_num)
    writer.add_scalar("simulated agent length", np.mean(length_vector), episode_num)
    