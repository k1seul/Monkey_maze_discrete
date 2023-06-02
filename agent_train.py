import os 
import time 
import subprocess
from monkeyqmazediscrete import MonkeyQmazeDiscrete
from monkeyqmazediscrete_v0 import MonkeyQmazeDiscreteV0
from monkeyqmazediscrete_v0_cheat import MonkeyQmazeDiscreteV0Cheat 
from agent_model_memory import Agent
from torch.utils.tensorboard import SummaryWriter 
import pickle 
from save_data import Save_data 
from shannon_info import shannon_info
import numpy as np 
from evaluate_fixed_agent import evaluate_fixed_agent

def agent_train(uniform_sample=True,TD_sample = False, sample_var_n = 100, 
    game_version = 1 , TD_switch = False, model_based=False,
    agent_memory_based=False,
    pre_train = False):
    if TD_sample:
        uniform_sample = False
    game_name = 'Monkey_Qmaze' + str(game_version)
    run_time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    log_dir = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop/tensorboard_Data')
    data_dir = os.path.join(os.path.join(os.path.expanduser('~'), 'Desktop/model_Data' + game_name + '/' + 
                                         "Uniformed_sample_" + str(uniform_sample) +'_'+ 'TD_switch_'+ str(TD_switch) + 
                                         '_model_based_' + str(model_based) + '/' + run_time ))
    os.makedirs(data_dir)
    port = 6983 ## random.randint(6000, 7000)
    subprocess.Popen(f"tensorboard --logdir={log_dir} --port={port} --reload_multifile=true", shell=True)

    log_dir = (log_dir + '/' + game_name + '_' + "Uniformed_sample_" + 
        str(uniform_sample) +'_TD_switch'+ str(TD_switch)  + 
        '_' + 'model_based_' + str(model_based) + '_' + 'memory_based_' + str(agent_memory_based)) + '_' + str(run_time)

    if game_version == 1:
        env = MonkeyQmazeDiscrete()
    elif game_version ==9:
        env = MonkeyQmazeDiscreteV0Cheat()
    else:
        env = MonkeyQmazeDiscreteV0() 

    state_size = env.state_n 
    action_size = env.action_n
    simulation = False

    hidden_size = 512 
    learning_rate = 0.001 
    memory_size = 10000 
    batch_size = 256 
    gamma = 0.99 

    agent = Agent(state_size= state_size,
                  action_size= action_size,
                  hidden_size= hidden_size,
                  learning_rate= learning_rate,
                  memory_size= memory_size,
                  batch_size= batch_size,
                  gamma= gamma,
                  model_based=model_based,
                  agent_memory_based=agent_memory_based)
    
    agent.sample_var_n = sample_var_n 

    
    # Set up TensorBoard output
    writer = SummaryWriter(log_dir=log_dir)

    agent_path_all = {} 

    num_episode = 500  
    reward_num = 9

    if agent_memory_based:
        agent.init_memory(agent_memory_size=2) 
        agent.init_goal_memory_dict() 
        agent.goal_memory2dict() 

    

    from monkeyqmazediscrete_v0_explore import MonkeyQmazeDiscreteV0Ex


    if agent_memory_based and pre_train :
        """
        pretraining of the memory based agent"""
        pre_env = MonkeyQmazeDiscreteV0Ex() 
        state, info = pre_env.reset() 

        for i_episode in range(0, 1000): 
            state, info = pre_env.reset(reward_idx = 1 ) 
            done = False 
            truncated = False 
            total_length = 1
            total_reward = 0 
           
            while not(done or truncated):
                action = agent.act(state)
                next_state, reward, done, truncated, info = pre_env.step(action)
        

                total_reward += reward 
                total_length += 1



                agent.remember(state, action, reward, next_state, done)
                agent.replay(uniformed_sample= uniform_sample, TD_sample = TD_sample )
                

                state = next_state

                if done:
                    agent.decay_epsilon()

            writer.add_scalar("pre train reward", total_reward, i_episode) 
            print(f"pre train network reward of {total_reward} took episode {total_length}")
            print(agent.guess_goal)




    agent.epsilon = 1.0 
        

    for reward_idx in range(0, reward_num):

        for i_episode in range(reward_idx*num_episode, (reward_idx + 1)*num_episode):
            state, info = env.reset(reward_idx=reward_idx)

        
            
            done = False
            truncated = False 
            total_length = 1
            total_reward = 0 
            state_trajectories = [] 
            action_trajectories = [] 
            while not(done or truncated):
                if agent_memory_based and np.array_equal([0,0], agent.guess_goal):
                   agent.goal_select_from_memory() 
                   
                
               

                action = agent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
        

                total_reward += reward 
                total_length += 1



                agent.remember(state, action, reward, next_state, done)
                
                if agent_memory_based & (done or truncated):
                   agent.record_goal_memory(next_state, reward)


                state = next_state

                if TD_switch: 

                    TD_sample = True if i_episode >= num_episode and i_episode % num_episode < 20 else False 

                agent.replay(uniformed_sample= uniform_sample, TD_sample = TD_sample )

                if model_based and i_episode > num_episode and i_episode % num_episode < 20 and i_episode % num_episode > 0 : 
                    agent.model_simulate(state, 10)


                state_trajectories.append(state)
                action_trajectories.append(action)

            if done:
                agent.decay_epsilon()
            
            

            shannon_value = shannon_info(state_trajectories, action_trajectories, env.action_n)
            
            if agent_memory_based:

                agent.goal_memory2dict()
        


            writer.add_scalar("reward", total_reward, i_episode) 
            writer.add_scalar("length", total_length, i_episode)
            writer.add_scalar("reward_rate", total_reward/total_length, i_episode)
            writer.add_scalar("epsilion", agent.epsilon, i_episode)
            writer.add_scalar("shannon_value:", shannon_value, i_episode)


            ## simulate agent in fixed network parameter and record mean length and reward of 100 trials 
            if simulation: 
                evaluate_fixed_agent(i_episode, agent=agent, writer=writer, goal_location_idx=reward_idx, game_ver=game_version)

            if model_based:
                writer.add_scalar("model_loss:", agent.model_loss, i_episode)

            Save_data(agent.q_network, i_episode, save_rate = 250 , name= data_dir + '/' + 'Q_network') 

            print("Episode: {}, total_reward: {:.2f}, epsilon: {:.2f}, length: {}".format(i_episode, total_reward, agent.epsilon, total_length))



    # Close the environment and writer
    env.close()
    writer.close()
