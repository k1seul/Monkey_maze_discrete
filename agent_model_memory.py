from network_module import * 
import torch
import torch.nn as nn 
import torch.optim as optim 
from collections import deque 
import random 
import numpy as np 
import scipy.stats as stats
from model_solver_agent import ModelSolver


class Agent():
    """
    batch learning dqn agent with selection of network structure
    """
    def __init__(self, state_size, action_size, hidden_size, learning_rate, 
                 memory_size, batch_size, gamma,
                 policy_network= 'Q_network', model_based= False, agent_memory_based= False, agent_memory_size = 2):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("currently running the calculation on " + str(self.device))
        
        self.gpu_usage = True if torch.cuda.is_available() else False

        self.alpha = 0.9
        self.TD_epsilon = 0.0001
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        if agent_memory_based:
            self.state_size += agent_memory_size


        ## model of the env model predicts S(t+1), r(t+1) = model(s,a) 
        self.model_based = model_based
        if model_based:
            self.model_network_input_n = self.state_size + 1
            self.model_hidden_size = hidden_size
            self.model_output_size = self.state_size + 1 
            self.model_max_simulation_n = 5 
            self.model_learning_rate = learning_rate
            self.model_loss = 0 
            self.model_epsilon = 0.1 
            self.model_loss_bound = 0.01 
            self.model_experience_memory = deque(maxlen=memory_size)
            self.simulation_batch_size = 64 
            

        ## memory and batch setup for batch learning 
        self.batch_size = batch_size
        self.sample_var_n = 100 
        self.memory_size = memory_size
        self.gamma = gamma
        self.experience_memory = deque(maxlen=self.memory_size)
        self.epsilon = 1.0 
        self.epsilon_min = 0.01 
        self.epsilon_decay_rate = 0.995

        ## initializing policy network of choosing

        if policy_network == 'Q_network':
            print("policy network is currently q network")
            self.q_network = QNetwork(self.state_size, action_size, hidden_size).to(self.device)
        elif policy_network == 'LSTM_Q':
            print("policy_network is currently LSTM network")
            self.q_network = LSTM_Q(self.state_size, action_size, hidden_size).to(self.device)
        else:
            raise Exception('Error!!!! network not defined')
        

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        ## initialize model if model_based 
        if model_based:
            self.model_network = Model_Network_valila(self.model_network_input_n, self.model_output_size, self.model_hidden_size)
            self.model_network.to(self.device)
            self.model_optimizer = optim.Adam(self.model_network.parameters(), lr=self.model_learning_rate)
            self.model_criterion = nn.MSELoss()
            ## supervised learning for the model when model_train_n reaches simul_start_n model prediction starts 
            self.model_train_n = 0 
            self.model_simul_start_n = 5000
            self.model_solver = ModelSolver() 
        
        if agent_memory_based:
            self.agent_memory_based = agent_memory_based
            self.agent_memory_size = agent_memory_size 
            self.init_memory(agent_memory_size)

    def init_memory(self, agent_memory_size):
        self.agent_memory = np.zeros(agent_memory_size)
        self.small_reward_memory_reset() 
        self.init_goal_memory_dict() 

    def init_goal_memory_dict(self):
        self.goal_memory_dict = {} 
        max_length = 10 
        self.goal_memory_deque = deque([np.array([0,0])]*max_length, maxlen=max_length)

    def record_goal_memory(self, next_state, reward):
        if reward > 6: 
            self.goal_memory_deque.append(next_state)
        else:
            self.goal_memory_deque.append([0,0])



    def goal_memory2dict(self):
        self.goal_memory_dict = {} 
        for mem in self.goal_memory_deque:
            mem = tuple(mem)
            if mem in self.goal_memory_dict:
                self.goal_memory_dict[mem] += 1 
            else:
                self.goal_memory_dict[mem] = 1 

        self.goal_select_from_memory()


    def goal_select_from_memory(self):
        counter_values = np.array(list(self.goal_memory_dict.values())) 
        p_values = counter_values/sum(counter_values)
        random_goal_idx = int(np.random.choice(len(p_values), 1, p=p_values))
        guss_goal = list(self.goal_memory_dict.keys())[random_goal_idx]
        
        self.guess_goal = np.array(guss_goal)
        self.agent_memory[:2] = self.guess_goal
        print(self.guess_goal)

    def goal_check_if_none(self): 
        return np.equal(np.zeros(2), self.agent_memory[:2]).all()
        
        

    
    
    def small_reward_memory_reset(self):
        self.agent_small_reward_memory = 0 
        self.agent_memory[-1] = self.agent_small_reward_memory

    def record_small_reward_memory(self, small_reward):
        ## decay of small reward memory first 
        self.agent_small_reward_memory = 0.9*self.agent_small_reward_memory
        self.agent_small_reward_memory = round(self.agent_small_reward_memory, 2)

        if small_reward == 1: 
            self.agent_small_reward_memory += small_reward
            


        

    def state_memory_wapper(self, state):
        """For newly defined state with memory attachment  
        """
        if self.agent_memory_based:
            memory_state = np.append(state, self.agent_memory)
        else: 
            return state

        return memory_state
             

    def decay_memory(self):
        pass

    def act(self, state, max_q_action= True): 

        state = self.state_memory_wapper(state) 

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size) 
        else:
            state_tensor = torch.Tensor(state).to(self.device)

            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            if max_q_action:
                return np.argmax(q_values.cpu().numpy())
            else: 
                raise NotImplementedError("currently working on it!!")
            

    def remember(self, state, action, reward, next_state, done):
        if self.agent_memory_based:
            state = self.state_memory_wapper(state)
            self.record_small_reward_memory(reward)
            next_state = self.state_memory_wapper(next_state)
        
        self.experience_memory.append((state, action, reward, next_state, done))
        ## sample_var_n is used to calculate 95% percentile of variance 
        ## if batch size is 1000 and sample_var_n = 100, 100 resent sample will be selected with 95 % confidance 
    def model_simulation_remember(self, state, action, reward, next_state, done):
        self.model_experience_memory.append((state, action, reward, next_state, done))



    def make_gaussian_sample(self, batch_length, sample_var_n = 100):
        gaussian_p = stats.norm.pdf([x for x in range(batch_length)], batch_length, sample_var_n/1.96)
        gaussian_p = gaussian_p/sum(gaussian_p)
        return gaussian_p
    
    def make_TD_PER(self):
        """for implementation of prioritized experience replay
        calculate td error and PER wights of p_values as output"""
        TD_all = np.zeros(len(self.experience_sub_memory))

        states = torch.FloatTensor(np.array([t[0] for t in self.experience_sub_memory])).to(self.device)
        actions = [t[1] for t in self.experience_sub_memory]
        rewards = [t[2] for t in self.experience_sub_memory]
        next_states = torch.FloatTensor(np.array([t[3] for t in self.experience_sub_memory])).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(states)
            q_values_next = self.q_network(next_states)

        for i in range(len(self.experience_sub_memory)):
            current_Q = q_values[i][actions[i]]
            reward = rewards[i]
    
            max_Q_next = np.argmax((q_values_next[i]).cpu().numpy())
            max_Q_next = q_values_next[i][max_Q_next]
            TD_error = reward + self.gamma * max_Q_next - current_Q
            TD_error = abs(TD_error)
            priority_i = (TD_error + self.TD_epsilon)**self.alpha
            TD_all[i] = priority_i

        TD_all = TD_all/sum(TD_all)
            
        

    

        return TD_all 

    def replay(self, uniformed_sample = True, TD_sample = False): 
        if len(self.experience_memory) < self.batch_size:
            return 
        
        if uniformed_sample:
            minibatch = random.sample(self.experience_memory, self.batch_size)
        elif TD_sample:
            middle_batch_size = 4 * self.batch_size
            if len(self.experience_memory) >= middle_batch_size:
                self.experience_sub_memory = random.sample(self.experience_memory, middle_batch_size)
            else:
                self.experience_sub_memory = self.experience_memory
            TD_PER = self.make_TD_PER()
            minibatch_idx = np.random.choice(list(range(len(self.experience_sub_memory))), self.batch_size, p=TD_PER)
            minibatch = [self.experience_memory[i] for i in minibatch_idx]
        else:
            gaussain_p = self.make_gaussian_sample(len(self.experience_memory), sample_var_n=self.sample_var_n)
            minibatch_idx = np.random.choice(list(range(len(self.experience_memory))), self.batch_size, p=gaussain_p)
            minibatch = [self.experience_memory[i] for i in minibatch_idx]
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)
        
        ## greedly optimized with TD error 
        q_values = self.q_network(states)
        next_q_values = self.q_network(next_states)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]
        q_values = q_values.gather(1, actions.unsqueeze(1))
        loss = self.criterion(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.model_based:
            self.model_train(states, actions, rewards, next_states)

    def model_train(self, states, actions, rewards, next_states):
        self.model_train_n += 1 
        state_action_pair = torch.cat((states, actions.unsqueeze(1)), dim=1)
        reward_next_state_pair = torch.cat((next_states, rewards.unsqueeze(1)), dim=1)
        model_next_states, model_next_reward = self.model_network(state_action_pair)

        current_model_output = torch.cat((model_next_states, model_next_reward), dim=1)
        target_model_output = reward_next_state_pair
        loss = self.model_criterion(current_model_output, target_model_output)
        self.model_loss = loss.item()
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()


    def model_simulate(self, state, simulation_size):
        if self.goal_check_if_none():
            return 


        self.model_experience_memory = deque(maxlen=self.memory_size)
        

        if (self.model_loss > self.model_loss_bound) or (self.model_train_n < self.model_simul_start_n):
            return 
        
        start_state = self.state_memory_wapper(state)
        
        for simulation_num in range(simulation_size):

            state = start_state 

        
            done = False 
            simul_n = 0

            while not(done):
                simul_n +=1 
                if simul_n >= self.model_max_simulation_n or np.array_equal(state[:2], state[-2:]):
                    done = True 

                if np.random.rand() <= 0.1:
                    action = np.random.randint(self.action_size) 
                else: 
                    action = self.model_solver.act(state)

                action = np.array([action])
                
            
                
                state_action_pair = np.concatenate((state, action))
                state_action_tensor = torch.FloatTensor(state_action_pair).to(self.device)
                next_state, reward = self.model_network(state_action_tensor)

                next_state = next_state.cpu().detach().numpy()
                next_state = np.round(next_state)
                reward = float(reward.cpu().detach().numpy())
          

                if reward > 5:
                    done = True 
                    for rem in range(50):
                        self.model_simulation_remember(state, action, reward, next_state, done)
                self.model_simulation_remember(state, action, reward, next_state, done)
                state = next_state

        self.simulation_learn() 

    def simulation_learn(self):
        if len(self.model_experience_memory) < self.batch_size:
            return
        # middle_batch_size = 4 * self.batch_size
        # if len(self.model_experience_memory) >= middle_batch_size:
        #     model_sub_memory = random.sample(self.model_experience_memory, middle_batch_size)
        # else:
        #     model_sub_memory = self.model_experience_memory
        # TD_PER = self.make_TD_PER(model_sub_memory)
        # minibatch_idx = np.random.choice(list(range(len(model_sub_memory))), self.batch_size, p=TD_PER)
        # minibatch = [model_sub_memory[i] for i in minibatch_idx]
        minibatch = random.sample(self.model_experience_memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)
        
        ## greedly optimized with TD error 
        q_values = self.q_network(states)
        next_q_values = self.q_network(next_states)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]
        q_values = q_values.gather(1, actions)
        loss = self.criterion(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay_rate  * self.epsilon)

    def upcay_epsilon(self):
        self.epsilon = min(1.0, self.epsilon/self.epsilon_decay_rate)


       