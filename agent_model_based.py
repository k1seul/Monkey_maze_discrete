from network_module import * 
import torch
import torch.nn as nn 
import torch.optim as optim 
from collections import deque 
import random 
import numpy as np 
import scipy.stats as stats


class Agent():
    """
    batch learning dqn agent with selection of network structure
    """
    def __init__(self, state_size, action_size, hidden_size, learning_rate, 
                 memory_size, batch_size, gamma,
                 policy_network= 'Q_network', model_based= False):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("currently running the calculation on " + str(self.device))
        
        self.gpu_usage = True if torch.cuda.is_available() else False

        self.alpha = 0.9
        self.TD_epsilon = 0.0001
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.learning_rate = learning_rate


        ## model of the env model predicts S(t+1), r(t+1) = model(s,a) 
        self.model_based = model_based ##change later 
        if self.model_based:
            self.model_network_input_n = state_size + 1
            self.model_hidden_size = hidden_size
            self.model_output_size = state_size + 1 
            self.model_max_simulation_n = 100 
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
            self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        elif policy_network == 'LSTM_Q':
            print("policy_network is currently LSTM network")
            self.q_network = LSTM_Q(state_size, action_size, hidden_size).to(self.device)
        else:
            raise Exception('Error!!!! network not defined')
        

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        ## initialize model if model_based 
        if self.model_based:
            self.model_network = Model_Network_valila(self.model_network_input_n, self.model_output_size, self.model_hidden_size)
            self.model_network.to(self.device)
            self.model_optimizer = optim.Adam(self.model_network.parameters(), lr=self.model_learning_rate)
            self.model_criterion = nn.MSELoss()

            ## supervised learning for the model when model_train_n reaches simul_start_n model prediction starts 
            self.model_train_n = 0 
            self.model_simul_start_n = 10000

    def init_memory(self, memory_size):
        self.agent_memory = np.zeros(memory_size)

    def decay_memory(self):
        pass

    def act(self, state, max_q_action= True): 
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
        self.experience_memory.append((state, action, reward, next_state, done))
        ## sample_var_n is used to calculate 95% percentile of variance 
        ## if batch size is 1000 and sample_var_n = 100, 100 resent sample will be selected with 95 % confidance 

    def model_simulation_remember(self, state, action, reward, next_state, done):
        self.model_experience_memory.append((state, action, reward, next_state, done))

    def make_gaussian_sample(self, batch_length, sample_var_n = 100):
        gaussian_p = stats.norm.pdf([x for x in range(batch_length)], batch_length, sample_var_n/1.96)
        gaussian_p = gaussian_p/sum(gaussian_p)
        return gaussian_p
    
    def make_TD_PER(self , memory):
        """for implementation of prioritized experience replay
        calculate td error and PER wights of p_values as output"""
        TD_all = np.zeros(len(memory))

        states = torch.FloatTensor(np.array([t[0] for t in memory])).to(self.device)
        actions = [t[1] for t in memory]
        rewards = [t[2] for t in memory]
        next_states = torch.FloatTensor(np.array([t[3] for t in memory])).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(states)
            q_values_next = self.q_network(next_states)

        for i in range(len(memory)):
            current_Q = q_values[i][actions[i]]
            reward = float(rewards[i])
            
    
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
            TD_PER = self.make_TD_PER(self.experience_sub_memory)
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


        self.model_experience_memory = deque(maxlen=self.memory_size)
        

        if (self.model_loss > self.model_loss_bound) or (self.model_train_n < self.model_simul_start_n):
            return 
        
        for simulation_num in range(simulation_size):
            done = False 
            simul_n = 0

            while not(done):
                simul_n +=1 
                if simul_n >= self.model_max_simulation_n:
                    done = True 

                if np.random.rand() < self.model_epsilon:
                    action = np.random.randint(self.action_size)
                else:
                    action = self.act(state)

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
       
    

    


