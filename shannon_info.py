import numpy as np 
import math 



def shannon_info(state_sequence, action_sequece, action_size):
    shannon_all = {} 
    for i, state in enumerate(state_sequence):
        if not(tuple(state) in shannon_all):
            shannon_all[tuple(state)] = np.zeros(action_size)
        
        action = int(action_sequece[i])
        shannon_all[tuple(state)][action] += 1

    shannon_sum = 0 
    

    for key in shannon_all.keys():
        actions_array = shannon_all[key]
        actions_array = actions_array/(sum(actions_array))

        for p_value in actions_array:
            if p_value != 0:
                shannon_sum += -(p_value)*math.log(p_value, action_size)
    return shannon_sum/len(shannon_all)





