import torch
import pickle
import numpy as np 


def Save_data(model, trial_num, save_rate=1000, name='model', memory_variable = np.array([None, None])):
    if trial_num % save_rate == 0: 
        name_pth = name + '_' + str(trial_num) +'.pth'
        torch.save(model.state_dict(), name_pth)

        if not(memory_variable.any() == None):
                with open(name + "memory_var_" +str(trial_num) + '.pickle', 'wb') as lf: 
                        data = list(memory_variable)
                        data = [str(dat) for dat in data]
                        string = ','.join(data)
                        pickle.dump(string, lf)
        
