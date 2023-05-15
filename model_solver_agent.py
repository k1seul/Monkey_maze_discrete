from network_module import * 
import torch

class ModelSolver():
    def __init__(self):
        self.state_n = 4
        self.action_n = 4 
        self.hidden_n = 1024 
        self.solver_network = QNetwork(self.state_n, self.action_n, self.hidden_n)
        solver_para_name = "model_solver_para.pth"
        self.solver_network.load_state_dict(torch.load(solver_para_name, map_location=torch.device('cpu')))
    def act(self, state): 
        state_tensor = torch.Tensor(state).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
       
        return np.argmax(q_values.cpu().numpy())
      


