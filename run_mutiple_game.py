from agent_train import agent_train

game_ver = 0
agent_train( TD_sample=False, game_version=game_ver, agent_memory_based=False)
agent_train( TD_sample=False, game_version=game_ver, agent_memory_based=True, pre_train=True)
agent_train( TD_sample=True, game_version=game_ver, agent_memory_based=False)
agent_train( TD_sample=True, game_version=game_ver, agent_memory_based=True, pre_train=True)




