from agent_train import agent_train

game_ver = 3 
agent_train(uniform_sample=True, TD_sample=False, game_version=game_ver, agent_memory_based=True)
agent_train(uniform_sample=True, TD_sample=False, game_version=game_ver, agent_memory_based=False)

