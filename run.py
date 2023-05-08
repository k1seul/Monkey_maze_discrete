from agent_train import agent_train

for game_ver in range(4):
    agent_train(uniform_sample=True, TD_sample=False, game_version=game_ver)
    agent_train(uniform_sample=True, TD_sample=False, game_version=game_ver, TD_switch= True)
    agent_train(uniform_sample=True, TD_sample=False, game_version=game_ver, TD_switch= False, model_based= True)
    agent_train(uniform_sample=True, TD_sample=False, game_version=game_ver, TD_switch= True, model_based= True)
    