from monkeyqmazediscrete_v2 import MonkeyQmazeDiscreteV2
from monkeyqmazediscrete_v3 import MonkeyQmazeDiscreteV3
from monkeyqmazediscrete_v0 import MonkeyQmazeDiscreteV0
from monkeyqmazediscrete import MonkeyQmazeDiscrete
from monkeyqmazediscrete_v0_cheat import MonkeyQmazeDiscreteV0Cheat
import pygame 





terminated = False
truncated = False 
game_ver = int(input("type in the game version"))
if game_ver == 0 or game_ver == 9:
    a = [3,2,0,1,4]
elif game_ver == 1:
    a = [3,2,1,0,4]
elif game_ver == 2:
    a = [3,2,0,1,4]
else: 
    a = [3,2,0,1,4]

if game_ver == 0:
    env = MonkeyQmazeDiscreteV0(render_mode='human')
elif game_ver == 1:
    env = MonkeyQmazeDiscrete(render_mode='human')
elif game_ver == 2:
    env = MonkeyQmazeDiscreteV2(render_mode='human')
elif game_ver == 3: 
    env = MonkeyQmazeDiscreteV3(render_mode='human')
elif game_ver == 9:
    env = MonkeyQmazeDiscreteV0Cheat(render_mode='human')
else:
    raise ValueError("game_Version is not defined!!")


for trial_num in range(9):
    state, info = env.reset(reward_idx = trial_num)
    terminated = False
    truncated = False 

    pygame.event.clear()
    sum_reward = 0

    while not(terminated or truncated):
        

        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_UP:
                    action = a[0]
                elif ev.key == pygame.K_LEFT:
                    action = a[1]
                elif ev.key == pygame.K_RIGHT:
                    action = a[2]
                elif ev.key == pygame.K_DOWN:
                    action = a[3]

                elif ev.key == pygame.K_SPACE:
                    action = a[4] 


                obs, reward, done, truncated, info = env.step(action)
                print(obs)
                print(reward)
                sum_reward = sum_reward + reward

        
                if done or truncated:
                    print("Game over! Final score: {}".format(sum_reward))
                    terminated = done 

                    break 



