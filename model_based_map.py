import gymnasium as gym 
import pygame 
import numpy as np 
from gymnasium import spaces
import itertools



class ModelBasedMap(gym.Env):
    
    version_num = "1.3.0"
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode = None, memory_state_size = None):
        self.window_size = 800
        self.size = 11 # 11 x 11 grid world 

        # minimal_state_n is given as (x,y) coordinate, head_direction, front_sight
        self.state_n = 4
        # action is four arrows with tank control
        # 0:go_front 1: turn_left, 2: turn_right ;; 3: go_behind(can be turned off)(off by default)
        self.action_n = 4
        self.max_episode_step = 200 

        ## front_sight is defined as 0: no_wall, 1: wall, 2: small_reward, 3: jackpot_reward 

        self.observation_space = spaces.Dict({
            "agent_coordinate": spaces.Box(low=0, high=11, shape=([2]), dtype=np.int64),
            "head_direction": spaces.Discrete(4),
            "front_sight": spaces.Box(low= 0, high= 3, shape=[1], dtype=np.int64)
        })
        self.action_space = spaces.Discrete(self.action_n)
        self.init_map()



        self._head_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None 


    
    def init_map(self): 
        """
        initialize map of 11 by 11 grid structure"""

        x = np.arange(self.size) 
        y = np.arange(self.size) 

        xx, yy = np.meshgrid(x,y) 
        self.road_map = np.stack([xx,yy], axis=-1).reshape(self.size**2, 2)
        
        
        self.whole_map = np.zeros([self.size, self.size])

        for road_point in self.road_map:
            self.whole_map[road_point[0]][road_point[1]] = 1

        self.wall_location = [] 

        for i, j in itertools.product(range(11), range(11)):
            if not(self.whole_map[i][j] == 1):
                self.wall_location.append([i,j])

        self.wall_location = np.array(self.wall_location)

    def init_reward(self):
        """
        initialize goal location of current trial if reward, start idx aren't given
         it will be randomized """
        
        reward_idx = self.np_random.integers(0, self.size**2)
    
        start_idx = self.np_random.integers(0,self.size**2)

        while start_idx == reward_idx:
            start_idx = self.np_random.integers(0,self.size**2)
        

        self.trial_goal = self.road_map[reward_idx]
        self.trial_start = self.road_map[start_idx]

    def _get_obs(self):
        """
        observation without any memory inputed back to the state
        default observation will be agent coordinate, head_direction
        """
        observation = {
            "agent_coordinate": self._agent_location,
            "goal_location": self.trial_goal,
        }

        obs_array = np.concatenate([observation[key] for key in observation]) 

        return obs_array

    def _get_info(self):

        return None
    
    def reset(self, seed=None, reward_idx=None, start_idx=None, random_head_direction=True):
        
        super().reset(seed=seed)
        self.init_reward()

        ## initializing starting agent state 
        self._agent_location = self.trial_start
        if random_head_direction:
            self._agent_head_direction = self.np_random.integers(0, 4)
        else:
            self._agent_head_direction = 0


        observation = self._get_obs()
        info = self._get_info()
        self.step_count = 1 

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info 
    
    def step(self, action):

        self.step_count += 1 
        head_turned = False 
        truncated = False 
        # 0:go_front 1: turn_left, 2: turn_right ;; 3: go_behind(can be turned off)(off by default)
        if action == 0 or action == 1 or action == 2 or action == 3:
            
            new_location = self._agent_location + self._head_to_direction[action]
        else: 
            raise ValueError("action is not defined")
        

        
        if not(new_location.tolist() in self.road_map.tolist()):
            new_location = self._agent_location

        

        wall_hit = np.array_equal(self._agent_location, new_location) and not(head_turned)

        self._agent_location = new_location

        terminated = np.array_equal(self._agent_location, self.trial_goal)
        if self.step_count >= self.max_episode_step:
            truncated = True  

        observation = self._get_obs() 
        info = self._get_info() 

        if terminated:
            reward = 8
        elif wall_hit:
            reward = -0.1
        else:
            reward = 0

        if self.render_mode == 'human':
            self._render_frame() 

        return observation, reward, terminated, truncated, info 
    



    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()
            




    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels







        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.trial_goal,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
    

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



