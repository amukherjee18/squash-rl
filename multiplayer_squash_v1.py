import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ray.rllib.env import EnvContext
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN

from gymnasium.wrappers import EnvCompatibility
import numpy as np
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
import gymnasium as gym
from gymnasium import spaces
import torch
from tqdm import tqdm
import ray


class CustomPongEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomPongEnv, self).__init__()

        # Measurements
        self.scale = 10

        # Court
        self.front_wall_length = 21 * self.scale
        self.side_wall_length = 32 * self.scale
        self.front_wall_to_t = int(17.85 * self.scale)
        self.box_side_length = int(5.25 * self.scale)
        self.t_to_back_wall = self.side_wall_length - self.front_wall_to_t
        self.line_width = int(0.2*self.scale)

        # Ball
        self.ball_position = [int(self.side_wall_length/2), int(self.front_wall_length/2)]
        self.ball_velocity = [int(2*self.scale / 10), int(np.random.choice([-1, 1])*self.scale / 10)]
        self.ball_radius = int(self.scale / 2)
        self.ball_size = int(self.scale / 2) # For state sent to policy net
        self.dot_radius = int(self.scale / 10)

        # Paddle
        self.paddle_center = int(self.front_wall_length / 2)
        self.ai_paddle_center = int(1*self.front_wall_length / 2)
        self.paddle_halfwidth = int(self.front_wall_length / 6)
        self.paddle_height = int(self.scale/2)
        self.paddle_velocity = 2
        self.paddle_border = 2
        self.ai_paddle_border = 1

        # State, Action Spaces
        self.action_space = spaces.Discrete(3)  # 0: stay, 1: up, 2: down
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.side_wall_length, self.front_wall_length, 3), dtype=np.uint8)

        self.done = False
        self.turn = False # False means ai paddle turn

    def step(self, action):
        self._take_action(action)
        self._update_ball(action)

        reward = self._get_reward()
        done = self._is_done()

        return self._get_obs(), reward, done, {}

    def reset(self):

        self.ball_position = [int(self.side_wall_length/2), int(self.front_wall_length/2)]
        self.ball_velocity = [int(2*self.scale / 10), int(np.random.choice([-1, 1])*self.scale / 10)]
        self.paddle_center = int(self.front_wall_length / 2)
        self.ai_paddle_center = int(1*self.front_wall_length / 2)

        self.turn = False # False means ai paddle turn
        self.done = False

        return self._get_obs()

    def render(self, mode='human', close=False):
        
        # Screen dimensions
        screen = np.zeros((self.side_wall_length, self.front_wall_length, 3), dtype=np.uint8)
        screen[:,:] = [149, 203, 231]

        # T
        screen[self.front_wall_to_t: self.front_wall_to_t + self.line_width , 0: self.front_wall_length] = [0,0,255]
        screen[self.front_wall_to_t:self.side_wall_length + 2*self.line_width , int((self.front_wall_length - self.line_width)/2):int((self.front_wall_length + self.line_width)/2)] = [0,0,255]

        # Boxes
        # Horizontal Lines
        screen[self.front_wall_to_t + self.line_width + self.box_side_length: self.front_wall_to_t + 2*self.line_width + self.box_side_length, 0:self.box_side_length] = [0,0,255]
        screen[self.front_wall_to_t + self.line_width + self.box_side_length: self.front_wall_to_t + 2*self.line_width + self.box_side_length, self.front_wall_length - self.box_side_length: self.front_wall_length] = [0,0,255]
        # Vertical Lines
        screen[self.front_wall_to_t + self.line_width: self.front_wall_to_t + 2*self.line_width + self.box_side_length, self.box_side_length:self.line_width + self.box_side_length] = [0,0,255]
        screen[self.front_wall_to_t + self.line_width: self.front_wall_to_t + 2*self.line_width + self.box_side_length, self.front_wall_length - self.box_side_length: self.front_wall_length + self.line_width - self.box_side_length] = [0,0,255]

        # Function to draw a filled circle in the numpy array
        def draw_circle(image, center, radius, color):
            # Generate array of coordinates for the circle
            rows, cols = np.ogrid[:image.shape[0], :image.shape[1]]
            circle_mask = (rows - center[0])**2 + (cols - center[1])**2 <= radius**2
            # Draw the circle on the image
            image[circle_mask] = color

        # Draw the ball
        ball_center = (self.ball_position[0], self.ball_position[1])
        ball_left_dot_center = (self.ball_position[0] - int(self.ball_radius/4), self.ball_position[1] - int(self.ball_radius/4))
        ball_right_dot_center = (self.ball_position[0] - int(self.ball_radius/4), self.ball_position[1] + int(self.ball_radius/4))
        draw_circle(screen, ball_center, self.ball_radius, [0, 0, 0])
        draw_circle(screen, ball_left_dot_center, self.dot_radius, [0, 255, 255])
        draw_circle(screen, ball_right_dot_center, self.dot_radius, [0, 255, 255])

        # Draw the paddle
        screen[self.side_wall_length - self.paddle_height:self.side_wall_length, self.paddle_center - self.paddle_halfwidth: self.paddle_center + self.paddle_halfwidth] = [0, 0, 0]
        screen[self.side_wall_length - self.paddle_height + self.paddle_border:self.side_wall_length - self.paddle_border, self.paddle_center - self.paddle_halfwidth + self.paddle_border: self.paddle_center + self.paddle_halfwidth - self.paddle_border] = [0, 117, 231]

        screen[self.side_wall_length - self.paddle_height:self.side_wall_length, self.ai_paddle_center - self.paddle_halfwidth: self.ai_paddle_center + self.paddle_halfwidth] = [0, 0, 0]
        screen[self.side_wall_length - self.paddle_height + self.ai_paddle_border:self.side_wall_length - self.ai_paddle_border, self.ai_paddle_center - self.paddle_halfwidth + self.ai_paddle_border: self.ai_paddle_center + self.paddle_halfwidth - self.ai_paddle_border] = [2, 209, 254] # [48, 28, 165]
        screen[self.side_wall_length - self.paddle_height + 2*self.ai_paddle_border:self.side_wall_length - 2*self.ai_paddle_border, self.ai_paddle_center - self.paddle_halfwidth + 2*self.ai_paddle_border: self.ai_paddle_center + self.paddle_halfwidth - 2*self.ai_paddle_border] = [103, 54, 1] # [48, 28, 165]


        if mode == 'human':
            import cv2
            cv2.imshow("Squash", screen)
            cv2.waitKey(1)

    def _take_action(self, action):
        if action == 1 and self.paddle_center - self.paddle_halfwidth > 0:
            self.paddle_center -= self.paddle_velocity  # Move paddle left
        elif action == 2 and self.paddle_center + self.paddle_halfwidth < self.front_wall_length:
            self.paddle_center += self.paddle_velocity # Move paddle right

        if self.ball_position[1] < self.ai_paddle_center and self.ai_paddle_center - self.paddle_halfwidth > 1: #1 fixes some rendering glitch
            self.ai_paddle_center -= self.paddle_velocity
        elif self.ball_position[1] > self.ai_paddle_center and self.ai_paddle_center + self.paddle_halfwidth < self.front_wall_length:
            self.ai_paddle_center += self.paddle_velocity


    def _update_ball(self, action):
        self.ball_position[0] += self.ball_velocity[0]
        self.ball_position[1] += self.ball_velocity[1]

        # Check for collision with front wall
        if self.ball_position[0] - self.ball_radius <= 0:
            self.ball_velocity[0] = -self.ball_velocity[0]

        # Check for collision with the side walls
        if self.ball_position[1] - self.ball_radius <= 0 or self.ball_position[1] + self.ball_radius >= self.front_wall_length:
            self.ball_velocity[1] = -self.ball_velocity[1]

        if self.turn:
            # Check for collision with the paddle
            if self.ball_position[0] + self.ball_radius >= self.side_wall_length - self.paddle_height and self.ball_position[1] >= self.paddle_center - self.paddle_halfwidth and self.ball_position[1] <= self.paddle_center + self.paddle_halfwidth:
                self.ball_velocity[0] = -self.ball_velocity[0]
                # if action == 1: # (paddle moving left)
                #     self.ball_velocity[1] -= 2*int(self.scale / 10)
                # elif action == 2: # (paddle moving right)
                #     self.ball_velocity[1] += 2*int(self.scale / 10)
                self.turn = False
        else:
            if self.ball_position[0] + self.ball_radius >= self.side_wall_length - self.paddle_height and self.ball_position[1] >= self.ai_paddle_center - self.paddle_halfwidth and self.ball_position[1] <= self.ai_paddle_center + self.paddle_halfwidth:
                self.ball_velocity[0] = -self.ball_velocity[0]
                self.turn = True

        # Check for scoring
        if self.ball_position[0] + self.ball_radius >= self.side_wall_length:
            self.done = True


    def _get_reward(self):
        
        # If we win
        if not self.turn and self.ball_position[0] + self.ball_radius >= self.side_wall_length:
            self.done = True
            return 1

        # If we lose
        elif self.turn and self.ball_position[0] + self.ball_radius >= self.side_wall_length:
            self.done = True
            return -1

        # Collision with our paddle
        if self.turn and self.ball_position[0] + self.ball_radius >= self.side_wall_length - self.paddle_height and self.ball_position[1] >= self.paddle_center - self.paddle_halfwidth and self.ball_position[1] <= self.paddle_center + self.paddle_halfwidth:
            return 0.5
        
        return 0
        # # Penalty for distance between our paddel and the ball
        # val = -50*np.abs(self.ball_position[1] - (self.paddle_center + self.paddle_halfwidth)) / self.front_wall_length
        # return val

    def _is_done(self):
        return self.done

    def _get_obs(self):
        obs = np.zeros((self.side_wall_length, self.front_wall_length, 3), dtype=np.uint8)
        obs[self.ball_position[0]-self.ball_size:self.ball_position[0]+self.ball_size, self.ball_position[1]-self.ball_size:self.ball_position[1]+self.ball_size] = 255
        obs[self.side_wall_length - self.paddle_height - 1:self.side_wall_length - self.paddle_height + 1, self.paddle_center - self.paddle_halfwidth:self.paddle_center+self.paddle_halfwidth] = 255
        obs[self.side_wall_length - self.paddle_height - 1:self.side_wall_length - self.paddle_height + 1, self.paddle_center - self.paddle_halfwidth:self.paddle_center+self.paddle_halfwidth] = 127
        return obs

def test_pong_environment(episodes=10):
    print(ray.__version__)

    def env_creator(env_config):
        return EnvCompatibility(CustomPongEnv())

    # Register the custom environment
    register_env("Squash", env_creator)

    config = {
        "env": "Squash",
        "num_workers": 1,
        "framework": "torch",
        "model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [512, 256, 256, 3],
            "conv_filters": [
        [32, [5, 5], 2], 
        [64, [5, 5], 2], 
        [128, [5, 5], 2], 
        [256, [4, 4], 24],
        [256, [2, 2], 2],
    ],
        },
    }

    trainer = PPO(config=config)
    # trainer = DQN(config=config)

    # Train for 100 iterations
    policy_net = trainer.get_policy().model
    policy_net.load_state_dict(torch.load("ppo_multi_step_21.pth", map_location=torch.device('cpu')))

    # Create the environment
    env = CustomPongEnv() 

    reward_sum = 0
    
    for episode in range(episodes):
        # breakpoint()
        done = False
        
        obs = env.reset()
        while not done:
            # Random action
            # action = env.action_space.sample()

            # Action according to trained policy
            obs = torch.FloatTensor(obs).unsqueeze(0)
            output, _ = policy_net({"obs": obs}, [], None)
            dist = Categorical(logits=output)
            action = dist.sample().numpy()

            obs, reward, done, info = env.step(action)

            reward_sum += reward
            # Render the game
            env.render()

        print(f"Episode {episode + 1}: Score = {reward_sum}")

    env.close()

if __name__ == "__main__":
    test_pong_environment()