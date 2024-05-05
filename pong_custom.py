import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def rgb_to_grayscale(rgb_tensor):
    # Convert RGB tensor to grayscale using weighted average
    grayscale_tensor = np.dot(rgb_tensor[..., :3], [0.2989, 0.5870, 0.1140])

    return torch.FloatTensor(grayscale_tensor)


class PolicyNetwork(nn.Module):

  """
    Implement the Policy Network.

    Your task is to complete the initialization of the policy network that maps states to action probabilities.
    This network should consist of several fully connected layers with ReLU activation, followed by a final layer
    that outputs logits for each action. The forward pass should return a Categorical distribution over actions.

    Instructions:
    1. Initialize the fully connected layers in the __init__ method.
    2. Implement the forward pass to return a Categorical distribution given state inputs.

    Hint: The constructor takes 'state_dim' and 'action_dim' as arguments, representing the dimensions
    of the state space and action space, respectively.
  """
  def __init__(self, state_dim, action_dim):
      super(PolicyNetwork, self).__init__()
      ##### Code implementation here #####
      hidden=64
      self.l1 = nn.Linear(state_dim, hidden)
      self.l2 = nn.Linear(hidden, hidden)
      self.l3 = nn.Linear(hidden, hidden)
      self.output = nn.Linear(hidden, action_dim)
      ############################

  def forward(self, x):
      ##### Code implementation here #####
      layer1 = F.relu(self.l1(x))
      layer2 = F.relu(self.l2(layer1))
      layer3 = F.relu(self.l3(layer2))
      output_logits = self.output(layer3)
      dist = Categorical(logits=output_logits)
      return dist


class CustomPongEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomPongEnv, self).__init__()

        # Court measurements
        self.scale = 10
        self.front_wall_length = 21 * self.scale
        self.side_wall_length = 32 * self.scale
        self.front_wall_to_t = int(17.85 * self.scale)
        self.box_side_length = int(5.25 * self.scale)
        self.t_to_back_wall = self.side_wall_length - self.front_wall_to_t

        # Rendering measurements, colors
        self.edge_width = int(0.5*self.scale)
        
        self.action_space = spaces.Discrete(3)  # 0: stay, 1: up, 2: down
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.side_wall_length, self.front_wall_length, 3), dtype=np.uint8)

        # Game settings
        self.ball_position = [int(self.side_wall_length/2), int(self.front_wall_length/2)]
        self.ball_velocity = [int(2*self.scale / 10), int(np.random.choice([-1, 1])*self.scale / 10)]
        self.ball_size = self.scale
        self.paddle_position = int(self.front_wall_length / 2)
        self.paddle_width = 80
        self.paddle_height = 10
        self.score = 0

        self.done = False

    def step(self, action):
        self._take_action(action)
        self._update_ball()

        reward = self._get_reward()
        done = self._is_done()
        info = {'score': self.score}

        return self._get_obs(), reward, done, info

    def reset(self):
        self.ball_position = [int(self.side_wall_length/2), int(self.front_wall_length/2)]
        self.ball_velocity = [int(2*self.scale / 10), int(np.random.choice([-1, 1])*self.scale / 10)]
        self.ball_size = 10
        self.paddle_position = int(self.front_wall_length / 2)
        self.paddle_width = 80
        self.paddle_height = 10
        # self.score = 0

        self.done = False
        
        # self.ball_position = [int(self.side_wall_length/2), int(self.front_wall_length/2)]
        # self.ball_velocity = [4, np.random.choice([-1, 1])*2]
        # self.paddle_position = int(self.front_wall_length / 2)

        # self.score = 0
        return self._get_obs(), []

    def render(self, mode='human', close=False):
        
        # Screen dimensions
        screen = np.zeros((self.side_wall_length + 2*self.edge_width, self.front_wall_length + 2*self.edge_width, 3), dtype=np.uint8)

        # Right Wall
        screen[self.side_wall_length + self.edge_width:self.side_wall_length + 2*self.edge_width, :] = [255, 255, 255]
        # Left Wall
        screen[:, 0:self.edge_width] = [255, 255, 255]
        # Front Wall
        screen[0:self.edge_width, :]
        # Back Wall
        screen[:, self.front_wall_length + self.edge_width:self.front_wall_length + 2*self.edge_width] = [255, 255, 255]

        # T
        screen[self.front_wall_to_t: self.front_wall_to_t + self.edge_width, 0: self.front_wall_length + 2*self.edge_width] = [255,255,255]
        screen[self.front_wall_to_t:self.side_wall_length + 2*self.edge_width , int((self.front_wall_length + 2*self.edge_width - self.edge_width)/2):int((self.front_wall_length+ 2*self.edge_width + self.edge_width)/2)] = [255,255,255]

        # Boxes
        screen[self.front_wall_to_t + self.edge_width + self.box_side_length: self.front_wall_to_t + 2*self.edge_width + self.box_side_length, self.edge_width: self.edge_width + self.box_side_length] = [255,255,255]
        screen[self.front_wall_to_t + self.edge_width + self.box_side_length: self.front_wall_to_t + 2*self.edge_width + self.box_side_length, self.front_wall_length + self.edge_width - self.box_side_length: self.front_wall_length + self.edge_width] = [255,255,255]

        screen[self.front_wall_to_t + self.edge_width: self.front_wall_to_t + 2*self.edge_width + self.box_side_length, self.box_side_length + self.edge_width:2*self.edge_width + self.box_side_length] = [255,255,255]
        screen[self.front_wall_to_t + self.edge_width: self.front_wall_to_t + 2*self.edge_width + self.box_side_length, self.front_wall_length - self.box_side_length: self.front_wall_length + self.edge_width - self.box_side_length] = [255,255,255]



        # Draw the ball
        screen[self.ball_position[0]-self.ball_size:self.ball_position[0]+self.ball_size, self.ball_position[1]-self.ball_size:self.ball_position[1]+self.ball_size] = [255, 0, 0]
        # Draw the paddle
        screen[self.side_wall_length:self.side_wall_length+self.paddle_height, self.paddle_position: self.paddle_position + self.paddle_width] = [0, 0, 255]

        if mode == 'human':
            import cv2
            cv2.imshow("Squash", screen)
            cv2.waitKey(1)

    def _take_action(self, action):
        if action == 1 and self.paddle_position > 0:
            self.paddle_position -= 2  # Move paddle left
        elif action == 2 and self.paddle_position < 160:
            self.paddle_position += 2  # Move paddle right

    def _update_ball(self):
        self.ball_position[0] += self.ball_velocity[0]
        self.ball_position[1] += self.ball_velocity[1]

        # Check for collision with front wall
        if self.ball_position[0] <= self.edge_width:
            self.ball_velocity[0] = -self.ball_velocity[0]

        # Check for collision with the side walls
        if self.ball_position[1] <= self.edge_width or self.ball_position[1] >= self.front_wall_length + self.edge_width:
            self.ball_velocity[1] = -self.ball_velocity[1]

        # Check for collision with the paddle
        if self.ball_position[0] >= self.edge_width + self.side_wall_length - self.paddle_height and self.ball_position[1] >= self.paddle_position and self.ball_position[1] <= self.paddle_position + self.paddle_width:
            self.ball_velocity[0] = -self.ball_velocity[0]

        # Check for scoring
        if self.ball_position[0] >= self.side_wall_length + 2*self.edge_width:
            self.score += 1
            self.ball_position = [105, 80]
            self.ball_velocity = [int(2*self.scale / 10), int(np.random.choice([-1, 1])*self.scale / 10)]

            # self.ball_velocity = [4, -2 if self.ball_velocity[1] > 0 else 2] 
            self.done = True

    def _get_reward(self):
        if self.ball_position[0] > self.side_wall_length + self.edge_width:
            return -10
        return 0

    def _is_done(self):
        # return False
        return self.done


    # possible there is some issue with ball_position in rendering (w borders and stuff) vs ball_position that agent gets 
    def _get_obs(self):
        obs = np.zeros((self.side_wall_length, self.front_wall_length, 3), dtype=np.uint8)
        obs[self.ball_position[0]-self.ball_size:self.ball_position[0]+self.ball_size, self.ball_position[1]-self.ball_size:self.ball_position[1]+self.ball_size] = 255
        obs[self.edge_width + self.side_wall_length - self.paddle_height - 1:self.edge_width + self.side_wall_length - self.paddle_height + 1, self.paddle_position:self.paddle_position+self.paddle_width] = 255
        return obs

def test_pong_environment(episodes=10):
    
    # Create the environment
    env = CustomPongEnv() 
    
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim)
    checkpoint = torch.load('ppo_step131', map_location=torch.device('cpu'))
    # policy_net.load_state_dict(checkpoint['policy network'])
    # print(policy_net)
    score = 0
    
    for episode in range(episodes):
        # breakpoint()
        done = False
        
        obs, _ = env.reset()
        while not done:
            # Random action
            # action = env.action_space.sample()

            obs = torch.FloatTensor(obs)
            obs = rgb_to_grayscale(obs).flatten()   
            
                 
            # breakpoint()
            # Action according to trained policy
            action = policy_net(obs).sample().numpy()
            # breakpoint()
            # action = action[0]


            obs, reward, done, info = env.step(action)


            # obs = rgb_to_grayscale(obs).flatten()
            # obs = torch.FloatTensor(obs)


            score += reward
            # Render the game
            env.render()

            # You can print observations, rewards, and info if you want to see details
            # print(f"Observation: {obs}")
            # print(f"Reward: {reward}, Info: {info}")

        print(f"Episode {episode + 1}: Score = {info['score']}")

    env.close()

if __name__ == "__main__":
    test_pong_environment()