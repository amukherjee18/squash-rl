import gym
from gym import spaces
import numpy as np
import random

class CustomPongEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomPongEnv, self).__init__()

        # Court measurements
        self.scale = 20
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
        self.ball_velocity = [4, 2]
        self.ball_size = 10
        self.paddle_position = int(1*self.front_wall_length / 4)
        self.ai_paddle_position = int(1*self.front_wall_length / 2)

        self.paddle_width = 80
        self.paddle_height = 10
        self.score = 0

        self.done = False

        self.turn = False # False means ai paddle turn

    def step(self, action):
        self._take_action(action)
        self._update_ball()

        reward = self._get_reward()
        done = self._is_done()
        info = {'score': self.score}

        return self._get_obs(), reward, done, info

    def reset(self):
        self.ball_position = [int(self.side_wall_length/2), int(self.front_wall_length/2)]
        self.ball_velocity = [4, 2]
        self.paddle_position = int(1*self.front_wall_length / 4)
        self.ai_paddle_position = int(3*self.front_wall_length / 4)

        self.turn = False # False means ai paddle turn


        # self.score = 0
        return self._get_obs()

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
        # Draw ai paddle
        screen[self.side_wall_length:self.side_wall_length+self.paddle_height, self.ai_paddle_position: self.ai_paddle_position + self.paddle_width] = [0, 255, 0]

        if mode == 'human':
            import cv2
            cv2.imshow("Squash", screen)
            cv2.waitKey(1)

    def _take_action(self, action):
        if action == 1 and self.paddle_position > 0:
            self.paddle_position -= 2  # Move paddle left
        elif action == 2 and self.paddle_position < self.front_wall_length + self.edge_width - self.paddle_width:
            self.paddle_position += 2  # Move paddle right

        # Determine action for ai paddle
        if self.ball_position[1] < self.ai_paddle_position and self.ai_paddle_position > 0:
            self.ai_paddle_position -= 3
        elif self.ball_position[1] > self.ai_paddle_position + self.paddle_width and self.ai_paddle_position < self.front_wall_length + self.edge_width - self.paddle_width:
            self.ai_paddle_position += 3
        # else:
            # self.ai_paddle_position = 0  # Stay


    def _update_ball(self):
        self.ball_position[0] += self.ball_velocity[0]
        self.ball_position[1] += self.ball_velocity[1]

        # Check for collision with front wall
        if self.ball_position[0] <= self.edge_width:
            self.ball_velocity[0] = -self.ball_velocity[0]

        # Check for collision with the side walls
        if self.ball_position[1] <= self.edge_width or self.ball_position[1] >= self.front_wall_length + self.edge_width:
            self.ball_velocity[1] = -self.ball_velocity[1]

        if self.turn:
            # Check for collision with player paddle
            if self.ball_position[0] >= self.edge_width + self.side_wall_length - self.paddle_height and self.ball_position[1] >= self.paddle_position and self.ball_position[1] <= self.paddle_position + self.paddle_width:
                self.ball_velocity[0] = -self.ball_velocity[0]
                self.turn = False
        else:
            # Check for collision with ai paddle
            if self.ball_position[0] >= self.edge_width + self.side_wall_length - self.paddle_height and self.ball_position[1] >= self.ai_paddle_position and self.ball_position[1] <= self.ai_paddle_position + self.paddle_width:
                self.ball_velocity[0] = -self.ball_velocity[0]
                self.turn = True
                
        # Check for scoring
        if self.ball_position[0] >= self.side_wall_length + 2*self.edge_width:
            self.score += 1
            self.ball_position = [105, 80]
            self.ball_velocity = [4, -2 if self.ball_velocity[1] > 0 else 2] 
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
    for episode in range(episodes):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            # Random action
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
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