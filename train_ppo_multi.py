import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from multiplayer_squash_v1 import CustomPongEnv
from tqdm import tqdm
import torch.nn.functional as F


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def compute_advantages(next_value, rewards, masks, values, gamma=0.99):
    values = values + [next_value]
    advantages = 0
    returns = []

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1].cpu() * masks[step] - values[step].cpu()
        advantages = delta + gamma * masks[step] * advantages
        returns.insert(0, advantages + values[step].cpu())
    return returns

def evaluate_policy(policy, env_name, seed=42):
    env_test = CustomPongEnv()
    # env_test.seed(seed)
    state, done, total_reward = env_test.reset()[0], False, 0
    state = torch.FloatTensor(state)
    state = rgb_to_grayscale(state).flatten()

    # while not done:
    for i in range(1000):        
        # print('state shape pre:', state.shape)
        # state = rgb_to_grayscale(state).flatten()
        state = state.to("cuda")
        dist = policy(state)
        next_state, reward, done, _ = env_test.step(dist.sample().unsqueeze(0).cpu().numpy()[0])
        state = rgb_to_grayscale(next_state).flatten()
        #state=next_state
        state = torch.FloatTensor(state)
        total_reward += reward
    return total_reward


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

# TODO: Implement the ValueNetwork class
class ValueNetwork(nn.Module):
  """
    Implement the Value Network.

    Your task is to complete the initialization of the value network that maps states to value estimates.
    Similar to the policy network, this network should consist of several fully connected layers with ReLU activation
    followed by a final layer that outputs a single value estimate for the input state.

    Instructions:
    1. Initialize the fully connected layers in the __init__ method.
    2. Implement the forward pass to return the value estimate given state inputs.

    Hint: The constructor takes 'state_dim' as an argument, representing the dimension of the state space.
    """
  def __init__(self, state_dim):
      super(ValueNetwork, self).__init__()
      ##### Code implementation here #####
      hidden = 64
      self.l1 = nn.Linear(state_dim, hidden)
      self.l2 = nn.Linear(hidden, hidden)
      self.l3 = nn.Linear(hidden, hidden)
      self.output = nn.Linear(hidden, 1)

  def forward(self, x):
      ##### Code implementation here #####
      layer1 = F.relu(self.l1(x))
      layer2 = F.relu(self.l2(layer1))
      layer3 = F.relu(self.l3(layer2))
      return self.output(layer3)

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    mini_batches = []

    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        mini_batch = states[rand_ids, :], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]
        mini_batches.append(mini_batch)

    return mini_batches

def ppo_update(policy_net, value_net, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    """
    Implement the PPO update algorithm.

    This function should perform the optimization of the policy and value networks using the Proximal Policy Optimization (PPO) algorithm.
    You'll need to compute the ratio of new and old policy probabilities, apply the clipping technique, and calculate the losses for both the actor (policy network) and critic (value network).

    Instructions:
    1. Iterate over the number of PPO epochs, which is the number of optimizer.step() with the current collected data.
    2. In each epoch, iterate over the mini-batches of experiences.
    3. Calculate the new log probabilities of the actions taken, using the policy network.
    4. Compute the ratio of new to old probabilities.
    5. Apply the PPO clipping technique to the computed ratios.
    6. Calculate the actor (policy) and critic (value) losses.
    7. Combine the losses and perform a backpropagation step.

    Hints:
    - Use `policy_net(state)` to get the distribution over actions for the given states.
    - The `dist.log_prob(action)` method calculates the log probabilities of the taken actions according to the current policy.
    - The ratio is computed as the exponential of the difference between new and old log probabilities (`(new_log_probs - old_log_probs).exp()`).
    - Use `torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)` to clip the ratio between `[1-clip_param, 1+clip_param]`.
    - The actor loss is the negative minimum of the clipped and unclipped objective, averaged over all experiences in the mini-batch.
    - The critic loss is the mean squared error between the returns and the value estimates from the value network.
    - Remember to zero the gradients of the optimizer before the backpropagation step with `optimizer.zero_grad()`.
    - After computing the loss and performing backpropagation with `loss.backward()`, take an optimization step with `optimizer.step()`.
    """
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist = policy_net(state)
            new_log_probs = dist.log_prob(action)

            ##### Code implementation here #####
            ratio = (new_log_probs-old_log_probs).exp()
            clipped = torch.clamp(ratio, 1.0-clip_param, 1.0+clip_param)
            actor_loss = -(torch.min(advantage*clipped,advantage*ratio)).mean()
            critic_loss = F.mse_loss(return_.unsqueeze(1), value_net(state))
            ##### Code implementation End #####

            loss = 0.5 * critic_loss + actor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def rgb_to_grayscale(rgb_tensor):
    # Convert RGB tensor to grayscale using weighted average
    grayscale_tensor = np.dot(rgb_tensor[..., :3], [0.2989, 0.5870, 0.1140])

    return torch.FloatTensor(grayscale_tensor)

def train(env_name='CartPole-v1', num_steps=2000, mini_batch_size=8, ppo_epochs=4, threshold=400):
    env = CustomPongEnv()
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    print('Observation space: ', env.observation_space.shape)
    print('State dim: ', state_dim)
    print('Action space: ', env.action_space)
    action_dim = env.action_space.n
    device = "cuda"
    policy_net = PolicyNetwork(state_dim, action_dim)
    policy_net.to(device)
    value_net = ValueNetwork(state_dim)
    value_net.to(device)
    optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=3e-3)

    state, _ = env.reset()
    early_stop = False
    reward_list = []

    for step in tqdm(range(num_steps), total=num_steps):
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        entropy = 0

        # Collect samples under the current policy
        for _ in tqdm(range(4096), total=4096):
            rgb_state = torch.FloatTensor(state)
            rgb_state = rgb_state.unsqueeze(0)
            # print('RGB State shape: ', rgb_state.shape)
            state_unflattened = rgb_to_grayscale(rgb_state)
            # print('Grayscale state shape: ', state_unflattened.shape)
            state = state_unflattened.flatten()
            # print('Flattened state shape: ', state.shape)
            state = state.to(device)
            dist, value = policy_net(state), value_net(state)

            action = dist.sample().unsqueeze(0)
            # print('Sampled Action: ', action)
            # print('Action to numpy: ', action.numpy())
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            masks.append(torch.tensor([1-done], dtype=torch.float32))
            states.append(state)
            actions.append(action)

            state = next_state
            if done:
                state, _ = env.reset()


        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        next_state = rgb_to_grayscale(next_state).flatten()
        next_state = next_state.to(device)
        next_value = value_net(next_state)
        returns = compute_advantages(next_value, rewards, masks, values)

        returns = torch.cat(returns).detach()
        returns = returns.to(device)
        # print('Log Probs: ', log_probs)
        log_probs = torch.cat(log_probs).detach()
        log_probs = log_probs.to(device)
        # print('Log Probs cat: ', log_probs)
        values = torch.cat(values).detach()
        states = torch.stack(states)
        # print('States cat: ', states.shape)
        # print('Actions: ', actions)
        actions = torch.cat(actions)
        actions = actions.to(device)
        # print('Actions, cat: ', actions)
        advantage = returns - values

        # run PPO update for policy and value networks
        ppo_update(policy_net, value_net, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
        
        if step % 50 == 1:
            checkpoint = {
                "step": step,
                "policy network": policy_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "value network": value_net.state_dict(),
            }
            torch.save(checkpoint, f'checkpoints_multi/ppo_multi_step{step}')
            test_reward = np.mean([evaluate_policy(policy_net, env_name) for _ in tqdm(range(10), total=10)])
            print(f'Step: {step}\tReward: {test_reward}')
            reward_list.append(test_reward)
            if test_reward > threshold:
                print("Solved!")
                early_stop = True
                break
    return early_stop, reward_list

if __name__ == "__main__":
    # Run the training function
    threshold = 400

    early_stop, reward_list = train(env_name='CartPole-v1', num_steps=10000, mini_batch_size=16, ppo_epochs=4, threshold=threshold)