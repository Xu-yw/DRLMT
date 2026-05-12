import os
import numpy as np

import torch
import torch.nn as nn
from encoder_init import EncodeState
from networks.on_policy.ppo.ppo import ActorCritic
from parameters import  *

device = torch.device("cpu")

class Buffer:
    def __init__(self):
         # Batch data
        self.observation = []  
        self.actions = []         
        self.log_probs = []     
        self.rewards = []         
        self.dones = []
        self.env_ids = []

    def clear(self):
        del self.observation[:]    
        del self.actions[:]        
        del self.log_probs[:]      
        del self.rewards[:]
        del self.dones[:]
        del self.env_ids[:]

class PPOAgent(object):
    def __init__(self, town, action_std_init=0.4):
        
        #self.env = env
        self.obs_dim = 100
        self.action_dim = 2
        self.clip = POLICY_CLIP
        # self.gamma = GAMMA
        # self.n_updates_per_iteration = 7
        self.n_updates_per_iteration = 20
        self.lr = PPO_LEARNING_RATE
        self.action_std = action_std_init
        self.encode = EncodeState(LATENT_DIM)
        self.memory = Buffer()
        self.town = town

        self.checkpoint_file_no = 0
        
        self.policy = ActorCritic(self.obs_dim, self.action_dim, self.action_std).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr}])

        self.old_policy = ActorCritic(self.obs_dim, self.action_dim, self.action_std).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()


    def get_action(self, obs, flag=None, reward=None, done=None, train=True, env_id=None):

        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float)
            action, logprob = self.old_policy.get_action_and_log_prob(obs.to(device))

        if train:
            # Store state/action/logprob now; reward/done belongs to env.step(action).
            self.memory.observation.append(obs.to(device))
            self.memory.actions.append(action)
            self.memory.log_probs.append(logprob)
            self.memory.env_ids.append(env_id)

        return action.detach().cpu().numpy().flatten()

    def record_transition(self, reward, done):
        self.memory.rewards.append(reward)
        self.memory.dones.append(done)

    def discard_last_action(self):
        if self.memory.observation:
            self.memory.observation.pop()
        if self.memory.actions:
            self.memory.actions.pop()
        if self.memory.log_probs:
            self.memory.log_probs.pop()
        if self.memory.env_ids:
            self.memory.env_ids.pop()
    
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)

    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)
        return self.action_std


    def learn(self):
        lengths = {
            "observation": len(self.memory.observation),
            "actions": len(self.memory.actions),
            "log_probs": len(self.memory.log_probs),
            "rewards": len(self.memory.rewards),
            "dones": len(self.memory.dones),
            "env_ids": len(self.memory.env_ids),
        }
        if len(set(lengths.values())) != 1:
            raise RuntimeError("PPO buffer length mismatch: {}".format(lengths))
        if lengths["rewards"] == 0:
            raise RuntimeError("PPO learn called with empty buffer")
        unfinished_envs = []
        for env_id in set(self.memory.env_ids):
            last_idx = max(i for i, cur_env in enumerate(self.memory.env_ids) if cur_env == env_id)
            if not self.memory.dones[last_idx]:
                unfinished_envs.append(env_id)
        if unfinished_envs:
            raise RuntimeError("PPO learn called with unfinished env trajectories: {}".format(unfinished_envs))

        # Monte Carlo estimate of returns. In vectorized training, samples from
        # multiple CARLA servers are interleaved, so returns must reset per env.
        returns = [0.0] * len(self.memory.rewards)
        discounted_by_env = {}
        gamma = GAMMA
        env_ids = self.memory.env_ids
        if len(env_ids) != len(self.memory.rewards):
            env_ids = [None] * len(self.memory.rewards)
        for idx in range(len(self.memory.rewards) - 1, -1, -1):
            reward = self.memory.rewards[idx]
            is_terminal = self.memory.dones[idx]
            env_id = env_ids[idx]
            if is_terminal:
                discounted_by_env[env_id] = 0.0
            discounted_reward = reward + (gamma * discounted_by_env.get(env_id, 0.0))
            returns[idx] = discounted_reward
            discounted_by_env[env_id] = discounted_reward

        # Normalize returns without clipping; high-progress episodes must keep their advantage signal.
        rewards = torch.tensor(returns, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.memory.observation, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.n_updates_per_iteration):

            # Evaluating old actions and values
            logprobs, values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match values tensor dimensions with rewards tensor
            values = torch.squeeze(values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    
    def save(self):
        os.makedirs(PPO_CHECKPOINT_DIR+self.town, exist_ok=True)
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2])
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)

    def chkpt_save(self):
        os.makedirs(PPO_CHECKPOINT_DIR+self.town, exist_ok=True)
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2])
        if self.checkpoint_file_no !=0:
            self.checkpoint_file_no -=1
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)
   
    def load(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2]) - 1  #加载最后一个（最新）的pth模型
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        self.old_policy.load_state_dict(torch.load(checkpoint_file))
        self.policy.load_state_dict(torch.load(checkpoint_file))
