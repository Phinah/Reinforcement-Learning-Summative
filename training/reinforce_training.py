import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.custom_env import BusinessLocationEnv


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def train_reinforce(num_episodes=300, gamma=0.95, lr=1e-3):
    env = BusinessLocationEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    all_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            probs = policy(obs_tensor)
            dist = Categorical(probs)
            action = dist.sample()

            next_obs, reward, terminated, truncated, _ = env.step(action.item())

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

            obs = next_obs
            done = terminated or truncated

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)

        if len(returns) > 1 and returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        all_rewards.append(total_reward)

        if (episode + 1) % 25 == 0:
            print(f"Episode {episode + 1}, Avg Reward: {sum(all_rewards[-25:])/25:.2f}")

    # ✅ SAVE CSV HERE (inside function)
    import csv, os
    os.makedirs("results/logs", exist_ok=True)

    with open("results/logs/reinforce_rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(all_rewards, start=1):
            writer.writerow([i, r])

    # save model
    os.makedirs("models/reinforce", exist_ok=True)
    torch.save(policy.state_dict(), "models/reinforce/business_reinforce.pth")

    print("REINFORCE training complete and logs saved.")

if __name__ == "__main__":
    train_reinforce()