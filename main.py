import os
import time
import torch
import torch.nn as nn
from stable_baselines3 import PPO, DQN, A2C

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


def run_dqn():
    model_path = "models/dqn/business_dqn.zip"
    if not os.path.exists(model_path):
        print(f"DQN model not found: {model_path}")
        print("Run this first: python training/dqn_training.py")
        return

    env = BusinessLocationEnv(use_gui=True)
    model = DQN.load("models/dqn/business_dqn")

    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        print("Action:", action, "Reward:", reward)
        time.sleep(1)
        done = terminated or truncated


def run_ppo():
    model_path = "models/ppo/business_ppo.zip"
    if not os.path.exists(model_path):
        print(f"PPO model not found: {model_path}")
        print("Run this first: python training/ppo_training.py")
        return

    env = BusinessLocationEnv(use_gui=True)
    model = PPO.load("models/ppo/business_ppo")

    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        print("Action:", action, "Reward:", reward)
        time.sleep(1)
        done = terminated or truncated


def run_a2c():
    model_path = "models/a2c/business_a2c.zip"
    if not os.path.exists(model_path):
        print(f"A2C model not found: {model_path}")
        print("Run this first: python training/a2c_training.py")
        return

    env = BusinessLocationEnv(use_gui=True)
    model = A2C.load("models/a2c/business_a2c")

    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        print("Action:", action, "Reward:", reward)
        time.sleep(1)
        done = terminated or truncated


def run_reinforce():
    model_path = "models/reinforce/business_reinforce.pth"
    if not os.path.exists(model_path):
        print(f"REINFORCE model not found: {model_path}")
        print("Run this first: python training/reinforce_training.py")
        return

    env = BusinessLocationEnv(use_gui=True)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, action_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = policy(obs_tensor)
        action = torch.argmax(probs, dim=1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        print("Action:", action, "Reward:", reward)
        time.sleep(1)
        done = terminated or truncated


if __name__ == "__main__":
    print("1 = DQN demo")
    print("2 = PPO demo")
    print("3 = A2C demo")
    print("4 = REINFORCE demo")
    choice = input("Choose model: ")

    if choice == "1":
        run_dqn()
    elif choice == "2":
        run_ppo()
    elif choice == "3":
        run_a2c()
    elif choice == "4":
        run_reinforce()
    else:
        print("Invalid choice.")