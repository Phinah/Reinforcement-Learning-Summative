import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)

files = {
    "DQN": "results/logs/dqn_rewards.csv",
    "PPO": "results/logs/ppo_rewards.csv",
    "A2C": "results/logs/a2c_rewards.csv",
    "REINFORCE": "results/logs/reinforce_rewards.csv",
}

# Combined plot
plt.figure(figsize=(10, 6))

for name, path in files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "reward" in df.columns:
            df["smoothed_reward"] = df["reward"].rolling(window=5, min_periods=1).mean()
            plt.plot(df["episode"], df["smoothed_reward"], label=name)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Cumulative Reward Trends Across Models")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/all_models_rewards.png")
plt.show()

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, (name, path) in zip(axes, files.items()):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "reward" in df.columns:
            df["smoothed_reward"] = df["reward"].rolling(window=5, min_periods=1).mean()
            ax.plot(df["episode"], df["smoothed_reward"])
            ax.set_title(name)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.grid(True)

plt.tight_layout()
plt.savefig("results/plots/subplots_rewards.png")
plt.show()