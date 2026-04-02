import os
import sys
import csv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.custom_env import BusinessLocationEnv


class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_file, verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append((len(self.episode_rewards) + 1, ep_reward, ep_length))
        return True

    def _on_training_end(self):
        os.makedirs("results/logs", exist_ok=True)
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "length"])
            writer.writerows(self.episode_rewards)


# Ensure folders exist
os.makedirs("models/dqn", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

env = BusinessLocationEnv()

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=200,
    batch_size=32,
    gamma=0.95,
    exploration_fraction=0.4,
    exploration_final_eps=0.05,
    verbose=1
)

# ✅ Attach callback properly
callback = RewardLoggerCallback("results/logs/dqn_rewards.csv")

model.learn(total_timesteps=10000, callback=callback)

model.save("models/dqn/business_dqn")
print("DQN training complete and rewards logged.")