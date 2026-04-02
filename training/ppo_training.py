import os
import sys
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.custom_env import BusinessLocationEnv


class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_file, verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.episode_data = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                episode_num = len(self.episode_data) + 1
                self.episode_data.append([episode_num, ep_reward, ep_length])
        return True

    def _on_training_end(self) -> None:
        os.makedirs("results/logs", exist_ok=True)
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "length"])
            writer.writerows(self.episode_data)


os.makedirs("models/ppo", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

env = BusinessLocationEnv()

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    gamma=0.95,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=1
)

callback = RewardLoggerCallback("results/logs/ppo_rewards.csv")

model.learn(total_timesteps=5000, callback=callback)
model.save("models/ppo/business_ppo")
print("PPO training complete and rewards logged.")