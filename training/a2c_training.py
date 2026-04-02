import os
import sys
import csv
from stable_baselines3 import A2C
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


os.makedirs("models/a2c", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

env = BusinessLocationEnv()

model = A2C(
    "MlpPolicy",
    env,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.95,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=1
)

callback = RewardLoggerCallback("results/logs/a2c_rewards.csv")

model.learn(total_timesteps=5000, callback=callback)
model.save("models/a2c/business_a2c")
print("A2C training complete and rewards logged.")