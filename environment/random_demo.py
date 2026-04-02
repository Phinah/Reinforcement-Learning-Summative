from custom_env import BusinessLocationEnv
import time
import random

env = BusinessLocationEnv()
obs, _ = env.reset()

done = False
while not done:
    env.render()
    action = random.randint(0, 9)
    print(f"Random action: {action}")
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"Reward: {reward}")
    done = terminated or truncated
    time.sleep(1)

print("Demo finished.")