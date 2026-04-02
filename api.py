from fastapi import FastAPI
from stable_baselines3 import PPO
from environment.custom_env import BusinessLocationEnv

app = FastAPI()

env = BusinessLocationEnv()
model = PPO.load("models/ppo/business_ppo")

action_names = {
    0: "Inspect Location 1",
    1: "Inspect Location 2",
    2: "Inspect Location 3",
    3: "Inspect Location 4",
    4: "Inspect Location 5",
    5: "Improve Accessibility",
    6: "Invest in Marketing",
    7: "Negotiate Rent",
    8: "Open Business",
    9: "Reject Location"
}

@app.get("/")
def home():
    return {"message": "Business Location RL API is running"}

@app.get("/predict")
def predict():
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    return {
        "observation": obs.tolist(),
        "action_id": int(action),
        "action_name": action_names[int(action)]
    }