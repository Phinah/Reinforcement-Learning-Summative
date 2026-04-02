import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from environment.rendering import BusinessLocationGUI
except:
    BusinessLocationGUI = None


class BusinessLocationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, use_gui=False):
        super().__init__()

        self.num_locations = 5
        self.max_steps = 15
        self.initial_budget = 100

        self.action_space = spaces.Discrete(10)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32
        )

        self.locations = None
        self.current_idx = 0
        self.remaining_budget = self.initial_budget
        self.steps = 0
        self.opened = False
        self.last_action = "None"
        self.last_reward = 0

        self.use_gui = use_gui
        self.gui = BusinessLocationGUI() if use_gui and BusinessLocationGUI else None

    def _generate_locations(self):
        locations = []
        for _ in range(self.num_locations):
            locations.append({
                "demand": np.random.uniform(0.2, 1.0),
                "rent": np.random.uniform(0.2, 1.0),
                "competition": np.random.uniform(0.1, 1.0),
                "accessibility": np.random.uniform(0.2, 1.0),
                "infrastructure": np.random.uniform(0.2, 1.0),
                "safety": np.random.uniform(0.2, 1.0),
            })
        return locations

    def _get_obs(self):
        loc = self.locations[self.current_idx]
        return np.array([
            loc["demand"],
            loc["rent"],
            loc["competition"],
            loc["accessibility"],
            loc["infrastructure"],
            loc["safety"],
            self.remaining_budget / self.initial_budget,
            1.0 - (self.steps / self.max_steps),
            self.current_idx / (self.num_locations - 1),
            float(self.opened)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.locations = self._generate_locations()
        self.current_idx = 0
        self.remaining_budget = self.initial_budget
        self.steps = 0
        self.opened = False
        self.last_action = "Reset"
        self.last_reward = 0
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        reward = 0
        terminated = False
        truncated = False

        self.steps += 1
        loc = self.locations[self.current_idx]

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
        self.last_action = action_names.get(action, "Unknown Action")

        if action in [0, 1, 2, 3, 4]:
            self.current_idx = action
            reward = 1

        elif action == 5:
            self.remaining_budget -= 10
            if loc["accessibility"] < 0.7:
                loc["accessibility"] = min(1.0, loc["accessibility"] + 0.15)
                reward = 2
            else:
                reward = -1

        elif action == 6:
            self.remaining_budget -= 10
            if loc["demand"] < 0.7:
                loc["demand"] = min(1.0, loc["demand"] + 0.15)
                reward = 2
            else:
                reward = -1

        elif action == 7:
            self.remaining_budget -= 5
            if loc["rent"] > 0.4:
                loc["rent"] = max(0.0, loc["rent"] - 0.1)
                reward = 2
            else:
                reward = -1

        elif action == 8:
            # Penalize opening too early
            if self.steps < 2:
                reward = -5  # discourage immediate opening
            else:
                score = (
                    12 * loc["demand"]
                    - 10 * loc["competition"]
                    - 8 * loc["rent"]
                    + 7 * loc["accessibility"]
                    + 6 * loc["infrastructure"]
                    + 5 * loc["safety"]
                )

                if score >= 8:
                    reward = 25
                elif score >= 3:
                    reward = 10
                else:
                    reward = -15

            self.opened = True
            terminated = True

        elif action == 9:
            bad_site = (loc["competition"] > 0.7 and loc["rent"] > 0.7)
            reward = 2 if bad_site else 0
            self.current_idx = (self.current_idx + 1) % self.num_locations

        if self.remaining_budget <= 0:
            reward -= 10
            terminated = True

        if self.steps >= self.max_steps:
            reward -= 5
            truncated = True

        self.last_reward = reward
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        loc = self.locations[self.current_idx]
        print("\n--- Business Location Environment ---")
        print(f"Step: {self.steps}/{self.max_steps}")
        print(f"Current Location: {self.current_idx + 1}")
        print(f"Budget: {self.remaining_budget}")
        print(f"Demand: {loc['demand']:.2f}")
        print(f"Rent: {loc['rent']:.2f}")
        print(f"Competition: {loc['competition']:.2f}")
        print(f"Accessibility: {loc['accessibility']:.2f}")
        print(f"Infrastructure: {loc['infrastructure']:.2f}")
        print(f"Safety: {loc['safety']:.2f}")

        if self.gui:
            self.gui.update(self, self.last_action, self.last_reward)

    def close(self):
        if self.gui:
            self.gui.close()