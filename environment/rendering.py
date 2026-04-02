import tkinter as tk


class BusinessLocationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Business Location RL Environment")
        self.root.geometry("700x500")

        self.title_label = tk.Label(
            self.root,
            text="Business Location Intelligence Environment",
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(pady=10)

        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(pady=10)

        self.step_label = tk.Label(self.info_frame, text="Step: ", font=("Arial", 12))
        self.step_label.pack(anchor="w")

        self.budget_label = tk.Label(self.info_frame, text="Budget: ", font=("Arial", 12))
        self.budget_label.pack(anchor="w")

        self.location_label = tk.Label(self.info_frame, text="Current Location: ", font=("Arial", 12))
        self.location_label.pack(anchor="w")

        self.stats_frame = tk.Frame(self.root, bd=2, relief="groove", padx=10, pady=10)
        self.stats_frame.pack(pady=10, fill="x", padx=20)

        self.demand_label = tk.Label(self.stats_frame, text="Demand: ", font=("Arial", 12))
        self.demand_label.pack(anchor="w")

        self.rent_label = tk.Label(self.stats_frame, text="Rent: ", font=("Arial", 12))
        self.rent_label.pack(anchor="w")

        self.competition_label = tk.Label(self.stats_frame, text="Competition: ", font=("Arial", 12))
        self.competition_label.pack(anchor="w")

        self.accessibility_label = tk.Label(self.stats_frame, text="Accessibility: ", font=("Arial", 12))
        self.accessibility_label.pack(anchor="w")

        self.infrastructure_label = tk.Label(self.stats_frame, text="Infrastructure: ", font=("Arial", 12))
        self.infrastructure_label.pack(anchor="w")

        self.safety_label = tk.Label(self.stats_frame, text="Safety: ", font=("Arial", 12))
        self.safety_label.pack(anchor="w")

        self.action_label = tk.Label(
            self.root,
            text="Last Action: None",
            font=("Arial", 12, "bold")
        )
        self.action_label.pack(pady=10)

        self.reward_label = tk.Label(
            self.root,
            text="Reward: 0",
            font=("Arial", 12, "bold")
        )
        self.reward_label.pack(pady=5)

        self.status_label = tk.Label(
            self.root,
            text="Environment Ready",
            font=("Arial", 11),
            fg="blue"
        )
        self.status_label.pack(pady=10)

    def update(self, env, action_text="None", reward=0):
        loc = env.locations[env.current_idx]

        self.step_label.config(text=f"Step: {env.steps}/{env.max_steps}")
        self.budget_label.config(text=f"Budget: {env.remaining_budget}")
        self.location_label.config(text=f"Current Location: {env.current_idx + 1}")

        self.demand_label.config(text=f"Demand: {loc['demand']:.2f}")
        self.rent_label.config(text=f"Rent: {loc['rent']:.2f}")
        self.competition_label.config(text=f"Competition: {loc['competition']:.2f}")
        self.accessibility_label.config(text=f"Accessibility: {loc['accessibility']:.2f}")
        self.infrastructure_label.config(text=f"Infrastructure: {loc['infrastructure']:.2f}")
        self.safety_label.config(text=f"Safety: {loc['safety']:.2f}")

        self.action_label.config(text=f"Last Action: {action_text}")
        self.reward_label.config(text=f"Reward: {reward}")

        self.root.update_idletasks()
        self.root.update()

    def close(self):
        self.root.destroy()