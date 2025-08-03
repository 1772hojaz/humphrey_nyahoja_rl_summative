import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PrEPDistributionEnv(gym.Env):
    def __init__(self, render_dir="plots"):
        super(PrEPDistributionEnv, self).__init__()
        self.num_regions = 9
        self.max_doses = 100000  # Increased for testing
        self.max_budget = 1000000  # Increased for testing
        self.cost_per_dose = 6
        self.efficacy_general = 0.95
        self.efficacy_key = 0.99
        self.max_steps = 20
        self.render_dir = render_dir
        if not os.path.exists(render_dir):
            os.makedirs(render_dir)
        
        # State space
        self.observation_space = spaces.Box(
            low=np.array([10000] * 9 + [0.05] * 9 + [0.001] * 9 + [0.1] * 9 +
                         [0.1] * 9 + [0] * 9 + [1] * 9 + [0.5] * 9 +
                         [0, 0, 0, 1], dtype=np.float32),
            high=np.array([2000000] * 9 + [0.35] * 9 + [0.01] * 9 + [0.9] * 9 +
                          [0.9] * 9 + [1] * 9 + [50] * 9 + [1] * 9 +
                          [self.max_doses, self.max_budget, 1, self.max_steps], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space
        self.action_space = spaces.Discrete(99)
        
        # Region data
        self.regions = [
            {"name": "Gauteng", "P": 150000, "H": 0.15, "I": 0.008, "A": 0.2, "S": 0.3, "K": 40, "R": 0.95, "pos": (470, 250)},
            {"name": "KwaZulu-Natal", "P": 180000, "H": 0.25, "I": 0.010, "A": 0.5, "S": 0.4, "K": 30, "R": 0.85, "pos": (470, 400)},
            {"name": "Western Cape", "P": 90000, "H": 0.10, "I": 0.005, "A": 0.3, "S": 0.2, "K": 35, "R": 0.90, "pos": (270, 600)},
            {"name": "Eastern Cape", "P": 100000, "H": 0.20, "I": 0.007, "A": 0.6, "S": 0.5, "K": 25, "R": 0.80, "pos": (420, 500)},
            {"name": "Limpopo", "P": 80000, "H": 0.12, "I": 0.006, "A": 0.8, "S": 0.6, "K": 15, "R": 0.75, "pos": (470, 100)},
            {"name": "Mpumalanga", "P": 70000, "H": 0.18, "I": 0.008, "A": 0.5, "S": 0.4, "K": 20, "R": 0.85, "pos": (520, 200)},
            {"name": "North West", "P": 60000, "H": 0.15, "I": 0.006, "A": 0.7, "S": 0.5, "K": 18, "R": 0.80, "pos": (370, 200)},
            {"name": "Free State", "P": 50000, "H": 0.17, "I": 0.007, "A": 0.6, "S": 0.4, "K": 22, "R": 0.85, "pos": (370, 350)},
            {"name": "Northern Cape", "P": 30000, "H": 0.08, "I": 0.004, "A": 0.9, "S": 0.3, "K": 10, "R": 0.70, "pos": (220, 400)}
        ]
        
        # History for visualization
        self.history = {
            "coverage": [], "incidence": [], "doses": [], "budget": [], "funding_dependency": [], "time_steps": [], "actions": []
        }
        self._np_random = None  # For seeding
        self.reset()

    def seed(self, seed=None):
        """Set the random seed for reproducibility."""
        self._np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
        
    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self.populations = np.array([r["P"] for r in self.regions], dtype=np.float32)
        self.hiv_prevalence = np.array([r["H"] for r in self.regions], dtype=np.float32)
        self.incidence = np.array([r["I"] for r in self.regions], dtype=np.float32)
        self.access_difficulty = np.array([r["A"] for r in self.regions], dtype=np.float32)
        self.stigma = np.array([r["S"] for r in self.regions], dtype=np.float32)
        self.coverage = np.array([0.1 if r["name"] == "KwaZulu-Natal" else 0.0 for r in self.regions], dtype=np.float32)  # Initial coverage for KwaZulu-Natal
        self.clinic_capacity = np.array([r["K"] for r in self.regions], dtype=np.int32)
        self.cold_chain = np.array([r["R"] for r in self.regions], dtype=np.float32)
        self.doses = self.max_doses
        self.budget = self.max_budget
        self.funding_dependency = 0.8
        self.time_step = 1
        self.history = {
            "coverage": [self.coverage.copy()],
            "incidence": [self.incidence.copy()],
            "doses": [self.doses],
            "budget": [self.budget],
            "funding_dependency": [self.funding_dependency],
            "time_steps": [self.time_step],
            "actions": []
        }
        logging.debug(f"Reset: Doses={self.doses}, Budget={self.budget}")
        for i, r in enumerate(self.regions):
            logging.debug(f"Reset {r['name']}: Coverage={self.coverage[i]:.6f}, Incidence={self.incidence[i]:.6f}")
        return self._get_state(), {}
    
    def _get_state(self):
        state = np.concatenate([
            self.populations, self.hiv_prevalence, self.incidence, self.access_difficulty,
            self.stigma, self.coverage, self.clinic_capacity.astype(np.float32), self.cold_chain,
            [self.doses, self.budget, self.funding_dependency, self.time_step]
        ]).astype(np.float32)
        if not np.all(np.isfinite(state)):
            logging.error(f"Invalid state: {state}")
        return state
    
    def step(self, action):
        doses_general = np.zeros(self.num_regions, dtype=np.float32)
        doses_key = np.zeros(self.num_regions, dtype=np.float32)
        costs = 0
        investments = {
            "K": np.zeros(self.num_regions, dtype=np.int32),
            "R": np.zeros(self.num_regions, dtype=np.float32),
            "S": np.zeros(self.num_regions, dtype=np.float32)
        }
        action_desc = "Action"
        
        # Decode action
        if action < 36:
            region_idx = action // 4
            dose_level = [500, 1000, 2000, 3000][action % 4]
            doses_general[region_idx] = dose_level
            action_desc = f"General: {dose_level} doses to {self.regions[region_idx]['name']}"
        elif action < 63:
            region_idx = (action - 36) // 3
            dose_level = [250, 500, 750][(action - 36) % 3]
            doses_key[region_idx] = dose_level
            action_desc = f"Key: {dose_level} doses to {self.regions[region_idx]['name']}"
        elif action < 81:
            region_idx = (action - 63) // 2
            if action % 2 == 0:
                investments["K"][region_idx] = 1
                costs += 2000
                action_desc = f"Clinic capacity +1 in {self.regions[region_idx]['name']}"
            else:
                investments["R"][region_idx] = 0.1
                costs += 1500
                action_desc = f"Cold chain +0.1 in {self.regions[region_idx]['name']}"
        elif action < 90:
            region_idx = action - 81
            investments["S"][region_idx] = 0.1
            costs += 1000
            action_desc = f"Awareness campaign in {self.regions[region_idx]['name']}"
        elif action < 99:
            pattern_idx = action - 90
            if pattern_idx == 0:
                urban = [0, 2, 3]
                doses_general[urban] = 1500
                action_desc = "Urban focus: 1500 doses to Gauteng, Western Cape, Eastern Cape"
            elif pattern_idx == 1:
                top = np.argsort(self.incidence)[-3:]
                doses_general[top] = 2000
                action_desc = f"High-incidence: 2000 doses to {', '.join(self.regions[i]['name'] for i in top)}"
            elif pattern_idx == 2:
                doses_key[:] = 250
                action_desc = "Key population sweep: 250 doses to all"
            elif pattern_idx == 3:
                doses_general[:] = 750
                action_desc = "Balanced: 750 doses to all"
            elif pattern_idx == 4:
                top = np.argsort(self.incidence)[-2:]
                doses_general[top] = 2500
                action_desc = f"Top-incidence: 2500 doses to {', '.join(self.regions[i]['name'] for i in top)}"
            elif pattern_idx == 5:
                investments["S"][:] = 0.05
                costs += 6000
                action_desc = "Nationwide awareness campaign"
            elif pattern_idx == 6:
                investments["R"][:] = 0.05
                costs += 9000
                action_desc = "Nationwide cold chain improvement"
            elif pattern_idx == 7:
                investments["K"][:] = 1
                costs += 12000
                action_desc = "Nationwide clinic capacity increase"
            elif pattern_idx == 8:
                high_access = np.argsort(self.access_difficulty)[-3:]
                doses_general[high_access] = 1500
                action_desc = f"High-access-difficulty: 1500 doses to {', '.join(self.regions[i]['name'] for i in high_access)}"
        
        # Calculate costs
        total_doses = np.sum(doses_general) + np.sum(doses_key)
        dose_cost = np.sum(doses_general * self.cost_per_dose * (1 + self.access_difficulty)) + \
                    np.sum(doses_key * self.cost_per_dose * (1 + self.access_difficulty))
        total_cost = dose_cost + costs
        logging.debug(f"Action {action}: {action_desc}, Doses={total_doses}, Cost={total_cost}, Budget={self.budget}")
        
        # Check constraints
        if total_doses > self.doses:
            self.history["coverage"].append(self.coverage.copy())
            self.history["incidence"].append(self.incidence.copy())
            self.history["doses"].append(self.doses)
            self.history["budget"].append(self.budget)
            self.history["funding_dependency"].append(self.funding_dependency)
            self.history["time_steps"].append(self.time_step)
            self.history["actions"].append(f"Failed: Doses exceeded ({action_desc})")
            self.funding_dependency = max(0.1, self.funding_dependency - 0.1)
            self.time_step += 1
            return self._get_state(), -50.0, self.time_step > self.max_steps, False, {"error": "Doses exceeded"}
        if total_cost > self.budget:
            self.history["coverage"].append(self.coverage.copy())
            self.history["incidence"].append(self.incidence.copy())
            self.history["doses"].append(self.doses)
            self.history["budget"].append(self.budget)
            self.history["funding_dependency"].append(self.funding_dependency)
            self.history["time_steps"].append(self.time_step)
            self.history["actions"].append(f"Failed: Budget exceeded ({action_desc})")
            self.funding_dependency = max(0.1, self.funding_dependency - 0.1)
            self.time_step += 1
            return self._get_state(), -50.0, self.time_step > self.max_steps, False, {"error": "Budget exceeded"}
        if np.any(doses_general + doses_key > self.clinic_capacity * 2000):
            self.history["coverage"].append(self.coverage.copy())
            self.history["incidence"].append(self.incidence.copy())
            self.history["doses"].append(self.doses)
            self.history["budget"].append(self.budget)
            self.history["funding_dependency"].append(self.funding_dependency)
            self.history["time_steps"].append(self.time_step)
            self.history["actions"].append(f"Failed: Clinic capacity exceeded ({action_desc})")
            self.funding_dependency = max(0.1, self.funding_dependency - 0.1)
            self.time_step += 1
            return self._get_state(), -50.0, self.time_step > self.max_steps, False, {"error": "Clinic capacity exceeded"}
        if self.funding_dependency > 0.4 and action >= 63 and action < 90:
            self.history["coverage"].append(self.coverage.copy())
            self.history["incidence"].append(self.incidence.copy())
            self.history["doses"].append(self.doses)
            self.history["budget"].append(self.budget)
            self.history["funding_dependency"].append(self.funding_dependency)
            self.history["time_steps"].append(self.time_step)
            self.history["actions"].append(f"Failed: Funding restricts ({action_desc})")
            self.funding_dependency = max(0.1, self.funding_dependency - 0.1)
            self.time_step += 1
            return self._get_state(), -50.0, self.time_step > self.max_steps, False, {"error": "Funding restricts infrastructure/awareness"}
        
        # Update state
        self.doses = max(0, float(np.clip(self.doses - total_doses, 0, self.max_doses)))
        self.budget = max(0, float(np.clip(self.budget - total_cost, 0, self.max_budget)))
        self.clinic_capacity += investments["K"]
        self.cold_chain = np.minimum(1.0, self.cold_chain + investments["R"])
        self.stigma = np.maximum(0.1, self.stigma - investments["S"])
        for i in range(self.num_regions):
            effective_doses = (doses_general[i] * self.efficacy_general + doses_key[i] * self.efficacy_key) * \
                             (1 - self.stigma[i]) * self.cold_chain[i]
            self.coverage[i] = min(1.0, self.coverage[i] + 10 * effective_doses / self.populations[i])  # Scaled for visibility
            self.incidence[i] = max(0.0, self.incidence[i] * (1 - 0.1 * effective_doses / self.populations[i]))  # Decrease incidence
        
        # Log state
        logging.debug(f"Post-step state:")
        for i, r in enumerate(self.regions):
            logging.debug(f"{r['name']}: Coverage={self.coverage[i]:.6f}, Incidence={self.incidence[i]:.6f}")
        
        # Reward calculation
        infections_prevented = np.sum(
            (doses_general * self.efficacy_general + doses_key * self.efficacy_key) *
            self.incidence * (1 - self.coverage) * (1 - self.stigma) * self.cold_chain
        )
        waste_penalty = 0.2 * np.sum((doses_general + doses_key) * (1 - self.incidence) * self.coverage)
        logistics_penalty = 0.1 * np.sum((doses_general + doses_key) * self.access_difficulty) + \
                          0.5 * np.sum(np.maximum(0, (doses_general + doses_key) - self.clinic_capacity * 2000))
        spoilage_penalty = 0.3 * np.sum((doses_general + doses_key) * (1 - self.cold_chain))
        investment_bonus = 10 * np.sum(investments["K"] + investments["R"] + investments["S"])
        reward = 100 * infections_prevented + investment_bonus - waste_penalty - logistics_penalty - spoilage_penalty
        
        # Update history
        self.history["coverage"].append(self.coverage.copy())
        self.history["incidence"].append(self.incidence.copy())
        self.history["doses"].append(self.doses)
        self.history["budget"].append(self.budget)
        self.history["funding_dependency"].append(self.funding_dependency)
        self.history["time_steps"].append(self.time_step)
        self.history["actions"].append(action_desc)
        
        # Update time step
        self.time_step += 1
        
        # Check termination
        done = self.time_step > self.max_steps
        
        # Restock resources
        if not done:
            self.doses = min(self.max_doses, self.doses + 10000)
            self.budget = min(self.max_budget, self.budget + 100000 * (1 - self.funding_dependency))
        
        logging.debug(f"Post-step: Doses={self.doses}, Budget={self.budget}, Reward={reward}")
        return self._get_state(), reward, done, False, {}
    
    def render(self):
        print(f"Time step: {self.time_step}, Doses: {self.doses:.0f}, Budget: {self.budget:.0f}, Funding Dependency: {self.funding_dependency:.2f}")
        for i, r in enumerate(self.regions):
            print(f"{r['name']}: Coverage={self.coverage[i]:.6f}, Incidence={self.incidence[i]:.6f}")
        print(f"Action: {self.history['actions'][-1] if self.history['actions'] else 'No action'}")
    
    def close(self):
        pass