import pygame
import imageio
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
from prep_distribution_env import PrEPDistributionEnv
import os
from datetime import datetime

# Initialize Pygame
pygame.init()
screen_width = 1000
screen_height = 700
screen = pygame.display.set_mode((screen_width, screen_height))
font = pygame.font.SysFont("arial", 18, bold=True)
small_font = pygame.font.SysFont("arial", 14)
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# Load South Africa map image
try:
    zaf_map = pygame.image.load("ZAF.png")
    zaf_map = pygame.transform.scale(zaf_map, (600, 700))
except pygame.error as e:
    print(f"Error loading ZAF.png: {e}. Ensure the file exists in the project directory.")
    exit()

# Province polygons (KwaZulu-Natal adjusted)
province_polygons = {
    "Gauteng": [(450, 250), (470, 230), (490, 250), (470, 270)],
    "KwaZulu-Natal": [(450, 400), (470, 380), (490, 400), (470, 420)],
    "Western Cape": [(250, 600), (270, 580), (290, 600), (270, 620)],
    "Eastern Cape": [(400, 500), (420, 480), (440, 500), (420, 520)],
    "Limpopo": [(450, 100), (470, 80), (490, 100), (470, 120)],
    "Mpumalanga": [(500, 200), (520, 180), (540, 200), (520, 220)],
    "North West": [(350, 200), (370, 180), (390, 200), (370, 220)],
    "Free State": [(350, 350), (370, 330), (390, 350), (370, 370)],
    "Northern Cape": [(200, 400), (220, 380), (240, 400), (220, 420)]
}

def save_metrics(history, regions, output_dir="plots", model_name="Random"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data = {
        "Time Step": history["time_steps"],
        **{f"Coverage_{region['name']}": history["coverage"][i] for i, region in enumerate(regions)},
        **{f"Incidence_{region['name']}": history["incidence"][i] for i, region in enumerate(regions)},
    }
    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/{model_name}_metrics_{timestamp}.csv", index=False)
    print(f"Metrics saved to {output_dir}/{model_name}_metrics_{timestamp}.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    for i, region in enumerate(regions):
        ax1.plot(history["time_steps"], history["coverage"][i], label=region["name"])
        ax2.plot(history["time_steps"], history["incidence"][i], label=region["name"])
    ax1.set_title(f"Coverage Over Time ({model_name})")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Coverage")
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax2.set_title(f"Incidence Over Time ({model_name})")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Incidence")
    ax2.set_ylim(0, 0.01)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_metrics_plot_{timestamp}.png")
    plt.close()
    print(f"Metrics plot saved to {output_dir}/{model_name}_metrics_plot_{timestamp}.png")

def visualize_episode(model_name="Random", seed=42):
    env = PrEPDistributionEnv(render_dir="plots")
    obs, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0
    action_pulse = {}
    history = {
        "coverage": [[] for _ in range(env.num_regions)],
        "incidence": [[] for _ in range(env.num_regions)],
        "time_steps": [],
        "doses": [],
        "budget": [],
        "funding_dependency": [],
        "actions": []
    }
    step = 0

    pygame.display.set_caption(f"{model_name} PrEP Distribution Simulation")
    
    # Cycle through regions for dose actions (0–35: general doses)
    region_actions = [3 + i * 4 for i in range(env.num_regions)]  # Actions 3, 7, 11, ..., 35 (3000 doses per region)

    while step < env.max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                save_metrics(history, env.regions, output_dir="plots", model_name=model_name)
                pygame.quit()
                return total_reward

        # Cycle through region-specific dose actions for first 9 steps, then random
        if step < len(region_actions):
            action = region_actions[step]  # Target each region in turn
        else:
            action = env.action_space.sample()
        print(f"Step {step}: Action = {action}")
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Log and store state
        for i in range(env.num_regions):
            history["coverage"][i].append(env.coverage[i])
            history["incidence"][i].append(env.incidence[i])
            print(f"Step {step}: {env.regions[i]['name']}: Coverage={env.coverage[i]:.6f}, Incidence={env.incidence[i]:.6f}")
        history["time_steps"].append(env.time_step)
        history["doses"].append(env.doses)
        history["budget"].append(env.budget)
        history["funding_dependency"].append(env.funding_dependency)
        history["actions"].append(env.history["actions"][-1] if env.history["actions"] else "No action")

        action_text = history["actions"][-1]
        print(f"Action Text: {action_text}")
        for i, region in enumerate(env.regions):
            if region["name"] in action_text and ("doses" in action_text or "Clinic" in action_text or "Cold chain" in action_text or "Awareness" in action_text):
                action_pulse[i] = 5

        screen.fill(WHITE)
        screen.blit(zaf_map, (0, 0))

        for i, region in enumerate(env.regions):
            coverage = env.coverage[i]
            incidence = env.incidence[i]
            # Gradual red-to-green color transition
            red = int(255 * (1 - coverage))
            green = int(255 * coverage)
            color = (red, green, 0)
            polygon = province_polygons[region["name"]]
            centroid_x = sum(x for x, _ in polygon) // 4
            centroid_y = sum(y for _, y in polygon) // 4
            if i in action_pulse and action_pulse[i] > 0:
                scale = 1 + 0.05 * np.sin(step * 0.5)
                scaled_polygon = [(centroid_x + (x - centroid_x) * scale, centroid_y + (y - centroid_y) * scale) for x, y in polygon]
                pygame.draw.polygon(screen, YELLOW, scaled_polygon, 3)
                action_pulse[i] -= 1
            else:
                action_pulse.pop(i, None)
            outline_thickness = int(5 * (incidence / 0.01))
            pygame.draw.polygon(screen, BLACK, polygon, outline_thickness)
            pygame.draw.polygon(screen, color, polygon)
            pygame.draw.circle(screen, RED, (centroid_x, centroid_y), 5)
            label = font.render(region["name"], True, BLACK)
            label_rect = label.get_rect(center=(centroid_x, centroid_y - 10))
            screen.blit(label, label_rect)
            inc_label = small_font.render(f"I:{incidence:.6f}", True, BLACK)
            inc_label_rect = inc_label.get_rect(center=(centroid_x, centroid_y + 20))
            screen.blit(inc_label, inc_label_rect)
            cov_label = small_font.render(f"C:{coverage:.6f}", True, BLACK)
            cov_label_rect = cov_label.get_rect(center=(centroid_x, centroid_y + 40))
            screen.blit(cov_label, cov_label_rect)

        pygame.draw.rect(screen, GRAY, (600, 0, 400, 700))
        screen.blit(font.render(f"{model_name} Simulation", True, BLACK), (610, 10))
        screen.blit(font.render(f"Time Step: {env.time_step}/{env.max_steps}", True, BLACK), (610, 40))
        screen.blit(font.render(f"Total Reward: {total_reward:.2f}", True, BLACK), (610, 70))
        screen.blit(font.render(f"Doses: {env.doses:.0f}/{env.max_doses}", True, BLACK), (610, 100))
        screen.blit(font.render(f"Budget: {env.budget:.0f}/{env.max_budget}", True, BLACK), (610, 130))

        doses_ratio = np.clip(env.doses / env.max_doses if env.max_doses != 0 else 0, 0, 1)
        budget_ratio = np.clip(env.budget / env.max_budget if env.max_budget != 0 else 0, 0, 1)
        doses_color = RED if env.doses < 5000 else BLUE
        budget_color = RED if env.budget < 50000 else GREEN
        pygame.draw.rect(screen, doses_color, (610, 150, 200 * doses_ratio, 20))
        pygame.draw.rect(screen, budget_color, (610, 180, 200 * budget_ratio, 20))

        if env.doses < 5000 or env.budget < 50000:
            screen.blit(font.render("WARNING: Low Resources!", True, RED), (610, 210))
        screen.blit(font.render(f"Funding Dependency: {env.funding_dependency:.2f}", True, BLACK), (610, 240))
        action_color = RED if action_text.startswith("Failed") else BLACK
        screen.blit(font.render(f"Action: {action_text[:40]}", True, action_color), (610, 270))

        y_offset = 300
        screen.blit(font.render("Region Details (Gauteng):", True, BLACK), (610, y_offset))
        y_offset += 30
        for key, value in env.regions[0].items():
            if key not in ["pos", "name"]:
                screen.blit(small_font.render(f"{key}: {value:.2f}", True, BLACK), (610, y_offset))
                y_offset += 20

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3))
        for i, region in enumerate(env.regions):
            ax1.plot(history["coverage"][i], label=region["name"][:3])
            ax2.plot(history["incidence"][i])
        ax1.set_title("Coverage Over Time")
        ax2.set_title("Incidence Over Time")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim(0, 1.0)
        ax2.set_ylim(0, 0.01)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        plot_surface = pygame.image.load(buf)
        plot_surface = pygame.transform.scale(plot_surface, (350, 250))
        screen.blit(plot_surface, (610, 400))

        screen.blit(small_font.render("Legend:", True, BLACK), (610, 660))
        pygame.draw.rect(screen, RED, (610, 680, 20, 10))
        screen.blit(small_font.render("Low Coverage", True, BLACK), (635, 680))
        pygame.draw.rect(screen, GREEN, (710, 680, 20, 10))
        screen.blit(small_font.render("High Coverage", True, BLACK), (735, 680))
        screen.blit(small_font.render("Thick Outline: High Incidence", True, BLACK), (610, 700))

        frame = pygame.surfarray.array3d(screen)
        frame = frame.transpose([1, 0, 2])
        frames.append(frame)

        pygame.display.flip()
        clock.tick(5)
        obs = next_obs
        step += 1
        if done:
            break

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    gif_path = f"plots/{model_name}_simulation_{timestamp}.gif"
    imageio.mimsave(gif_path, frames, fps=5)
    print(f"GIF saved to {gif_path}")
    save_metrics(history, env.regions, output_dir="plots", model_name=model_name)
    env.close()
    pygame.quit()
    return total_reward

if __name__ == "__main__":
    visualize_episode(model_name="Random", seed=42)

# Cleanup on script exit
import atexit
atexit.register(pygame.quit)