import os
import random
from server.environment import AdaptiveUIEnv
from models import UIAction

def run_tests():
    tasks = ["ui_easy_retention", "ui_med_conversion", "ui_hard_fatigue"]
    report = []

    for task_id in tasks:
        env = AdaptiveUIEnv(task_id=task_id)
        obs = env.reset()
        
        step_count = 0
        rewards = []
        annoyances = []
        engagements = []
        organics = []

        # A decent hardcoded heuristic policy to show good results
        while not obs.done and step_count < 30:
            if obs.observation.annoyance_level > 0.4:
                # calm user down
                act = UIAction(layout="list", cta_style="subtle", color_scheme="dark", font_size="small")
            elif obs.observation.engagement_score < 0.8:
                # push for engagement
                act = UIAction(layout="grid", cta_style="bold", color_scheme="light", font_size="large")
            else:
                # maintain
                act = UIAction(layout="grid", cta_style="subtle", color_scheme="light", font_size="medium")
                
            obs = env.step(act)
            rewards.append(obs.reward)
            annoyances.append(obs.observation.annoyance_level)
            engagements.append(obs.observation.engagement_score)
            organics.append(obs.observation.organic_exit_risk)
            step_count += 1
            
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        
        report.append(f"\n### Task: {task_id}")
        report.append(f"- **Steps Survived**: {step_count}")
        report.append(f"- **Final Average Score**: {avg_reward:.4f}")
        report.append(f"- **Peak Annoyance Hit**: {max(annoyances):.2f}")
        report.append(f"- **Peak Engagement Hit**: {max(engagements):.2f}")
        report.append(f"- **Final Organic Exit Risk**: {organics[-1]:.2f}")
        report.append(f"- **Termination Cause**: {'Goal Achieved/Finished' if engagements[-1] > 0.8 else 'Rage Quit/Boredom'}")
        
    print("\n".join(report))

if __name__ == "__main__":
    run_tests()
