import os
import json
from openai import OpenAI

# Read environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

SYSTEM_PROMPT = """\
You are an AI tasked with dynamically adjusting a web application's user interface.
Your goal is to maximize the user's engagement and achieve a CTA click, while keeping annoyance and organic exit risk low.

Rules:
- Read the current observation representing `annoyance_level`, `engagement_score`, and `organic_exit_risk`.
- If annoyance is high (>0.5), use "subtle", "list" and "dark" actions to calm the user.
- If organic exit risk is rising but annoyance is low, push a "pulsing" CTA on a "hero" layout to grab attention.
- Options for layout: "grid", "list", "hero".
- Options for cta_style: "subtle", "bold", "pulsing".
- Options for color_scheme: "light", "dark", "high-contrast".
- Options for font_size: "small", "medium", "large".

Respond critically ONLY with a valid JSON format. Example:
{"layout": "grid", "cta_style": "bold", "color_scheme": "light", "font_size": "medium"}
"""

def llm_act(obs_text: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()
    return json.loads(raw)

def run_inference():
    from server.environment import AdaptiveUIEnv
    from models import UIAction

    TASKS = ["ui_easy_retention", "ui_med_conversion", "ui_hard_fatigue"]

    for task_id in TASKS:
        env = AdaptiveUIEnv(task_id=task_id)
        obs = env.reset()
        
        # [START] structured log
        print(f"[START] task={task_id} env=adaptive_ui model={MODEL_NAME}", flush=True)

        step_count = 0
        rewards_list = []
        
        while not obs.done and step_count < 30:
            # text represent of observation
            obs_text = f"Annoyance: {obs.observation.annoyance_level:.2f}, Engagement: {obs.observation.engagement_score:.2f}, Organic Risk: {obs.observation.organic_exit_risk:.2f}"
            
            error_msg = "null"
            action_str = ""
            
            try:
                action_dict = llm_act(obs_text)
                action = UIAction(
                    layout=action_dict.get("layout", "grid"),
                    cta_style=action_dict.get("cta_style", "subtle"),
                    color_scheme=action_dict.get("color_scheme", "light"),
                    font_size=action_dict.get("font_size", "medium")
                )
                action_str = f"lay:{action.layout}_cta:{action.cta_style}_col:{action.color_scheme}_fnt:{action.font_size}"
            except Exception as e:
                error_msg = str(e).replace('\n', ' ')
                action = UIAction(layout="grid", cta_style="subtle", color_scheme="light", font_size="medium") # fallback
                action_str = f"lay:{action.layout}_cta:{action.cta_style}_col:{action.color_scheme}_fnt:{action.font_size}"

            # Step
            try:
                obs = env.step(action)
                step_reward = obs.reward
            except Exception as e:
                error_msg = str(e).replace('\n', ' ')
                step_reward = 0.50
                obs.done = True 
            
            rewards_list.append(step_reward)
            step_count += 1
            done_str = "true" if obs.done else "false"
            
            # Formatted log
            print(f"[STEP] step={step_count} action={action_str} reward={step_reward:.2f} done={done_str} error={error_msg}", flush=True)

        # Average step reward acting as grader total proxy locally
        final_score = sum(rewards_list) / len(rewards_list) if rewards_list else 0.5
        
        adjusted_rewards = [f"{float(r):.2f}" for r in rewards_list]
        rewards_str = ",".join(adjusted_rewards)

        # [END] structured log
        success_str = "true" if final_score > 0.8 else "false"
        print(f"[END] success={success_str} steps={step_count} score={final_score:.2f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    run_inference()
