import os
import json
from openai import OpenAI

# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

SYSTEM_PROMPT = """\
You are a cognitive radio agent controlling a Software-Defined Radio (SDR).
Your goal is to transmit data on RF channels that are NOT occupied by Primary Users.

Rules:
- CRITICAL: ALWAYS select the channel with the absolute lowest occupancy near 0.0.
- NEVER transmit on a channel if its occupancy is > 0.5. That is a collision.
- If the current task is medium or hard, primary users might sweep or swap channels. Ensure you always look at the current text observation, ignoring past channel success if it is now occupied.
- Set tx_power_dbm to 25.0 if the channel is completely free (0.00), otherwise use 15.0 to avoid interference.

Respond ONLY with a valid JSON object. No reasoning. No markdown blocks. Example:
{"channel_id": 2, "tx_power_dbm": 25.0}
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
    from server.smart_radio_environment import SmartRadioEnvironment
    from models import SpectrumAction

    TASKS = [1, 2, 3]
    SEEDS = [42, 123, 456]

    for i in range(len(TASKS)):
        task_id = TASKS[i]
        seed = SEEDS[i]

        env = SmartRadioEnvironment(task_id=task_id)
        obs = env.reset(seed=seed)
        task_name = env.state.task_name.replace(" ", "_") # ensure no spaces for the grader metric value

        # [START] task=<task_name> env=<benchmark> model=<model_name>
        print(f"[START] task={task_name} env=smart_radio model={MODEL_NAME}", flush=True)

        step_count = 0
        rewards_list = []
        
        while not obs.done:
            obs_text = obs.to_text()
            error_msg = "null"
            action_str = ""
            
            try:
                action_dict = llm_act(obs_text)
                action = SpectrumAction(
                    channel_id=int(action_dict.get("channel_id", 0)),
                    tx_power_dbm=float(action_dict.get("tx_power_dbm", 20.0)),
                )
                action_str = f"ch:{action.channel_id}_pwr:{action.tx_power_dbm}"
            except Exception as e:
                error_msg = str(e).replace('\n', ' ')
                action = SpectrumAction(channel_id=0, tx_power_dbm=20.0) # fallback
                action_str = f"ch:{action.channel_id}_pwr:{action.tx_power_dbm}"

            # execute step
            try:
                obs = env.step(action)
                step_reward = obs.last_reward
                rewards_list.append(step_reward)
            except Exception as e:
                error_msg = str(e).replace('\n', ' ')
                step_reward = 0.0
                rewards_list.append(step_reward)
                obs.done = True  # force end on exception for safety
            
            step_count += 1
            done_str = "true" if obs.done else "false"
            
            # [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
            print(f"[STEP] step={step_count} action={action_str} reward={step_reward:.2f} done={done_str} error={error_msg}", flush=True)

        # Post-episode
        state = env.state
        success_str = "true" if state.grader_score > 0.5 else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)

        # [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
        print(f"[END] success={success_str} steps={step_count} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    run_inference()
