"""Quick standalone test — does NOT import openenv (avoids slow startup)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

t0 = time.time()

# Test pure Python modules only
from server.primary_user import PrimaryUserSimulator
from server.channel_model import compute_reward, compute_snr, MAX_THROUGHPUT_MBPS
from server.tasks import task1_find_quiet, task2_maximize_tput, task3_coexist_pu
print(f"Core imports: {time.time()-t0:.2f}s")

import random

print("\n=== Primary User Simulator ===")
pu = PrimaryUserSimulator(n_channels=8, difficulty="easy", seed=42)
pu.reset()
active = pu.step()
print(f"Easy PU step: {[int(a) for a in active]}")
pu2 = PrimaryUserSimulator(n_channels=8, difficulty="hard", seed=42)
pu2.reset()
active2 = pu2.step()
print(f"Hard PU step: {[int(a) for a in active2]}")

print("\n=== Channel Model ===")
snr = compute_snr(tx_power_dbm=20.0, channel_id=3)
tput = MAX_THROUGHPUT_MBPS
print(f"SNR (ch3, 20dBm): {snr:.2f}  Max throughput: {tput:.4f} Mbps")
reward, throughput, collision = compute_reward(3, 20.0, [False]*8, snr)
print(f"Reward (no collision): {reward:.4f}  Throughput: {throughput:.4f}")
reward2, throughput2, collision2 = compute_reward(3, 20.0, [False,False,False,True]+[False]*4, snr)
print(f"Reward (collision):    {reward2:.4f}  Collision: {collision2}")

print("\n=== Grader Variance (anti-DQ check) ===")
rng = random.Random(99)
for task_mod, label in [(task1_find_quiet,"T1"), (task2_maximize_tput,"T2"), (task3_coexist_pu,"T3")]:
    # Random behavior
    score_rand = task_mod.grade(
        collision_count=int(task_mod.MAX_STEPS * 0.35),
        total_steps=task_mod.MAX_STEPS,
        total_throughput=MAX_THROUGHPUT_MBPS * task_mod.MAX_STEPS * 0.15,
        theoretical_max_throughput=MAX_THROUGHPUT_MBPS * task_mod.MAX_STEPS,
    )
    # Smart behavior
    score_smart = task_mod.grade(
        collision_count=int(task_mod.MAX_STEPS * 0.03),
        total_steps=task_mod.MAX_STEPS,
        total_throughput=MAX_THROUGHPUT_MBPS * task_mod.MAX_STEPS * 0.85,
        theoretical_max_throughput=MAX_THROUGHPUT_MBPS * task_mod.MAX_STEPS,
    )
    print(f"  {label}: random={score_rand:.3f}  smart={score_smart:.3f}  varies={'YES ✓' if score_rand != score_smart else 'NO ✗ DQ!'}")

print(f"\nTotal time: {time.time()-t0:.2f}s  — BLOCK 1 CORE LOGIC OK")
