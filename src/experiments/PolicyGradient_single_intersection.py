import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import glob
import pandas as pd

print(sys.path)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np

project_src = os.path.join(os.path.dirname(__file__), "..")
project_src = os.path.abspath(project_src)
sys.path.insert(0, project_src)

print("PYTHONPATH contains:", sys.path[0])

from sumo_rl import SumoEnvironment
from sumo_rl.agents import PolicyGradientAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Policy Gradient Single-Intersection""",
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="src/sumo_rl/nets/single-intersection/single-intersection.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate (if used).\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument(
        "-s", dest="seconds", type=int, default=10000, required=False, help="Number of simulation seconds per episode.\n"
    )
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of episodes.\n")
    args = prs.parse_args()

    # ---- output path ----
    output_dir = Path("outputs/single-intersection")
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Folder for this experiment
    base_dir = Path("outputs/single-intersection")
    exp_dir = base_dir / f"PG_{experiment_time}_alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{args.decay}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Base prefix for CSVs inside this folder
    out_prefix = exp_dir / "ep"

    # ---- environment ----
    env = SumoEnvironment(
    net_file="src/sumo_rl/nets/single-intersection/single-intersection.net.xml",
    route_file=args.route,
    out_csv_name=str(out_prefix),   # <- use prefix inside the folder
    use_gui=args.gui,
    num_seconds=args.seconds,
    min_green=args.min_green,
    max_green=args.max_green,
    )

    # ---- create agents ONCE (so they don't reset every episode) ----
    initial_states = env.reset()
    pg_agents = {}
    for ts in env.ts_ids:
        obs = env.encode(initial_states[ts], ts)
        obs_dim = np.array(obs, dtype=np.float32).flatten().shape[0]
        pg_agents[ts] = PolicyGradientAgent(
            obs_dim=obs_dim,
            action_space=env.action_space,  # or env.action_spaces[ts] if needed
            lr=1e-3,
            beta_rew=0.01,
        )

    # ---- run multiple episodes ----
    for run in range(1, args.runs + 1):
        # reset environment at the START of each episode
        current_states = env.reset()
        done = {"__all__": False}

        if args.fixed:
            while not done["__all__"]:
                _, _, done, _ = env.step({})
        else:
            while not done["__all__"]:
                # ACT
                actions = {}
                for ts, agent in pg_agents.items():
                    obs = env.encode(current_states[ts], ts)
                    actions[ts] = agent.act(obs)

                # STEP
                next_states, rewards, done, _ = env.step(action=actions)

                # LEARN
                for ts, agent in pg_agents.items():
                    next_obs = env.encode(next_states[ts], ts)
                    done_flag = done.get(ts, done["__all__"])
                    agent.learn(next_state=next_obs, reward=rewards[ts], done=done_flag)

                current_states = next_states

        env.save_csv(str(out_prefix), run)

    env.close()
    pattern = str(exp_dir / "ep*_conn*_ep*.csv")
    episode_files = sorted(glob.glob(pattern))

    all_dfs = []
    max_step_per_episode = None
    step_size = None

    for episode_idx, f in enumerate(episode_files):
        df = pd.read_csv(f)

        if "step" not in df.columns:
            raise ValueError(f"CSV {f} has no 'step' column!")

        # Infer step_size and max_step_per_episode from the first file
        if max_step_per_episode is None:
            step_values = sorted(df["step"].unique())
            if len(step_values) >= 2:
                step_size = step_values[1] - step_values[0]
            else:
                step_size = 1  # fallback
            max_step_per_episode = df["step"].max()

        # Make step continuous across episodes:
        # episode 0: stays as-is
        # episode 1: shifted by max_step_per_episode + step_size
        # episode 2: shifted by 2 * (max_step_per_episode + step_size), etc.
        df["step"] = df["step"] + episode_idx * (max_step_per_episode + step_size)

        # Keep track of which episode this row came from (optional but useful)
        df["episode"] = episode_idx

        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)

        combined_path = exp_dir / "combined_episodes_continuous.csv"
        combined.to_csv(combined_path, index=False)

        print(f"Continuous combined CSV saved to: {combined_path}")
    else:
        print("No episode CSVs found to combine.")

