import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import glob
import pandas as pd
import numpy as np

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

project_src = os.path.join(os.path.dirname(__file__), "..")
project_src = os.path.abspath(project_src)
sys.path.insert(0, project_src)

from sumo_rl import SumoEnvironment
from sumo_rl.agents import RMHLAgent_Legenstein
from sumo_rl.exploration import EpsilonGreedy


def main():
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="RMHL (Legenstein EH) Single-Intersection (2-way) experiment",
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="src/sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        help="Route definition xml file.",
    )
    prs.add_argument("-a", dest="alpha", type=float, default=0.001, help="Learning rate (Hebbian / EH).")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, help="Gamma discount rate (kept for compatibility).")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, help="Epsilon for epsilon-greedy.")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.01, help="Minimum epsilon.")
    prs.add_argument("-d", dest="decay", type=float, default=0.999, help="Epsilon decay.")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, help="Minimum green time.")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=30, help="Maximum green time.")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.")
    prs.add_argument("-s", dest="seconds", type=int, default=1000, help="Number of simulation seconds per episode.")
    prs.add_argument(
        "-r",
        dest="reward",
        type=str,
        default="wait",
        help="Reward function: [-r queue] for average queue reward or [-r wait] for waiting time reward.",
    )
    prs.add_argument("-runs", dest="runs", type=int, default=50, help="Number of episodes.")
    prs.add_argument("-v", action="store_true", default=False, help="Verbose: print per-step reward info.")
    args = prs.parse_args()

    # ---------- output paths ----------
    experiment_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = Path("outputs/2way-single-intersection")
    exp_dir = base_dir / (
        f"RMHL_Legenstein_{experiment_time}_alpha{args.alpha}_gamma{args.gamma}"
        f"_eps{args.epsilon}_decay{args.decay}_reward{args.reward}"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    out_prefix = exp_dir / "ep"  # env will use this as prefix for its CSVs

    # training log (same columns as PG script; map them to EH stats)
    training_log_path = exp_dir / "training_log.csv"
    training_log_cols = [
        "run",
        "agent",
        "episode_return",
        "discounted_return",
        "mean_return",
        "R_bar",              # here: RMHLAgent_Legenstein.R_mean
        "theta_norm",         # here: ||W||
        "theta_change",       # optional (0 unless you track it)
        "theta_update_norm",  # optional
    ]
    pd.DataFrame(columns=training_log_cols).to_csv(training_log_path, index=False)

    # ---------- environment ----------
    # You can switch reward_fn based on args.reward if you like;
    # this keeps "total-waiting-time" as before.
    env = SumoEnvironment(
        net_file="src/sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file=args.route,
        out_csv_name=str(out_prefix),
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        sumo_warnings=False,
        reward_fn="total-waiting-time",
    )

    # ---------- create RMHL agents ONCE ----------
    initial_states = env.reset()
    rl_agents = {
        ts: RMHLAgent_Legenstein(
            starting_state=initial_states[ts],
            state_space=env.observation_space,
            action_space=env.action_space,
            lr=args.alpha,
            gamma=args.gamma,
            exploration_strategy=EpsilonGreedy(
                initial_epsilon=args.epsilon,
                min_epsilon=args.min_epsilon,
                decay=args.decay,
            ),
        )
        for ts in env.ts_ids
    }

    # to track θ change per episode (optional)
    prev_W = {ts: agent.W.copy() for ts, agent in rl_agents.items()}

    # ---------- run episodes ----------
    for run in range(1, args.runs + 1):
        print(f"Starting run {run}/{args.runs}...")
        current_states = env.reset()
        done = {"__all__": False}

        # tell each agent a new episode started
        for ts, agent in rl_agents.items():
            agent.start_episode(current_states[ts])

        episode_reward = 0.0
        episode_steps = 0

        # per-agent reward history (environment reward)
        episode_rewards = {ts: [] for ts in rl_agents.keys()}

        if args.fixed:
            # no learning, just fixed-timing control
            while not done["__all__"]:
                _, _, done, _ = env.step({})
        else:
            while not done["__all__"]:
                # ---- ACT ----
                actions = {ts: rl_agents[ts].act() for ts in rl_agents.keys()}

                # ---- STEP ----
                next_states, rewards, done, _ = env.step(action=actions)

                if args.v:
                    all_r = list(rewards.values())
                    print(
                        "reward:", rewards,
                        " mean_wait:", {k: next_states[k].get("mean_waiting_time", "?") for k in next_states},
                        " min_step_r:", min(all_r),
                        " max_step_r:", max(all_r),
                    )

                # ---- LEARN ----
                for ts, agent in rl_agents.items():
                    done_flag = done.get(ts, done["__all__"])
                    r_env = float(rewards.get(ts, 0.0))
                    episode_rewards[ts].append(r_env)

                    # NOTE: if env reward is actually a *cost* (waiting time),
                    # you may prefer to pass -r_env instead to make "higher is better".
                    agent.learn(
                        next_state=next_states[ts],
                        reward=-r_env,
                        done=done_flag,
                    )

                episode_reward += float(np.mean(list(rewards.values())))
                episode_steps += 1
                current_states = next_states

        # ---------- per-episode logging to training_log.csv ----------
        for ts, agent in rl_agents.items():
            rewards_list = episode_rewards[ts]
            if rewards_list:
                episode_return = float(np.sum(rewards_list))
                discounted_return = float(np.sum([(args.gamma ** i) * r for i, r in enumerate(rewards_list)]))
                mean_return = float(np.mean(rewards_list))
            else:
                episode_return = 0.0
                discounted_return = 0.0
                mean_return = 0.0

            # Map PG-style columns to EH agent quantities
            R_bar = float(getattr(agent, "R_mean", 0.0))  # running reward mean in EH rule
            theta_norm = float(np.linalg.norm(getattr(agent, "W", np.array(0.0))))

            # optional: track how much W changed since previous episode
            W_prev = prev_W[ts]
            theta_change = float(np.linalg.norm(agent.W - W_prev))
            theta_update_norm = theta_change  # simple proxy
            prev_W[ts] = agent.W.copy()

            log_row = {
                "run": run,
                "agent": ts,
                "episode_return": episode_return,
                "discounted_return": discounted_return,
                "mean_return": mean_return,
                "R_bar": R_bar,
                "theta_norm": theta_norm,
                "theta_change": theta_change,
                "theta_update_norm": theta_update_norm,
            }

            pd.DataFrame([log_row]).to_csv(training_log_path, mode="a", header=False, index=False)
            print(
                f"Episode {run} | agent={ts} | return={log_row['episode_return']:.3f} | "
                f"G0={log_row['discounted_return']:.3f} | meanR={log_row['mean_return']:.3f} | "
                f"R_bar={log_row['R_bar']:.3f} | ||W||={log_row['theta_norm']:.4f} | "
                f"dtheta={log_row['theta_change']:.6f}"
            )

        avg_reward = episode_reward / max(episode_steps, 1)
        print(f"Episode {run}: steps={episode_steps}, total_reward={episode_reward:.3f}, avg_reward={avg_reward:.4f}")

        # save SUMO per-episode CSVs
        env.save_csv(str(out_prefix), run)

    env.close()

    # ---------- combine per-episode CSVs into one continuous file ----------
    pattern = str(exp_dir / "ep*_conn*_ep*.csv")
    episode_files = sorted(glob.glob(pattern))

    all_dfs = []
    max_step_per_episode = None
    step_size = None

    for episode_idx, f in enumerate(episode_files):
        df = pd.read_csv(f)

        if "step" not in df.columns:
            raise ValueError(f"CSV {f} has no 'step' column!")

        # infer step_size and max_step_per_episode from the first file
        if max_step_per_episode is None:
            step_values = sorted(df["step"].unique())
            if len(step_values) >= 2:
                step_size = step_values[1] - step_values[0]
            else:
                step_size = 1  # fallback
            max_step_per_episode = df["step"].max()

        # shift steps so each episode is contiguous in time
        df["step"] = df["step"] + episode_idx * (max_step_per_episode + step_size)
        df["episode"] = episode_idx
        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = exp_dir / "combined_episodes_continuous.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Continuous combined CSV saved to: {combined_path}")
    else:
        print("No episode CSVs found to combine.")


if __name__ == "__main__":
    main()
