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
from sumo_rl.agents import PolicyGradientAgent


def main():
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Policy Gradient Two-Way Single Intersection",
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="src/sumo_rl/nets/2way-single-intersection/single-intersection-gen.rou.xml",
        help="Route definition xml file.",
    )
    prs.add_argument("-a", dest="alpha", type=float, default=1e-3, help="Alpha learning rate for policy gradient.")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, help="Gamma discount rate.")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, help="Minimum green time.")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, help="Maximum green time.")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.")
    prs.add_argument("-ns", dest="ns", type=int, default=42, help="Fixed green time for NS.")
    prs.add_argument("-we", dest="we", type=int, default=42, help="Fixed green time for WE.")
    prs.add_argument("-s", dest="seconds", type=int, default=1000, help="Number of simulation seconds per episode.")
    prs.add_argument("-v", action="store_true", default=False, help="Verbose: print per-step reward info.")
    prs.add_argument("-gc", dest="grad_clip", type=float, default=10.0, help="Gradient clipping norm.")
    prs.add_argument("--method", choices=["reinforce", "actor_critic"], default="reinforce", help="PG update type.")
    prs.add_argument("--beta-rew", dest="beta_rew", type=float, default=0.1, help="Baseline smoothing for REINFORCE.")
    prs.add_argument("-runs", dest="runs", type=int, default=3, help="Number of episodes.")
    args = prs.parse_args()

    experiment_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = Path("outputs/single-intersection")
    exp_dir = base_dir / f"PG_{experiment_time}_alpha{args.alpha}_gamma{args.gamma}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = exp_dir / "ep"

    training_log_path = exp_dir / "training_log.csv"
    training_log_cols = [
        "run",
        "agent",
        "episode_return",
        "discounted_return",
        "mean_return",
        "R_bar",
        "theta_norm",
        "theta_change",
        "theta_update_norm",
    ]
    pd.DataFrame(columns=training_log_cols).to_csv(training_log_path, index=False)

    env = SumoEnvironment(
        net_file="src/sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file=args.route,
        out_csv_name=str(out_prefix),
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
    )

    initial_states = env.reset()
    pg_agents = {}
    for ts in env.ts_ids:
        obs = env.encode(initial_states[ts], ts)
        obs_dim = np.array(obs, dtype=np.float32).flatten().shape[0]
        pg_agents[ts] = PolicyGradientAgent(
            obs_dim=obs_dim,
            action_space=env.action_space,
            lr=args.alpha,
            gamma=args.gamma,
            beta_rew=args.beta_rew,
            grad_clip=args.grad_clip,
            method=args.method,
        )

    for run in range(1, args.runs + 1):
        current_states = env.reset()
        done = {"__all__": False}
        for agent in pg_agents.values():
            agent.start_episode()

        episode_rewards = {ts: [] for ts in pg_agents}
        episode_stats = {ts: None for ts in pg_agents}

        if args.fixed:
            while not done["__all__"]:
                _, _, done, _ = env.step({})
        else:
            while not done["__all__"]:
                actions = {}
                for ts, agent in pg_agents.items():
                    obs = env.encode(current_states[ts], ts)
                    actions[ts] = agent.act(obs)

                next_states, rewards, done, _ = env.step(action=actions)

                if args.v:
                    all_r = list(rewards.values())
                    print(
                        "reward:", rewards,
                        " mean_wait:", {k: next_states[k].get("mean_waiting_time", "?") for k in next_states},
                        " min_step_r:", min(all_r),
                        " max_step_r:", max(all_r),
                    )

                for ts, agent in pg_agents.items():
                    done_flag = done.get(ts, done["__all__"])
                    episode_rewards[ts].append(rewards.get(ts, 0.0))
                    next_obs = env.encode(next_states[ts], ts) if not done_flag else None

                    stats = agent.learn(
                        reward=rewards.get(ts, 0.0),
                        done=done_flag,
                        next_state=next_obs,
                    )
                    if stats is not None:
                        episode_stats[ts] = stats

                current_states = next_states

        for ts, agent in pg_agents.items():
            rewards_list = episode_rewards[ts]
            episode_return = float(np.sum(rewards_list)) if rewards_list else 0.0
            discounted_return = float(np.sum([(args.gamma**i) * r for i, r in enumerate(rewards_list)])) if rewards_list else 0.0
            mean_return = float(np.mean(rewards_list)) if rewards_list else 0.0

            if args.method == "reinforce":
                stats = episode_stats[ts] if episode_stats[ts] is not None else agent.finish_episode()
            else:
                param_stats = agent.get_param_stats()
                stats = {
                    "episode_return": episode_return,
                    "discounted_return": discounted_return,
                    "mean_return": mean_return,
                    "R_bar": float(agent.R_bar),
                    "theta_norm": param_stats["theta_norm"],
                    "theta_change": param_stats["theta_change"],
                    "theta_update_norm": param_stats["theta_update_norm"],
                }

            stats_row = {
                "run": run,
                "agent": ts,
                "episode_return": stats["episode_return"],
                "discounted_return": stats["discounted_return"],
                "mean_return": stats["mean_return"],
                "R_bar": stats["R_bar"],
                "theta_norm": stats["theta_norm"],
                "theta_change": stats["theta_change"],
                "theta_update_norm": stats["theta_update_norm"],
            }
            pd.DataFrame([stats_row]).to_csv(training_log_path, mode="a", header=False, index=False)
            print(
                f"Run {run} | agent={ts} | ep_return={stats_row['episode_return']:.2f} "
                f"| G0={stats_row['discounted_return']:.2f} | meanG={stats_row['mean_return']:.2f} | R_bar={stats_row['R_bar']:.2f} "
                f"| ||theta||={stats_row['theta_norm']:.4f} | dtheta={stats_row['theta_change']:.6f}"
            )

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

        if max_step_per_episode is None:
            step_values = sorted(df["step"].unique())
            if len(step_values) >= 2:
                step_size = step_values[1] - step_values[0]
            else:
                step_size = 1
            max_step_per_episode = df["step"].max()

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
