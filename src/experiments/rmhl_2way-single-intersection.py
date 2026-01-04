import argparse
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import RMHLAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Q-Learning Single-Intersection"""
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-a", dest="alpha", type=float, default=0.01, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.01, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=0.995, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=True, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=1000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument(
        "-r",
        dest="reward",
        type=str,
        default="wait",
        required=False,
        help="Reward function: [-r queue] for average queue reward or [-r wait] for waiting time reward.\n",
    )
    
    prs.add_argument("-runs", dest="runs", type=int, default=2, help="Number of runs.\n") ###### EPISODE CHANGE HERE #####
    prs.add_argument("-seed",dest="seed",type=int,default=42,help="Random seed for reproducibility.\n",)

    args = prs.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)


    experiment_time = str(datetime.now()).replace(":", "-").split(".")[0]
    out_csv = f"outputs/2way-single-intersection/{experiment_time}_alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{args.decay}_reward{args.reward}"

    env = SumoEnvironment(
        net_file="C:/Users/jjfva/Documents/Radboud/Neuromorphic Computing/Project/sumo-rl/sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file="C:/Users/jjfva/Documents/Radboud/Neuromorphic Computing/Project/sumo-rl/sumo_rl/nets/2way-single-intersection/single-intersection-gen.rou.xml",
        out_csv_name=out_csv,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        sumo_warnings=False,
        reward_fn = "total-waiting-time",
        sumo_seed = args.seed,
    )
    
    initial_states = env.reset()
    rl_agents = {
            ts: RMHLAgent(
                starting_state=initial_states[ts],    
                state_space=env.observation_space,
                action_space=env.action_space,
                lr=args.alpha,
                gamma=args.gamma, 
                exploration_strategy=EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay),
            )
            for ts in env.ts_ids
        }
    
    # Metrics amount of vehicles
    prev_departed = 0
    prev_arrived = 0
    prev_teleported = 0

    for run in range(1, args.runs + 1):
        run_seed = args.seed + run

        random.seed(run_seed)
        np.random.seed(run_seed)

        env.sumo_seed = run_seed

        print(f"Starting run {run}/{args.runs}...")

        s = env.reset()
        done = {"__all__": False}

        # ==========================
        # Traffic Light Phase Tracking
        # ==========================
        prev_departed = 0
        prev_arrived = 0
        prev_teleported = 0

        prev_phase = None
        phase_start_step = 0
        phase_durations = {}

        # ==========================

        # Tracking performance
        episode_rewards = {ts: [] for ts in rl_agents.keys()}
        step_metrics = []

        if args.fixed:
            while not done["__all__"]:
                _, _, done, _ = env.step({})
        else:
            while not done["__all__"]:
                actions = {ts: rl_agents[ts].act() for ts in rl_agents.keys()}

                s, r, done, _ = env.step(action=actions)
                
                # ==========================
                # Traffic Light Phase Tracking
                # ==========================

                state_string = env.sumo.trafficlight.getRedYellowGreenState(env.ts_ids[0])
                ns_lanes = state_string[0:4]   # example: first 4 chars = north-south
                ew_lanes = state_string[4:8]   # example: next 4 chars = east-west

                if 'G' in ns_lanes and 'G' not in ew_lanes:
                    simple_phase = "NS_GREEN"
                elif 'G' in ew_lanes and 'G' not in ns_lanes:
                    simple_phase = "EW_GREEN"
                elif 'G' in ns_lanes and 'G' in ew_lanes:
                    simple_phase = "BOTH_GREEN"  # if possible in your TL program
                elif 'y' in state_string.lower():
                    simple_phase = "YELLOW"
                else:
                    simple_phase = "ALL_RED"

                if prev_phase is None:
                    prev_phase = simple_phase
                    phase_start_step = env.sim_step

                elif simple_phase != prev_phase:
                    # Phase switch detected
                    duration = env.sim_step - phase_start_step

                    if prev_phase not in phase_durations:
                        phase_durations[prev_phase] = []
                    phase_durations[prev_phase].append(duration)

                    # Reset for new phase
                    prev_phase = simple_phase
                    phase_start_step = env.sim_step

                # ==========================
                # ==========================

                row = {"step": env.sim_step}

                for agent_id in rl_agents.keys():
                    rl_agents[agent_id].learn(next_state=s[agent_id],reward=r[agent_id])
                    
                    # Store rewards
                    row[f"{agent_id}_reward"] = r[agent_id]
                    row[f"{agent_id}_acc_reward"] = rl_agents[agent_id].acc_reward

                # SUMO reward metrics
                row['episode_total_reward'] = sum(r.values())

                # SUMO throughput metrics
                departed = env.sumo.simulation.getDepartedNumber() - prev_departed
                arrived = env.sumo.simulation.getArrivedNumber() - prev_arrived
                teleported = env.sumo.simulation.getEndingTeleportNumber() - prev_teleported

                prev_departed = env.sumo.simulation.getDepartedNumber()
                prev_arrived = env.sumo.simulation.getArrivedNumber()
                prev_teleported = env.sumo.simulation.getEndingTeleportNumber()

                row['departed_this_step'] = departed
                row['arrived_this_step'] = arrived
                row['teleported_this_step'] = teleported

                
                step_metrics.append(row)

        # ==========================
        # Traffic Light Phase Tracking
        # ==========================

        summary_rows = []
        for phase, durations in phase_durations.items():
            summary_rows.append({
                "episode": run,
                "phase": phase,
                "avg_duration": sum(durations) / len(durations),
                "num_switches": len(durations)
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(f"{out_csv}_trafficlights_ep{run}.csv", index=False)

        # ==========================
        # ==========================

        df_metrics = pd.DataFrame(step_metrics)
        df_metrics['episode_total_acc_reward'] = df_metrics[[f"{ts}_acc_reward" for ts in rl_agents.keys()]].sum(axis=1)
        df_metrics.to_csv(f"{out_csv}_ep{run}.csv", index=False)

        env.save_csv(out_csv, run)
        env.close()



