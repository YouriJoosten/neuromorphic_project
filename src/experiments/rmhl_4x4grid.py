import os
import sys
import random
import numpy as np
import pandas as pd
from sumo_rl import SumoEnvironment
from sumo_rl.agents import RMHLAgent
from sumo_rl.exploration import EpsilonGreedy

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

if __name__ == "__main__":
    alpha = 0.01
    gamma = 0.99
    decay = 1
    runs = 1
    episodes = 10
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    out_dir = "outputs/4x4/"
    os.makedirs(out_dir, exist_ok=True)

    env = SumoEnvironment(
        net_file="C:/Users/jjfva/Documents/Radboud/Neuromorphic Computing/Project/sumo-rl/sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml",
        route_file="C:/Users/jjfva/Documents/Radboud/Neuromorphic Computing/Project/sumo-rl/sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml",
        use_gui=False,
        num_seconds=100,
        min_green=5,
        delta_time=5,
        sumo_seed=seed
    )

    for run in range(1, runs + 1):
        run_seed = seed + run
        random.seed(run_seed)
        np.random.seed(run_seed)
        env.sumo_seed = run_seed

        initial_states = env.reset()
        rl_agents = {
            ts: RMHLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                lr=alpha,
                gamma=gamma,
                exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay)
            )
            for ts in env.ts_ids
        }

        for episode in range(1, episodes + 1):
            initial_states = env.reset()
            for ts in rl_agents.keys():
                agent = rl_agents[ts]
                agent.state = env.encode(initial_states[ts], ts)  # reset state
                agent.acc_reward = 0.0                            # reset accumulated reward
                agent.eligibility[:] = 0                          # reset eligibility traces
                agent.reward_baseline = 0.0                       # reset baseline for RMHL
                agent.exploration.epsilon = 0.05                  # reset exploration if needed

            # -----------------------------
            # Initialize tracking
            # -----------------------------
            prev_departed = prev_arrived = prev_teleported = 0
            prev_phase = None
            phase_start_step = 0
            phase_durations = {}
            step_metrics = []

            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: rl_agents[ts].act() for ts in rl_agents.keys()}
                s, r, done, _ = env.step(action=actions)

                # -----------------------------
                # Traffic light phase tracking (average NS vs EW across intersections)
                # -----------------------------
                ns_green_count = 0
                ew_green_count = 0

                for ts_id in env.ts_ids:
                    state_string = env.sumo.trafficlight.getRedYellowGreenState(ts_id)
                    ns_lanes = state_string[0:4]
                    ew_lanes = state_string[4:8]

                    if 'G' in ns_lanes:
                        ns_green_count += 1
                    if 'G' in ew_lanes:
                        ew_green_count += 1

                # Compute fraction of intersections green in each direction
                ns_fraction = ns_green_count / len(env.ts_ids)
                ew_fraction = ew_green_count / len(env.ts_ids)

                # Track durations per direction
                if prev_phase is None:
                    prev_phase = {"NS": ns_fraction, "EW": ew_fraction}
                    phase_start_step = env.sim_step
                else:
                    # For simplicity, consider a change if fraction changes (you can add a threshold)
                    if ns_fraction != prev_phase["NS"]:
                        duration = env.sim_step - phase_start_step
                        phase_durations.setdefault("NS_GREEN", []).append(duration)
                        prev_phase["NS"] = ns_fraction
                        phase_start_step = env.sim_step
                    if ew_fraction != prev_phase["EW"]:
                        duration = env.sim_step - phase_start_step
                        phase_durations.setdefault("EW_GREEN", []).append(duration)
                        prev_phase["EW"] = ew_fraction
                        phase_start_step = env.sim_step

                # -----------------------------
                # Step-wise metrics
                # -----------------------------
                row = {"step": env.sim_step}
                for ts_id in rl_agents.keys():
                    rl_agents[ts_id].learn(next_state=env.encode(s[ts_id], ts_id), reward=r[ts_id])
                    row[f"{ts_id}_reward"] = r[ts_id]
                    row[f"{ts_id}_acc_reward"] = rl_agents[ts_id].acc_reward

                row['episode_total_reward'] = sum(r.values())

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

            # -----------------------------
            # Saving metrics
            # -----------------------------
            df_metrics = pd.DataFrame(step_metrics)
            df_metrics['episode_total_acc_reward'] = df_metrics[[f"{ts}_acc_reward" for ts in rl_agents.keys()]].sum(axis=1)
            df_metrics.to_csv(f"{out_dir}rmhl-4x4grid_run{run}_ep{episode}.csv", index=False)

        
            summary_rows = []
            for phase, durations in phase_durations.items():
                summary_rows.append({
                    "episode": episode,
                    "phase": phase,
                    "avg_duration": sum(durations) / len(durations),
                    "num_switches": len(durations)
                })
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(f"{out_dir}rmhl-4x4grid_run{run}_trafficlights_ep{episode}.csv", index=False)

            # -----------------------------
            # Print accumulated rewards
            # -----------------------------
            for ts, agent in rl_agents.items():
                print(f"Run {run}, Episode {episode}, Agent {ts}, Accumulated Reward: {agent.acc_reward}")

            env.save_csv(f"{out_dir}rmhl-4x4grid_run{run}", episode)

    env.close()






