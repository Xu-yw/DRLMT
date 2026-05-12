import os

os.environ.setdefault("VEC_RUN_ID", "baseline_vec2_s0")
os.environ.setdefault("MUTATION_TYPE", os.environ.get("VEC_RUN_ID", "baseline_vec2_s0"))
os.environ.setdefault("MUTATION_SEED", os.environ.get("TRAINING_SEED", "0"))

import argparse
import csv
import logging
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime

import numpy as np
import torch
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter

from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from parameters import *
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo")
    parser.add_argument("--env-name", type=str, default="carla")
    parser.add_argument("--learning-rate", type=float, default=PPO_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--action-std-init", type=float, default=ACTION_STD_INIT)
    parser.add_argument("--episode-length", type=int, default=EPISODE_LENGTH)
    parser.add_argument("--train", default=True, type=boolean_string)
    parser.add_argument("--town", type=str, default="Town07")
    parser.add_argument("--load-checkpoint", type=lambda x: bool(strtobool(x)), default=MODEL_LOAD, nargs="?", const=True)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--termination_of_reward", type=int, default=TERMINATION_OF_REWARD)
    parser.add_argument("--carla-ports", type=str, default=os.environ.get("CARLA_PORTS", "2002,2004"))
    parser.add_argument("--run-id", type=str, default=os.environ.get("VEC_RUN_ID", "baseline_vec2_s0"))
    parser.add_argument("--max-episodes", type=int, default=int(os.environ.get("MAX_EPISODES", "0")))
    parser.add_argument("--step-timeout", type=float, default=float(os.environ.get("STEP_TIMEOUT", "120")))
    return parser.parse_args()


def _ports_from_arg(value):
    ports = [int(p.strip()) for p in value.split(",") if p.strip()]
    if not ports:
        raise ValueError("--carla-ports must contain at least one port")
    return ports


def _write_episode_log_header(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "timestep_at_done", "total_reward", "total_steps",
            "distance_m", "done_reason", "final_waypoint_idx", "route_length",
            "mutation_type", "seed", "spawn_idx", "start_x", "start_y",
            "start_yaw", "episode_wall_time_s", "env_id", "carla_port",
        ])


def _append_episode_log(path, episode, timestep, reward, steps, distance, reason,
                        final_idx, route_len, mutation_type, seed, spawn, wall_time,
                        env_id, carla_port):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            episode, timestep, reward, steps, distance, reason, final_idx, route_len,
            mutation_type, seed, spawn.get("spawn_idx"), spawn.get("start_x"),
            spawn.get("start_y"), spawn.get("start_yaw"), round(wall_time, 4),
            env_id, carla_port,
        ])


def _save_meta_checkpoint(meta_checkpoint_dir, episode, timestep, cumulative_score, action_std_init):
    os.makedirs(meta_checkpoint_dir, exist_ok=True)
    chkt_file_nums = len(next(os.walk(meta_checkpoint_dir))[2])
    if chkt_file_nums != 0:
        chkt_file_nums -= 1
    chkpt_file = os.path.join(meta_checkpoint_dir, "checkpoint_ppo_" + str(chkt_file_nums) + ".pickle")
    data_obj = {
        "cumulative_score": cumulative_score,
        "episode": episode,
        "timestep": timestep,
        "action_std_init": action_std_init,
    }
    with open(chkpt_file, "wb") as handle:
        pickle.dump(data_obj, handle)


def runner():
    args = parse_args()
    if args.exp_name != "ppo":
        sys.exit("continuous_driver_vec.py currently supports --exp-name ppo only")
    if not args.train:
        sys.exit("continuous_driver_vec.py is a training-only entry point")

    ports = _ports_from_arg(args.carla_ports)
    os.environ["MUTATION_TYPE"] = args.run_id
    os.environ.setdefault("TRAINING_SEED", str(args.seed))

    run_name = "PPO"
    total_timesteps = int(args.total_timesteps)
    action_std_init = args.action_std_init
    termination_of_rewards = args.termination_of_reward
    meta_checkpoint_dir = os.environ.get(
        "PPO_META_CHECKPOINT_DIR",
        os.path.join("checkpoints", "PPO_" + args.run_id, args.town),
    )

    tb_dir = os.environ.get(
        "TENSORBOARD_RUN_DIR",
        os.path.join("runs", args.run_id, args.town),
    )
    writer = SummaryWriter(tb_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}" for key, value in vars(args).items()]),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    encode = EncodeState(LATENT_DIM)
    agent = PPOAgent(args.town, action_std_init)

    if args.load_checkpoint:
        chkt_file_nums = len(next(os.walk(meta_checkpoint_dir))[2]) - 1
        chkpt_file = os.path.join(meta_checkpoint_dir, "checkpoint_ppo_" + str(chkt_file_nums) + ".pickle")
        with open(chkpt_file, "rb") as f:
            data = pickle.load(f)
            episode = data["episode"]
            timestep = data["timestep"]
            cumulative_score = data["cumulative_score"]
            action_std_init = data["action_std_init"]
        agent.load()
    else:
        episode = 0
        timestep = 0
        cumulative_score = 0

    envs = []
    for port in ports:
        try:
            client, world = ClientConnection(args.town, port=port).setup()
            envs.append(CarlaEnvironment(client, world, args.town))
            logging.info("Connection on CARLA port %s has been setup successfully.", port)
        except Exception:
            logging.exception("Connection has been refused by CARLA port %s.", port)
            raise

    ep_log_path = os.environ.get("EPISODE_LOG_PATH")
    if ep_log_path:
        _write_episode_log_header(ep_log_path)

    reward_window = 50
    reward_threshold = float(os.environ.get("REWARD_THRESHOLD", "0"))
    reward_hold = 10
    reward_history = []
    hold_count = 0
    scores = []
    episodic_length = []
    deviation_from_center = 0.0
    distance_covered = 0.0

    action_std_decay_rate = 0.05
    min_action_std = 0.05
    action_std_decay_freq = 5e5

    observations = [None] * len(envs)
    current_ep_reward = [0.0] * len(envs)
    current_ep_steps = [0] * len(envs)
    current_ep_start = [datetime.now()] * len(envs)

    def reset_env(env_id):
        observation = envs[env_id].reset()
        if observation is None:
            observations[env_id] = None
            return False
        observation = encode.process(observation)
        if torch.any(torch.isnan(observation)):
            observation = torch.zeros_like(observation)
        observations[env_id] = observation
        current_ep_reward[env_id] = 0.0
        current_ep_steps[env_id] = 0
        current_ep_start[env_id] = datetime.now()
        return True

    t0 = datetime.now()
    with ThreadPoolExecutor(max_workers=len(envs)) as executor:
        while timestep < total_timesteps and (args.max_episodes <= 0 or episode < args.max_episodes):
            if reward_threshold > 0 and len(reward_history) >= reward_window:
                window_avg = sum(reward_history[-reward_window:]) / reward_window
                if window_avg >= reward_threshold:
                    hold_count += 1
                    if hold_count >= reward_hold:
                        print(f"[STOP] reward converged: window_avg={window_avg:.2f} >= threshold={reward_threshold} for {reward_hold} eps")
                        break
                else:
                    hold_count = 0

            active = []
            for env_id in range(len(envs)):
                if observations[env_id] is None:
                    reset_env(env_id)
                if observations[env_id] is not None:
                    active.append(env_id)
            if not active:
                time.sleep(1.0)
                continue

            action_by_env = {}
            for env_id in active:
                action_by_env[env_id] = agent.get_action(
                    observations[env_id],
                    train=True,
                    env_id=env_id,
                )

            future_by_env = {
                env_id: executor.submit(envs[env_id].step, action_by_env[env_id])
                for env_id in active
            }

            learn_due = False
            save_due = False
            latest_cumulative_score = cumulative_score
            for env_id in active:
                try:
                    result = future_by_env[env_id].result(timeout=args.step_timeout)
                except TimeoutError:
                    print(
                        f"[STEP-TIMEOUT] env_id={env_id} port={ports[env_id]} "
                        f"exceeded {args.step_timeout}s; exiting for watchdog restart",
                        flush=True,
                    )
                    os._exit(3)
                except Exception as e:
                    print(f"[STEP-EXCEPTION] env_id={env_id} port={ports[env_id]} error={e}", flush=True)
                    result = None
                if result is None:
                    reward = -10
                    done = True
                    next_observation = None
                    info = [0, 0]
                    done_reason = "step_failed"
                else:
                    next_observation, reward, done, info = result
                    done_reason = None
                    if next_observation is None and not done:
                        reward = -10
                        done = True
                        done_reason = "step_failed"

                agent.record_transition(reward, done)
                timestep += 1
                current_ep_reward[env_id] += reward
                current_ep_steps[env_id] += 1

                if timestep % action_std_decay_freq == 0:
                    action_std_init = agent.decay_action_std(action_std_decay_rate, min_action_std)
                if done:
                    episode += 1
                    ep_wall = (datetime.now() - current_ep_start[env_id]).total_seconds()
                    episodic_length.append(ep_wall)
                    distance = info[0] if info else 0
                    deviation = info[1] if info else 0
                    distance_covered += distance
                    deviation_from_center += deviation
                    scores.append(current_ep_reward[env_id])
                    reward_history.append(current_ep_reward[env_id])
                    cumulative_score = np.mean(scores)
                    latest_cumulative_score = cumulative_score

                    env = envs[env_id]
                    if done_reason is None:
                        done_reason = env.get_last_done_reason() if hasattr(env, "get_last_done_reason") else "unknown"
                    final_idx = getattr(env, "current_waypoint_index", 0)
                    route_len = len(env.route_waypoints) if getattr(env, "route_waypoints", None) else 0
                    spawn = env.get_last_spawn_info() if hasattr(env, "get_last_spawn_info") else {}
                    if ep_log_path:
                        try:
                            _append_episode_log(
                                ep_log_path, episode, timestep, current_ep_reward[env_id],
                                current_ep_steps[env_id], distance, done_reason or "unknown",
                                final_idx, route_len, args.run_id, args.seed, spawn, ep_wall,
                                env_id, ports[env_id],
                            )
                        except Exception as e:
                            print(f"[EP-LOG] write failed: {e}")

                    print(
                        "Episode: {}".format(episode),
                        ", Env: {}".format(env_id),
                        ", Port: {}".format(ports[env_id]),
                        ", Timestep: {}".format(timestep),
                        ", Reward:  {:.2f}".format(current_ep_reward[env_id]),
                        ", Average Reward:  {:.2f}".format(cumulative_score),
                        ", Time: {}".format(datetime.now() - t0),
                        flush=True,
                    )

                    if episode % 10 == 0:
                        learn_due = True

                    if episode % 5 == 0:
                        writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                        writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                        writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                        writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-5:]), episode)
                        writer.add_scalar("Average Reward/(t)", np.mean(scores[-5:]), timestep)
                        writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                        writer.add_scalar("Reward/(t)", current_ep_reward[env_id], timestep)
                        writer.add_scalar("Average Deviation from Center/episode", deviation_from_center / 5, episode)
                        writer.add_scalar("Average Deviation from Center/(t)", deviation_from_center / 5, timestep)
                        writer.add_scalar("Average Distance Covered (m)/episode", distance_covered / 5, episode)
                        writer.add_scalar("Average Distance Covered (m)/(t)", distance_covered / 5, timestep)
                        episodic_length = []
                        deviation_from_center = 0
                        distance_covered = 0

                    if episode % 100 == 0 or current_ep_reward[env_id] >= termination_of_rewards:
                        save_due = True

                    observations[env_id] = None
                else:
                    next_observation = encode.process(next_observation)
                    if torch.any(torch.isnan(next_observation)):
                        next_observation = torch.zeros_like(next_observation)
                    observations[env_id] = next_observation

            # Learn only after every env in the current vectorized step has
            # recorded its reward, otherwise the PPO buffer can have one more
            # state/action than reward/done.
            if learn_due:
                agent.learn()
                agent.chkpt_save()
                _save_meta_checkpoint(meta_checkpoint_dir, episode, timestep, latest_cumulative_score, action_std_init)

            if save_due:
                agent.save()
                _save_meta_checkpoint(meta_checkpoint_dir, episode, timestep, latest_cumulative_score, action_std_init)

    for env in envs:
        try:
            for sensor in list(getattr(env, "sensor_list", [])):
                sensor.destroy()
            for actor in list(getattr(env, "actor_list", [])):
                actor.destroy()
            env.remove_sensors()
        except Exception as e:
            print(f"[CLEANUP] failed: {e}")

    print("Terminating the vectorized run.")
    flag_path = os.environ.get("TRAINING_DONE_FLAG", "/root/autodl-tmp/runs/training_done_vec.flag")
    try:
        os.makedirs(os.path.dirname(flag_path), exist_ok=True)
        with open(flag_path, "w") as f:
            f.write(f"timestep={timestep} episode={episode} ports={','.join(map(str, ports))}\n")
        print(f"[DONE-FLAG] wrote {flag_path}")
    except Exception as e:
        print(f"[DONE-FLAG] failed to write {flag_path}: {e}")
    sys.exit()


if __name__ == "__main__":
    try:
        runner()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("\nExit")
