import argparse
import os
import json

import numpy as np

from utils import get_vectorized_cassie_envs
from config import BenchmarkConfig


def run_vec_env_diagnostics(envs, model, n_datapoints, on_episode_end_callback):
    log_keys = ["v_x", "v_y", "omega_z", "done_fail", "torque"]
    data = {key: [] for key in log_keys}
    n = 0
    obs = envs.reset()
    hidden_states, episode_starts = None, [True] * envs.num_envs
    while True:
        actions, hidden_states = model.predict(obs, state=hidden_states, episode_start=episode_starts, deterministic=True)
        obs, _, dones, infos = envs.step(actions)
        episode_starts = dones
        for i, done in enumerate(dones):
            if done:
                reset_rnn_hidden_states(hidden_states, i)
                data = log_data(infos[i], data, log_keys)
                on_episode_end_callback(i)
                n += 1
                if n == n_datapoints:
                    return data


def reset_rnn_hidden_states(hidden_states, i):
    for rnn in hidden_states:
        for state in rnn:
            state[:, i] = 0


def log_data(info, data, keys):
    info = info["benchmark"]
    for key in keys:
        data[key].append(info[key])
    return data


class Benchmarker():
    def __init__(self, command_following_benchmarks=True, verbose=0):
        self.verbose = verbose
        self.benchmarks = {
            "general_terrain_benchmark": self.general_terrain_benchmark,
            "step_height_benchmark": self.step_height_benchmark,
            "trench_width_benchmark": self.trench_width_benchmark,
            "stairs_descent_benchmark": self.stair_descent_benchmark
        }
        self.command_following_benchmarks = {
            "command_following_vx_benchmark": self.command_following_vx_benchmark,
            "command_following_vy_benchmark": self.command_following_vy_benchmark,
            "command_following_omegaz_benchmark": self.command_following_omegaz_benchmark,
        }
        if command_following_benchmarks:
            self.benchmarks.update(self.command_following_benchmarks)

    def run_benchmarks(self, envs, model, n_datapoints):
        results = {"n_datapoints": n_datapoints}
        for i, (benchmark, benchmark_fn) in enumerate(self.benchmarks.items()):
            results[benchmark] = benchmark_fn(envs=envs, model=model, n_datapoints=n_datapoints)
            if self.verbose:
                print(f"{(i+1)/len(self.benchmarks)*100:.2f}%")
        return results

    def save_results(self, results, save_path):
        with open(f"{save_path}.json", "w") as f:
            f.write(json.dumps(results))

    def get_mean_and_std(self, data):
        dones = data["done_fail"]
        del data["done_fail"]
        for key, value in data.items():
            data[key] = [v.mean() for v in value]

        return {
            **{"mean_mean_"+key: np.array(value).mean() for key, value in data.items()},
            **{"std_mean_"+key: np.array(value).std() for key, value in data.items()},
            **{"done_fail": np.array(dones).mean()}
        }

    def general_terrain_benchmark(self, envs, model, n_datapoints):
        def terrain_callback(env_index):
            envs.env_method("set_terrain", terrain_mode, indices=[env_index])
            envs.env_method("set_student_exteroception_noise", "nominal", indices=[env_index])

        data = {}
        for terrain_mode in ["stairs", "hills", "quantized_hills", "edges", "squares", "flat"]:
            if self.verbose:
                print(f"Running {terrain_mode} benchmark...")
            for env_index in range(envs.num_envs):
                terrain_callback(env_index)
            raw_data = run_vec_env_diagnostics(envs, model, n_datapoints, on_episode_end_callback=terrain_callback)
            data[terrain_mode] = self.get_mean_and_std(raw_data)
        return data

    def step_height_benchmark(self, envs, model, n_datapoints):
        def step_height_callback(env_index):
            envs.env_method("set_terrain", "step_height_benchmark", step_height, indices=[env_index])
            envs.env_method("set_student_exteroception_noise", "nominal", indices=[env_index])

        data = {}
        for step_height in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            if self.verbose:
                print(f"Running step height benchmark for {step_height}m...")
            for env_index in range(envs.num_envs):
                step_height_callback(env_index)
            raw_data = run_vec_env_diagnostics(envs, model, n_datapoints, on_episode_end_callback=step_height_callback)
            data[step_height] = self.get_mean_and_std(raw_data)
        return data

    def trench_width_benchmark(self, envs, model, n_datapoints):
        def trench_width_callback(env_index):
            envs.env_method("set_terrain", "trench_width_benchmark", trench_width, indices=[env_index])
            envs.env_method("set_student_exteroception_noise", "nominal", indices=[env_index])

        data = {}
        for trench_width in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            if self.verbose:
                print(f"Running trench width benchmark for {trench_width}m...")
            for env_index in range(envs.num_envs):
                trench_width_callback(env_index)
            raw_data = run_vec_env_diagnostics(envs, model, n_datapoints, on_episode_end_callback=trench_width_callback)
            data[trench_width] = self.get_mean_and_std(raw_data)
        return data

    def stair_descent_benchmark(self, envs, model, n_datapoints):
        def stair_descent_callback(env_index):
            envs.env_method("set_terrain", "stair_descent_benchmark", stair_height, indices=[env_index])
            envs.env_method("set_student_exteroception_noise", "nominal", indices=[env_index])

        data = {}
        for stair_height in [0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]:
            if self.verbose:
                print(f"Running stair descent benchmark for {stair_height}m...")
            for env_index in range(envs.num_envs):
                stair_descent_callback(env_index)
            raw_data = run_vec_env_diagnostics(envs, model, n_datapoints, on_episode_end_callback=stair_descent_callback)
            data[stair_height] = self.get_mean_and_std(raw_data)
        return data

    def command_following_vx_benchmark(self, envs, model, n_datapoints):
        vel_command_trajectory = np.zeros((301, 3))
        vel_command_trajectory[50:180, 0] = 1
        vel_command_trajectory[180:250, 0] = -1
        def command_following_vx_callback(env_index):
            envs.env_method("set_terrain", "squares", 1, indices=[env_index])
            envs.env_method("set_student_exteroception_noise", "nominal", indices=[env_index])
            envs.env_method("set_benchmark_velocity_commands", vel_command_trajectory, indices=[env_index])

        if self.verbose:
            print(f"Running command following v_x benchmark...")
        for env_index in range(envs.num_envs):
            command_following_vx_callback(env_index)
        raw_data = run_vec_env_diagnostics(envs, model, n_datapoints, on_episode_end_callback=command_following_vx_callback)
        v_x = []
        for trajec in raw_data["v_x"]:
            if trajec.shape[0] == 301:
                v_x.append(trajec)
        print(len(v_x))
        if len(v_x) == 0:
            return {"mean": 0, "min": 0, "max": 0, "std": 0}

        v_x = np.stack(v_x, axis=1)
        return {"mean": v_x.mean(axis=1).tolist(), "min": v_x.min(axis=1).tolist(), "max": v_x.max(axis=1).tolist(), "std": v_x.std(axis=1).tolist()}

    def command_following_vy_benchmark(self, envs, model, n_datapoints):
        vel_command_trajectory = np.zeros((301, 3))
        vel_command_trajectory[50:180, 1] = 1
        vel_command_trajectory[180:250, 1] = -1
        def command_following_vy_callback(env_index):
            envs.env_method("set_terrain", "squares", 1, indices=[env_index])
            envs.env_method("set_student_exteroception_noise", "nominal", indices=[env_index])
            envs.env_method("set_benchmark_velocity_commands", vel_command_trajectory, indices=[env_index])

        if self.verbose:
            print(f"Running command following v_y benchmark...")
        for env_index in range(envs.num_envs):
            command_following_vy_callback(env_index)
        raw_data = run_vec_env_diagnostics(envs, model, n_datapoints, on_episode_end_callback=command_following_vy_callback)
        v_y = []
        for trajec in raw_data["v_y"]:
            if trajec.shape[0] == 301:
                v_y.append(trajec)
        print(len(v_y))
        if len(v_y) == 0:
            return {"mean": 0, "min": 0, "max": 0, "std": 0}

        v_y = np.stack(v_y, axis=1)
        return {"mean": v_y.mean(axis=1).tolist(), "min": v_y.min(axis=1).tolist(), "max": v_y.max(axis=1).tolist(), "std": v_y.std(axis=1).tolist()}

    def command_following_omegaz_benchmark(self, envs, model, n_datapoints):
        vel_command_trajectory = np.zeros((301, 3))
        vel_command_trajectory[50:150, 2] = 1
        vel_command_trajectory[150:250, 2] = -1
        def command_following_omegaz_callback(env_index):
            envs.env_method("set_terrain", "squares", 1, indices=[env_index])
            envs.env_method("set_student_exteroception_noise", "nominal", indices=[env_index])
            envs.env_method("set_benchmark_velocity_commands", vel_command_trajectory, indices=[env_index])

        if self.verbose:
            print(f"Running command following omega_z benchmark...")
        for env_index in range(envs.num_envs):
            command_following_omegaz_callback(env_index)
        raw_data = run_vec_env_diagnostics(envs, model, n_datapoints, on_episode_end_callback=command_following_omegaz_callback)
        omega_z = []
        for trajec in raw_data["omega_z"]:
            if trajec.shape[0] == 301:
                omega_z.append(trajec)
        print(len(omega_z))
        if len(omega_z) == 0:
            return {"mean": 0, "min": 0, "max": 0, "std": 0}

        omega_z = np.stack(omega_z, axis=1)
        return {"mean": omega_z.mean(axis=1).tolist(), "min": omega_z.min(axis=1).tolist(), "max": omega_z.max(axis=1).tolist(), "std": omega_z.std(axis=1).tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, choices=['student', 'teacher'])
    parser.add_argument('observationFn', type=str, choices=["privileged", "exteroception", "proprioception"])
    parser.add_argument('model_name', type=str)
    parser.add_argument('epoch', type=int)
    args = parser.parse_args()


    config = BenchmarkConfig()
    config.env.obs_fn = args.observationFn
    config.student.student_mode = False if args.model_type == "teacher" else True

    envs = get_vectorized_cassie_envs(config=config, n_envs=os.cpu_count(), verbose=0)

    if args.model_type == "teacher":
        from sb3_contrib import RecurrentPPO
        load_path = os.path.join(f"{args.model_type}_log", args.model_name, f"teacher_model_{args.epoch}k.zip")
        model = RecurrentPPO.load(load_path)
    elif args.model_type == "student":
        from train_student import Model # here to avoid teacher training depending on student
        load_path = os.path.join(f"{args.model_type}_log", args.model_name, f"model_{args.epoch}")
        model = Model.load(envs, load_path)

    benchmarker = Benchmarker(verbose=1)
    benchmark_results = benchmarker.run_benchmarks(envs, model, n_datapoints=100)

    print(f"Saving results as {args.model_name}_{args.epoch}")
    save_path = os.path.join("benchmarks", f"{args.model_name}_{args.epoch}")
    benchmarker.save_results(benchmark_results, save_path)
