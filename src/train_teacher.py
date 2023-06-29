import argparse
import os
import math

import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from utils import get_vectorized_cassie_envs
from benchmark import Benchmarker
from config import PPOConfig, BenchmarkConfig


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1) # dummy features_dim, PyTorch requires calling nn.Module.__init__ before adding modules

        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "proprioception":
                extractors[key] = nn.Identity()
                total_concat_size += subspace.shape[0]
            elif key == "exteroception_left":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 256),
                                                nn.LeakyReLU(),
                                                nn.Linear(256, 160),
                                                nn.LeakyReLU(),
                                                nn.Linear(160, 96)
                )
                total_concat_size += 96
            elif key == "exteroception_right":
                extractors[key] = extractors["exteroception_left"] # use same encoder for both height obs
                total_concat_size += 96
            elif key == "privileged":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 32),
                                                nn.LeakyReLU(),
                                                nn.Linear(32, 32),
                                                nn.LeakyReLU(),
                                                nn.Linear(32, 24)
                )
                total_concat_size += 24

        print(f"Feature extractor output dim: {total_concat_size}")
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
        total_params = 0
        for m in self.extractors:
            for p in self.extractors[m].parameters():
                total_params += torch.numel(p)
        print("input encoders total params:", total_params)

    def forward(self, observations):
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=-1)


class CurriculumCallback(BaseCallback):
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.terrain_ramp_start = config.curriculum.terrain_start
        self.terrain_ramp_end = config.curriculum.terrain_end
        self.gait_switch = config.curriculum.gait_clock_reward_end

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        curriculum_factor = max(0, min(1, (self.num_timesteps-self.terrain_ramp_start)/(self.terrain_ramp_end-self.terrain_ramp_start)))
        self.training_env.set_attr("curriculum_factor", curriculum_factor)
        self.logger.record("train/curriculum_factor", curriculum_factor)

        if self.gait_switch + 51e3 > self.num_timesteps > self.gait_switch:
            self.training_env.env_method("switch_gait_reward")


class BenchmarkCallback(BaseCallback):
    def __init__(self, ppo_config: PPOConfig, logdir):
        super().__init__()
        self.benchmarking_start = ppo_config.curriculum.benchmarks_start
        self.benchmarking_interval = ppo_config.curriculum.benchmarks_interval
        self.logdir = logdir
        self.n_datapoints = ppo_config.curriculum.datapoints_per_benchmark
        self.benchmarker = Benchmarker(command_following_benchmarks=False)

        config = BenchmarkConfig()
        config.env.obs_fn = ppo_config.env.obs_fn
        self.benchmark_envs = get_vectorized_cassie_envs(config, n_envs=os.cpu_count(), verbose=0)

    def _on_step(self):
        if self.num_timesteps >= self.benchmarking_start and self.num_timesteps % self.benchmarking_interval < 2*os.cpu_count(): # this assumes num_envs == num_cpus
            self.benchmark()
        return True

    def benchmark(self):
        print("Benchmarking...")
        results = self.benchmarker.run_benchmarks(self.benchmark_envs, self.model, n_datapoints=self.n_datapoints)
        if self.check_new_best(results):
            self.model.save(os.path.join(self.logdir, f"best_model_{int(self.num_timesteps/1000)}k.zip" ))
            self.benchmarker.save_results(results, os.path.join(self.logdir, f"benchmark_results_{self.num_timesteps}.json"))

    def check_new_best(self, results):
        scores = {}
        for benchmark, modes in results.items():
            if benchmark == "general_terrain_benchmark":
                for mode, metrics in modes.items():
                    score = 1 - metrics["done_fail"]
                    self.logger.record(f"benchmark/{mode}", score)
                    scores[mode] = score
            elif "benchmark" in benchmark:
                total = 0
                for metrics in modes.values():
                    total += 1 - metrics["done_fail"]
                mean = total/len(modes)
                self.logger.record(f"benchmark/{benchmark}", mean)
                scores[mode] = mean

        if not hasattr(self, "best_scores"):
            self.best_scores = scores
            return True
        else:
            for key, score in scores.items():
                if score < self.best_scores[key]:
                    return False
            self.best_scores = scores
            return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('steps', type=int)
    parser.add_argument('observationFn', type=str, choices=["privileged", "exteroception", "proprioception"])
    parser.add_argument('normalization', type=str, choices=["noNorm", "norm"])
    parser.add_argument('terrain', type=str, choices=["terrain", "flat"])
    parser.add_argument('--load_model', type=str, help="path to .zip in log folder")
    args = parser.parse_args()

    normalization = True if args.normalization == "norm" else False

    print("number of cpus available:", os.cpu_count())

    config = PPOConfig()
    config.env.obs_fn = args.observationFn
    config.randomization.randomization = True
    config.terrain.generation = True if args.terrain == "terrain" else False
    cassie_envs = get_vectorized_cassie_envs(config, n_envs=config.PPO.num_envs, normalization=normalization)

    model = RecurrentPPO(
        RecurrentActorCriticPolicy,
        cassie_envs,
        learning_rate=config.PPO.learning_rate,
        n_steps=math.ceil(config.PPO.min_buffer_length/config.PPO.num_envs),
        batch_size=8,
        whole_sequences=True,
        n_epochs=5,
        tensorboard_log=config.PPO.log_dir,
        policy_kwargs={
            "features_extractor_class": CustomCombinedExtractor,
            "net_arch": [],
            "lstm_hidden_size": 256,
            "n_lstm_layers": 2,
        },
        verbose=2,
        )

    n_savepoints = min(max(1, int(args.steps/5e6)), 10)
    name = f"{args.name}-{int(args.steps/1000)}k-{args.observationFn[:7]}-{args.normalization}-{args.terrain}"

    benchmark_callback = BenchmarkCallback(config, logdir=os.path.join(config.PPO.log_dir, f"{name}_0"))

    if args.load_model != None:
        model.set_parameters(os.path.join(config.PPO.log_dir, args.load_model))
        model.env.set_attr("curriculum_factor", 1)
        callbacks = [benchmark_callback]
    else:
        curriculum_callback = CurriculumCallback(config)
        callbacks = [curriculum_callback, benchmark_callback]

    for i in range(n_savepoints):
        model.learn(round(args.steps/n_savepoints), callback=callbacks, tb_log_name=name, reset_num_timesteps=False)
        log_dir = model.logger.get_dir()
        model.save(os.path.join(log_dir, f"teacher_model_{int((i+1)/n_savepoints*args.steps/1000)}k.zip"))
        if normalization:
            cassie_envs.save(os.path.join(log_dir, f"VecNormalize_{int((i+1)/n_savepoints*args.steps/1000)}k.zip"))