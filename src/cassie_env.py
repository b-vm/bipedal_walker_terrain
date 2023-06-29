import time
import random

import gym
from gym.spaces import Dict, Box
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from cassie_mujoco_sim import CassieSim, CassieVis, state_out_t
from env_utils import generate_perlin, get_pd_input, BenchmarkLogger
from config import CassieBaseConfig


class CassieEnv(gym.Env):
    def __init__(self, config: CassieBaseConfig, verbose=1):
        super().__init__()
        self.cfg = config
        self.verbose = verbose

        # control vars
        self.velocity_commands = self.cfg.env.init_velocity_commands
        self.velocity_commands_change_timestep = random.randint(0, self.cfg.env.episode_step_limit)
        self.noise_change_timestep = random.randint(0, self.cfg.env.episode_step_limit)
        self.curriculum_factor = self.cfg.env.curriculum_factor
        self.step_counter = 0

        # objects
        self.sim = CassieSim(self.cfg.env.cassie_model_file)
        self.randomizer = Randomizer(self.sim, self.cfg)
        self.terrain_generator = TerrainGenerator(self.sim)
        self.exteroceptor = Exteroceptor(self.sim, self.cfg.student.student_mode)
        self.reward_calculator = RewardCalculator(self.cfg)
        if self.cfg.env.benchmarking_mode:
            self.benchmark_logger = BenchmarkLogger(self.cfg)

        # gym env compliance
        self.action_space = Box(low=-1e4, high=1e4, shape=(self.cfg.env.n_actuators,))

        obs_dummy = self._get_observation(state_out_t())
        observation_space = {}
        for key, sub_obs in obs_dummy.items():
            observation_space[key] = Box(low=-1e4, high=1e4, shape=(sub_obs.size,))
        self.observation_space = Dict(observation_space)

        if self.verbose:
            print(f"observation dim: {[array.size for _, array in obs_dummy.items()]} || action dim: {self.cfg.env.n_actuators}")

    def step(self, action):
        if self.cfg.env.action_scaling:
            action = action * self.cfg.env.action_scale
        if self.cfg.env.action_offset:
            action += self.cfg.env.neutral_action
        if self.cfg.randomization.randomization and self.step_counter == self.velocity_commands_change_timestep:
            self.velocity_commands = self.randomizer.randomize_velocity_commands()

        pd_input = get_pd_input(action, self.cfg)
        for _ in range(self.cfg.env.pd_sim_steps_per_policy_step):
            state_estimate = self.sim.step_pd(pd_input)

        observation = self._get_observation(state_estimate)
        done, failure = self._check_done_and_failure()

        if self.cfg.student.student_mode and self.cfg.randomization.randomization:
            reward = 0
            if self.step_counter == self.noise_change_timestep:
                self.exteroceptor.update_noises()
        else:
            reward = self.reward_calculator.calculate_reward(
                self.step_counter,
                state_estimate,
                self.sim.qpos(),
                self.sim.qvel(),
                self.velocity_commands,
                action,
                self._get_foot_vars(),
                self.curriculum_factor,
                failure,
            )

        info = {}
        if self.cfg.env.benchmarking_mode:
            info["benchmark"] = self.benchmark_logger.log(done, failure, state_estimate, self.sim.qvel())
            self.velocity_commands = self.benchmark_logger.get_velocity_commands(self.step_counter)

        self.step_counter += 1
        return observation, reward, done, info

    def _get_foot_vars(self):
        foot_vars = {}

        foot_pos = np.array(self.sim.foot_pos())
        foot_vars["l_foot_vel"] = np.linalg.norm(foot_pos[0:3] - self.prev_foot_pos[0:3]) * self.cfg.env.sim_rate_hz / self.cfg.env.pd_sim_steps_per_policy_step
        foot_vars["r_foot_vel"] = np.linalg.norm(foot_pos[3:6] - self.prev_foot_pos[3:6]) * self.cfg.env.sim_rate_hz / self.cfg.env.pd_sim_steps_per_policy_step
        self.prev_foot_pos = foot_pos

        foot_vars["l_foot_frc"], foot_vars["r_foot_frc"] = self.sim.get_foot_forces()

        foot_vars["l_foot_orient"] = np.array(self.sim.xquat("left-foot"))
        foot_vars["r_foot_orient"] = np.array(self.sim.xquat("right-foot"))

        return foot_vars

    def _get_observation(self, state_estimate):
        observation = {}
        proprioception = [
            np.array(state_estimate.motor.position) + self.randomizer.get_joint_pos_noise(state_estimate.motor.position),
            np.array(state_estimate.motor.velocity) + self.randomizer.get_joint_vel_noise(state_estimate.motor.velocity),
            np.array(state_estimate.joint.position) + self.randomizer.get_joint_pos_noise(state_estimate.joint.position),
            np.array(state_estimate.joint.velocity) + self.randomizer.get_joint_vel_noise(state_estimate.joint.velocity),
            np.array(state_estimate.pelvis.orientation) + self.randomizer.get_pelvis_ori_noise(state_estimate.pelvis.orientation),
            np.array(state_estimate.pelvis.rotationalVelocity) + self.randomizer.get_pelvis_rot_vel_noise(state_estimate.pelvis.rotationalVelocity),
            self.velocity_commands,
        ]

        if self.cfg.env.obs_clock_input:
            phi = self.step_counter % self.cfg.env.gait_period
            proprioception.append([
                np.sin(2 * np.pi * (phi/self.cfg.env.gait_period + self.cfg.env.phase_shifts[0])),
                np.sin(2 * np.pi * (phi/self.cfg.env.gait_period + self.cfg.env.phase_shifts[1]))
            ])
        observation["proprioception"] = np.concatenate(proprioception)

        if self.cfg.env.obs_fn == "exteroception" or self.cfg.env.obs_fn == "privileged":
            observation["exteroception_left"], observation["exteroception_right"] = self.exteroceptor.sample()

        if self.cfg.env.obs_fn == "privileged":
            observation["privileged"] = np.concatenate([
                *self.sim.get_heeltoe_forces(),
                self.sim.get_foot_vel(),
            ])

        if self.cfg.student.student_mode:
            observation["noisy_exteroception_left"], observation["noisy_exteroception_right"] = self.exteroceptor.noisy_sample()

        return observation

    def _check_done_and_failure(self):
        pelvis_height = self.sim.qpos()[2] - self.exteroceptor.get_z_offset()
        if pelvis_height < self.cfg.env.height_limit_min or pelvis_height > self.cfg.env.height_limit_max:
            return True, True
        elif self.step_counter == self.cfg.env.episode_step_limit:
            return True, False
        return False, False

    def reset(self):
        self.sim.full_reset()
        self.step_counter = 0
        self.prev_foot_pos = np.array(self.sim.foot_pos())
        self.reward_calculator.reset()

        if self.cfg.randomization.randomization:
            self.randomizer.randomize_sim_params()
            self.randomizer.randomize_velocity_commands()
            self.velocity_commands_change_timestep = random.randint(0, self.cfg.env.episode_step_limit)

        if self.cfg.terrain.generation:
            self.terrain_generator.generate_terrain(self.curriculum_factor, self.cfg.student.student_mode)
            self.exteroceptor.update_height_sampler()

        if self.cfg.student.student_mode and self.cfg.randomization.randomization:
            self.exteroceptor.update_noises()
            self.noise_change_timestep = random.randint(0, self.cfg.env.episode_step_limit)
            self.curriculum_factor = random.choice((random.uniform(0, 1), 1))

        if self.cfg.env.benchmarking_mode:
            self.benchmark_logger.reset()
            self.velocity_commands = self.cfg.env.init_velocity_commands

        pd_input_zero = get_pd_input(np.zeros(self.action_space.shape[0]), self.cfg)
        state_estimate = self.sim.step_pd(pd_input_zero)
        observation = self._get_observation(state_estimate)
        return observation

    def render(self):
        raise NotImplementedError

    def close (self):
        pass

    def switch_gait_reward(self):
        self.cfg.reward.switch_gait_reward(self.cfg.reward.scales)

    def set_velocity_commands(self, velocity_commands):
        assert isinstance(velocity_commands, tuple)
        if len(velocity_commands) == 2:
            self.velocity_commands = (np.cos(velocity_commands[0]), np.sin(velocity_commands[0]), velocity_commands[1])
        elif len(velocity_commands) == 3:
            self.velocity_commands = velocity_commands
        else:
            raise ValueError("Velocity commands must be a tuple of length 2 or 3.")

    def get_rewards(self):
        return self.reward_calculator.rewards

    def set_terrain(self, terrain_type="flat", amplitude=1):
        assert terrain_type in self.terrain_generator.terrains_all
        self.terrain_generator.terrains_all[terrain_type](amplitude)
        self.exteroceptor.update_height_sampler()

    def get_exteroception_config(self):
        return self.exteroceptor.get_config()

    def set_exteroception_memory(self, pointers):
        self.exteroceptor.set_memory(pointers)

    def get_pelvis_coords(self):
        return self.sim.qpos()[:2]

    def set_student_exteroception_noise(self, noise):
        self.exteroceptor.update_noises(noise)

    def set_benchmark_velocity_commands(self, velocity_command_trajectory):
        self.benchmark_logger.set_velocity_command_trajectory(velocity_command_trajectory)


class Randomizer:
    def __init__(self, sim_ptr, cfg: CassieBaseConfig):
        self.sim_ptr = sim_ptr
        self.cfg = cfg.randomization

        self.default_dampings = self.sim_ptr.get_dof_damping()
        self.default_masses = self.sim_ptr.get_body_mass()
        self.default_frictions = self.sim_ptr.get_geom_friction()

    def randomize_sim_params(self):
        dampings = np.random.uniform(self.cfg.domain.damping_range[0]*self.default_dampings, self.cfg.domain.damping_range[1]*self.default_dampings)
        self.sim_ptr.set_dof_damping(dampings)

        masses = np.random.uniform(self.cfg.domain.mass_range[0]*self.default_masses, self.cfg.domain.mass_range[1]*self.default_masses)
        self.sim_ptr.set_body_mass(masses)

        frictions = np.random.uniform(self.cfg.domain.friction_range[0]*self.default_frictions, self.cfg.domain.friction_range[1]*self.default_frictions)
        self.sim_ptr.set_geom_friction(frictions)

    def randomize_velocity_commands(self):
        if self.cfg.command.uniform:
            velocity_commands = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
            if np.linalg.norm(velocity_commands) < 0.2: # p = 0.00425 # TODO increase p?
                velocity_commands = (0, 0, 0)
            return velocity_commands
        else:
            p = random.random()
            if p < 0.15:
                return (0, 0, 0)
            elif p < 0.20:
                return (0, 0, random.choice([-1, 1]))
            elif p < 0.9:
                random_heading = random.choices([0, np.pi/2, np.pi, 1.5*np.pi], weights=[0.6, 0.1, 0.2, 0.1])[0]
                return (np.cos(random_heading), np.sin(random_heading), 0)
            else:
                random_heading = random.uniform(0, 2*np.pi)
                return (np.cos(random_heading), np.sin(random_heading), random.choice([-1, 0, 1]))

    def get_joint_pos_noise(self, signals):
        if self.cfg.proprioception.add_noise:
            return np.random.uniform(*self.cfg.proprioception.joint_pos_range, size=len(signals))
        return 0

    def get_joint_vel_noise(self, signals):
        if self.cfg.proprioception.add_noise:
            return np.random.uniform(*self.cfg.proprioception.joint_vel_range, size=len(signals))
        return 0

    def get_pelvis_ori_noise(self, signal):
        if self.cfg.proprioception.add_noise:
            return np.random.uniform(*self.cfg.proprioception.pelvis_orientation_range, size=len(signal))
        return 0

    def get_pelvis_rot_vel_noise(self, signal):
        if self.cfg.proprioception.add_noise:
            return np.random.uniform(*self.cfg.proprioception.pelvis_angular_velocity_range, size=len(signal))
        return 0


class TerrainGenerator:
    def __init__(self, sim_ptr):
        self.sim_ptr = sim_ptr
        self.vis_ptr = None

        self.m = sim_ptr.get_hfield_nrow() # width
        self.n = sim_ptr.get_hfield_ncol() # length

        self.resolution = self.sim_ptr.get_hfield_size()[0]*2/self.n
        assert self.resolution == self.sim_ptr.get_hfield_size()[1]*2/self.m, "resolution should be same in both directions"

        self.cassie_pos = (100, 100)
        self.terrains_train_teacher = {
            "stairs": self.stairs,
            "hills": self.hills,
            "quantized_hills": self.quantized_hills,
            "edges": self.edges,
            "squares": self.squares
        }
        self.terrains_train_student = {
            **self.terrains_train_teacher,
            "flat": self.flat
        }
        self.terrains_all = {
            **self.terrains_train_teacher,
            "flat": self.flat,
            "step_height_benchmark": self.step_height_benchmark,
            "stair_descent_benchmark": self.stair_descent_benchmark,
            "trench_width_benchmark": self.trench_width_benchmark
        }
        self.flat()

    def set_vis_ptr(self, vis_ptr):
        self.vis_ptr = vis_ptr

    def generate_terrain(self, curriculum_factor, student_mode):
        if student_mode:
            random.choice(list(self.terrains_train_student.values()))(curriculum_factor)
        else:
            random.choice(list(self.terrains_train_teacher.values()))(curriculum_factor)

    def _set_hfield(self, hfield):
        hfield_normalized = (hfield + 2) / 4
        self.sim_ptr.set_hfield_data(hfield_normalized.flatten(), self.vis_ptr)

    def flat(self, _=None):
        hfield = np.zeros((self.m, self.n))
        self._set_hfield(hfield)

    def stairs(self, curriculum_factor):
        def stair_wave(n, n_stairs): # to heaven
            val = n % (n_stairs*2)
            if val > n_stairs:
                val = n_stairs - val % n_stairs
            return int(val)

        step_height = random.uniform(0.1, 0.22) * random.choice((1, -1))
        step_depth = int(random.uniform(0.3, 0.4) // self.resolution)
        n_stairs = 10
        landing_size = 1

        hfield  = np.zeros((self.m, self.n))
        n_steps = self.n // step_depth

        i = 0
        n = round(self.cassie_pos[1] / step_depth)
        while n < n_steps:
            hfield[:, n*step_depth:(n+1)*step_depth] = stair_wave(i, n_stairs) * step_height
            n += 1
            if i % n_stairs == 0 and i != 0:
                hfield[:, n*step_depth:(n+landing_size)*step_depth] = stair_wave(i, n_stairs) * step_height
                n += landing_size
            i += 1

        i = 0
        n = round(self.cassie_pos[1] / step_depth)
        while n > 0:
            hfield[:, (n-1)*step_depth:n*step_depth] = stair_wave(i, n_stairs) * step_height
            n -= 1
            if i % n_stairs == 0 and i != 0:
                hfield[:, (n-landing_size)*step_depth:n*step_depth] = stair_wave(i, n_stairs) * step_height
                n -= landing_size
            i += 1

        hfield -= hfield[self.cassie_pos[0], self.cassie_pos[1]]
        hfield *= curriculum_factor
        self._set_hfield(hfield)

    def hills(self, curriculum_factor):
        hfield = generate_perlin(frequency=0.02, size=(self.m, self.n))
        hfield += 0.2 * generate_perlin(frequency=0.07, size=(self.m, self.n))
        hfield -= hfield.min()
        hfield *= 0.8 * curriculum_factor / hfield.max()

        hfield = self._clear_cassie_collisions(hfield, hfield[self.cassie_pos[0], self.cassie_pos[1]])
        self._set_hfield(hfield)

    def quantized_hills(self, curriculum_factor):
        step_size = random.uniform(0.12, 0.18)
        hfield = generate_perlin(frequency=0.02, size=(self.m, self.n))
        hfield = np.digitize(hfield, [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype('float64')

        hfield -= hfield.min()
        hfield *= step_size * curriculum_factor

        hfield = self._clear_cassie_collisions(hfield, hfield[self.cassie_pos[0], self.cassie_pos[1]])
        self._set_hfield(hfield)

    def edges(self, curriculum_factor):
        step_height = random.uniform(0.15, 0.25) * curriculum_factor
        hfield = generate_perlin(frequency=0.05, size=(self.m, self.n))
        hfield = np.where(hfield < 0, 0, step_height)

        hfield = self._clear_cassie_collisions(hfield, random.choice((0, step_height)))
        self._set_hfield(hfield)

    def squares(self, curriculum_factor):
        period = random.randint(8, 12) # 1 = 0.05m
        hfield = np.zeros((self.m, self.n))
        for i in range(int(self.m/period)):
            for j in range(int(self.n/period)):
                hfield[i*period:(i+1)*period, j*period:(j+1)*period] = random.uniform(0.1, 0.4) * curriculum_factor

        hfield = self._clear_cassie_collisions(hfield, hfield.mean())
        self._set_hfield(hfield)

    def step_height_benchmark(self, step_height):
        hfield = np.zeros((self.m, self.n))
        hfield[:, 120:140] = step_height
        self._set_hfield(hfield)

    def trench_width_benchmark(self, trench_width):
        hfield = np.zeros((self.m, self.n))
        trench_width = int(trench_width // self.resolution)
        hfield[:, 120:120+trench_width] = -0.5
        self._set_hfield(hfield)

    def stair_descent_benchmark(self, stair_height):
        assert 0 < stair_height <= 0.2
        stair_depth = int(0.35 // self.resolution)
        hfield = 2 * np.ones((self.m, self.n))
        for stair in range(10):
            hfield[:, 120+stair*stair_depth:120+(stair+1)*stair_depth] -= stair_height * stair
        hfield[:, 120+(stair+1)*stair_depth:self.n] -= stair_height * stair

        hfield = self._clear_cassie_collisions(hfield, hfield[self.cassie_pos[0], self.cassie_pos[1]])
        self._set_hfield(hfield)

    def _clear_cassie_collisions(self, hfield, cassie_start_height):
        hfield[self.cassie_pos[0]-5:self.cassie_pos[0]+5, self.cassie_pos[1]-4:self.cassie_pos[1]+4] = cassie_start_height
        hfield -= hfield[self.cassie_pos[0], self.cassie_pos[1]]
        return hfield


class Exteroceptor:
    def __init__(self, sim_ptr, student_mode=False):
        self.sim_ptr = sim_ptr
        self.render_memory = None
        self.student_mode = student_mode
        self.height_sampler = None

        self.n_row = self.sim_ptr.get_hfield_nrow()
        self.n_col = self.sim_ptr.get_hfield_ncol()
        self.sizes = self.sim_ptr.get_hfield_size()

        self.x_coords = np.linspace(0, self.sizes[0]*2, self.n_col)
        self.y_coords = np.linspace(0, self.sizes[1]*2, self.n_row)

        self.sample_coords = self.generate_sampling_pattern(
            radius_per_ring=[0.08, 0.12, 0.16, 0.21, 0.26, 0.31, 0.36, 0.42, 0.48, 0.54, 0.60, 0.67, 0.74, 0.82, 0.90, 0.98],
            n_points_per_ring=[8, 10, 12, 14, 16, 18, 20, 24, 20, 20, 20, 22, 24, 26, 30, 34]
        )

        # noise params for student sampling
        self.current_noise = None
        self.noises = {
            # noise per: [step_xy, step_z, foot_xy, foot_z, outlier_size, outlier_p, episode_xy, episode_z]
            "nominal":   [0.004,   0.005,  0,       0.01,   0.04,         0.03,      0.05,       0.1],
            "offset":    [0.004,   0.005,  0,       0.1,    0.1,          0.1,       0.2,        0.1],
            "noisy":     [0.004,   0.1,    0,       0.1,    0.3,          0.3,       0.3,        0.1]
        }
        self.ep_noises = {
            "left_x": 0,
            "left_y": 0,
            "left_z": 0,
            "right_x": 0,
            "right_y": 0,
            "right_z": 0
        }
        self.update_height_sampler()
        self.update_noises()

    def get_config(self):
        return self.n_row, self.n_col, self.sizes, self.x_coords, self.y_coords, self.sample_coords

    def set_memory(self, pointers):
        self.render_memory = pointers

    def plot_height_sample_pattern(self):
        canv = np.zeros((250, 250))
        for point in self.sample_coords:
            canv[int(point[0]*100) + canv.shape[0]//2, int(point[1]*100) + canv.shape[1]//2] = 1

        fig = plt.figure(figsize=(3.5, 3))
        ax = fig.add_subplot()
        ax.scatter(self.sample_coords[:, 0], self.sample_coords[:, 1], s=2)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        # plt.subplots_adjust(left=0.19, right=0.9, bottom=0.145, top=0.905)
        plt.subplots_adjust(left=0.21, right=0.955, bottom=0.17, top=0.9995)
        plt.show()

    def generate_sampling_pattern(self, radius_per_ring: list, n_points_per_ring: list):
        assert len(radius_per_ring) == len(n_points_per_ring)
        x, y = [], []
        for radius, points in zip(radius_per_ring, n_points_per_ring):
            for p in range(points):
                x.append(radius * np.cos(2*np.pi * p/points))
                y.append(radius * np.sin(2*np.pi * p/points))
        return np.column_stack((x, y))

    def update_height_sampler(self):
        hfield = self._get_hfield()
        if self.render_memory:
            self.render_memory["hfield"][:] = hfield 
        hfield = np.flip(hfield.reshape(self.n_row, self.n_col), 0)
        self.height_sampler = RectBivariateSpline(self.y_coords, self.x_coords, hfield)
        if self.student_mode:
            # hfield_offset = self.generate_hfield_offset()
            hfield_offset = 0 # turned off for now
            self.noisy_height_sampler = RectBivariateSpline(self.y_coords, self.x_coords, hfield + hfield_offset)

    def generate_hfield_offset(self):
        perlin = generate_perlin(frequency=0.02, size=(self.n_row, self.n_col))
        hfield_offset = np.where(perlin < 0, 0, random.uniform(-0.2, 0.2))
        return hfield_offset

    def update_noises(self, noise=None):
        if noise == None:
            self.current_noise = random.choices(list(self.noises.values()), [0.6, 0.3, 0.1])[0]
        else:
            assert noise in self.noises
            self.current_noise = self.noises[noise]
        self.sample_episode_noises()

    def sample_episode_noises(self):
        for noise in self.ep_noises:
            if "x" in noise or "y" in noise:
                self.ep_noises[noise] = np.random.normal(0, self.current_noise[6])
            elif "z" in noise:
                self.ep_noises[noise] = np.random.normal(0, self.current_noise[7])

    def _get_hfield(self):
        hfield_normalized = self.sim_ptr.get_hfield_data()
        hfield = (hfield_normalized * 4) - 2
        return hfield

    def sample(self):
        foot_pos_hfield_left, foot_pos_hfield_right = self._get_foot_pos_hfield()
        sample_coords_x, sample_coords_y = self._get_sample_coords()

        left_x = sample_coords_x + foot_pos_hfield_left[0]
        left_y = sample_coords_y + foot_pos_hfield_left[1]
        left_z = self.height_sampler(left_x, left_y, grid=False)

        right_x = sample_coords_x + foot_pos_hfield_right[0]
        right_y = sample_coords_y + foot_pos_hfield_right[1]
        right_z = self.height_sampler(right_x, right_y, grid=False)

        if self.render_memory:
            self.render_memory["left_samples"][:] = np.array((left_x, left_y, left_z)).flatten()
            self.render_memory["right_samples"][:] = np.array((right_x, right_y, right_z)).flatten()

        z_offset = self.get_z_offset()
        return left_z - z_offset, right_z - z_offset

    def noisy_sample(self):
        foot_pos_hfield_left, foot_pos_hfield_right = self._get_foot_pos_hfield()
        sample_coords_x, sample_coords_y = self._get_sample_coords()

        left_x = sample_coords_x + foot_pos_hfield_left[0] + np.random.normal(0, self.current_noise[0], sample_coords_x.size) + np.random.normal(0, self.current_noise[2]) + self.ep_noises["left_x"]
        left_y = sample_coords_y + foot_pos_hfield_left[1] + np.random.normal(0, self.current_noise[0], sample_coords_y.size) + np.random.normal(0, self.current_noise[2]) + self.ep_noises["left_y"]
        left_z = self.noisy_height_sampler(left_x, left_y, grid=False) + np.random.normal(0, self.current_noise[1], left_x.size) + np.random.normal(0, self.current_noise[3]) + self.ep_noises["left_z"]
        outliers = np.where(np.random.uniform(0, 1, left_z.shape) < self.current_noise[5])[0]
        left_z[outliers] += np.random.normal(0, self.current_noise[4], outliers.shape)

        right_x = sample_coords_x + foot_pos_hfield_right[0] + np.random.normal(0, self.current_noise[0], sample_coords_x.size) + np.random.normal(0, self.current_noise[2]) + self.ep_noises["right_x"]
        right_y = sample_coords_y + foot_pos_hfield_right[1] + np.random.normal(0, self.current_noise[0], sample_coords_y.size) + np.random.normal(0, self.current_noise[2]) + self.ep_noises["right_y"]
        right_z = self.noisy_height_sampler(right_x, right_y, grid=False) + np.random.normal(0, self.current_noise[1], right_x.size) + np.random.normal(0, self.current_noise[3]) + self.ep_noises["right_z"]
        outliers = np.where(np.random.uniform(0, 1, right_z.shape) < self.current_noise[5])[0]
        right_z[outliers] += np.random.normal(0, self.current_noise[4], outliers.shape)

        if self.render_memory:
            self.render_memory["left_samples_noisy"][:] = np.array((left_x, left_y, left_z)).flatten()
            self.render_memory["right_samples_noisy"][:] = np.array((right_x, right_y, right_z)).flatten()

        z_offset = self.get_z_offset()
        return left_z - z_offset, right_z - z_offset

    def _get_foot_pos_hfield(self):
        foot_pos_sim = self.sim_ptr.foot_pos()
        foot_pos_hfield = np.array(foot_pos_sim[1::-1] + foot_pos_sim[4:2:-1]) * np.array([-1, 1, -1, 1]) + self.sizes[1]
        return foot_pos_hfield[:2], foot_pos_hfield[2:]

    def _get_sample_coords(self):
        wxyz = self.sim_ptr.qpos()[3:7]
        r = Rotation.from_quat((*wxyz[1:], wxyz[0]))
        rz = -1 * r.as_euler("xyz")[2]
        sample_coords_x, sample_coords_y = self._rotate(self.sample_coords, rz)
        return sample_coords_x, sample_coords_y

    def _rotate(self, coords, angle):
        x_coords = coords[:, 0] * np.cos(angle) + coords[:, 1] * np.sin(angle)
        y_coords = coords[:, 1] * np.cos(angle) - coords[:, 0] * np.sin(angle)
        return x_coords, y_coords

    def get_z_offset(self):
        pelvis_coords = np.array(self.sim_ptr.qpos()[1::-1]) * np.array([-1, 1]) + self.sizes[1]
        z_offset = self.height_sampler(pelvis_coords[0], pelvis_coords[1])[0, 0]
        if self.render_memory:
            self.render_memory["z_offset"].value = z_offset
        return z_offset


class GaitClock:
    def __init__(self, cfg: CassieBaseConfig):
        self.period = cfg.env.gait_period
        self.sigma = 0.8

        self.l_frc_wave, self.l_spd_wave = self._get_waves(cfg.env.gait_ground_ratios[0], cfg.reward.incentive_clock, cfg.env.phase_shifts[0])
        self.r_frc_wave, self.r_spd_wave = self._get_waves(cfg.env.gait_ground_ratios[1], cfg.reward.incentive_clock, cfg.env.phase_shifts[1])

    def _get_waves(self, ground_ratio, incentive, phase_shift):
        ground = [1] * round(self.period*ground_ratio) if incentive else [0] * round(self.period*ground_ratio)
        aerial = [-1.0] * round(self.period*(1-ground_ratio))

        wave = gaussian_filter(ground + aerial, self.sigma, mode="wrap")
        force_wave = np.roll(wave, round(phase_shift * self.period))
        speed_wave = -1 * force_wave if incentive else -1 * (force_wave + 1)

        return force_wave, speed_wave

    def get_gait_clock_coefs(self, step_counter):
        phi = step_counter % self.period
        return self.l_frc_wave[phi], self.l_spd_wave[phi], self.r_frc_wave[phi], self.r_spd_wave[phi]

    def plot_gait_clocks(self):
        time = np.arange(self.period)/self.period

        fig, axs = plt.subplots(2, sharex=True, figsize=(6, 3))
        axs[0].plot(time, self.l_frc_wave, label="force", c='r')
        axs[0].plot(time, self.l_spd_wave, label="velocity", c='g')
        axs[0].set_ylabel('left foot')
        axs[1].plot(time, self.r_frc_wave, c='r')
        axs[1].plot(time, self.r_spd_wave, c='g')
        axs[1].set_ylabel('right foot')
        axs[1].set_xlabel(r"$\phi$")
        fig.legend(frameon=False, loc='upper center', ncol=2)
        fig.subplots_adjust(top=0.87, right=0.95, left=0.12, bottom=0.16)
        plt.show()


class RewardCalculator:
    def __init__(self, config: CassieBaseConfig):
        self.cfg = config
        self.rewards = {key: 0 for key in self.cfg.reward.scales.keys()}
        self.rewards["reward"] = 0
        self.gait_clock = GaitClock(config)
        self.reset()

    def reset(self):
        self.prev_action = 0
        self.foot_airtime = np.zeros(2)

    # following function is inspired by github.com/osudrl/apex and https://arxiv.org/abs/2011.01387
    def calculate_reward(self, step_counter, state_estimate, qpos, qvel, vel_cmd, action, foot_vars, curriculum_factor, failure):
        pelvis_rotation = Rotation.from_quat((*qpos[4:7], qpos[3]))
        heading_basis_change_xy = Rotation.from_euler('xyz', pelvis_rotation.as_euler('xyz') * np.array([0, 0, 1])).inv()
        xy_vel_pelvis_frame = heading_basis_change_xy.apply(qvel[0:3])[0:2]

        left_frc_clock, left_vel_clock, right_frc_clock, right_vel_clock = self.gait_clock.get_gait_clock_coefs(step_counter)
        # using tanh instead of exp to make incentive/noIncentive clock both work
        left_frc_reward = np.tanh(np.pi * left_frc_clock * foot_vars["l_foot_frc"]/self.cfg.reward.foot_force_normalize)
        right_frc_reward = np.tanh(np.pi * right_frc_clock * foot_vars["r_foot_frc"]/self.cfg.reward.foot_force_normalize)
        foot_frc_reward = left_frc_reward + right_frc_reward

        left_vel_reward = np.tanh(np.pi * left_vel_clock * foot_vars["l_foot_vel"]/self.cfg.reward.foot_vel_normalize)
        right_vel_reward = np.tanh(np.pi * right_vel_clock * foot_vars["r_foot_vel"]/self.cfg.reward.foot_vel_normalize)
        foot_vel_reward = left_vel_reward + right_vel_reward

        # foot artime reward
        foot_frcs = np.array([foot_vars["l_foot_frc"], foot_vars["r_foot_frc"]])
        air = foot_frcs < 10
        first_contact = (foot_frcs > 10) * (self.foot_airtime > 0)
        self.foot_airtime += air / (self.cfg.env.sim_rate_hz / self.cfg.env.pd_sim_steps_per_policy_step)
        foot_airtime_reward = ((self.foot_airtime - 0.5) * first_contact).sum() * (sum(vel_cmd[0:2]) > 0.01)
        self.foot_airtime *= air # reset airtime to 0 if foot is not in air

        # TODO consider zero vel command?
        # single foot reward
        contact = foot_frcs > 10
        single_foot_reward = contact.sum() == 1

        # xy planar foot orientations
        l_foot_orient = Rotation.from_quat((*foot_vars["l_foot_orient"][1:], foot_vars["l_foot_orient"][0])).apply([0, 1, 0])
        l_foot_orient_error = np.abs(np.dot([0, 0, 1], l_foot_orient))

        r_foot_orient = Rotation.from_quat((*foot_vars["r_foot_orient"][1:], foot_vars["r_foot_orient"][0])).apply([0, 1, 0])
        r_foot_orient_error = np.abs(np.dot([0, 0, 1], r_foot_orient))

        foot_orient_error = l_foot_orient_error + r_foot_orient_error
        foot_orient_reward = np.exp(-1.5 * foot_orient_error)
        foot_orient_reward = (foot_orient_reward * (1 - curriculum_factor) + curriculum_factor) # TODO fix plotting

        # xy linear velocity rewards
        max_vel = 1
        if np.linalg.norm(vel_cmd[0:2]) == 0:
            lin_vel_reward = np.exp(-2.5 * np.linalg.norm(xy_vel_pelvis_frame)**2)
        elif np.dot(vel_cmd[0:2], xy_vel_pelvis_frame) > max_vel:
            lin_vel_reward = 1
        else:
            lin_vel_reward = np.exp(-2 * (np.dot(vel_cmd[0:2], xy_vel_pelvis_frame) - max_vel)**2)

        #mse xy linear velocity rewards
        lin_vel_mse = np.linalg.norm(vel_cmd[0:2] - xy_vel_pelvis_frame)**2
        lin_vel_mse_reward = np.exp(-2.5 * lin_vel_mse) # TODO tune

        # z rotational velocity reward
        max_rot_vel = 1 
        if vel_cmd[2] == 0:
            ang_vel_reward = np.exp(-qvel[5]**2)
        elif vel_cmd[2] * qvel[5] > max_rot_vel:
            ang_vel_reward = 1
        else:
            ang_vel_reward = np.exp(-2 * (vel_cmd[2]*qvel[5] - max_rot_vel)**2)

        # xy planar pelvis orientation
        pelvis_orient_error = np.sum(np.abs(pelvis_rotation.as_euler('xyz')[:2]))
        pelvis_orient_reward = np.exp(-3 * pelvis_orient_error)

        # linear orthogonal velocity reward
        lin_ort_vel = xy_vel_pelvis_frame - np.dot(vel_cmd[0:2], xy_vel_pelvis_frame) * np.asarray(vel_cmd[0:2])
        lin_ort_vel_reward = np.exp(-5 * np.linalg.norm(lin_ort_vel))

        # body motion reward
        pelvis_motion = qvel[2]**2 + qvel[3]**2 + qvel[4]**2
        pelvis_motion_reward = np.exp(-pelvis_motion)

        # smoothness rewards
        torque = np.asarray(state_estimate.motor.torque)
        torque_penalty = np.abs(torque).mean()
        torque_reward = np.exp(-0.02 * torque_penalty)

        action_penalty = np.abs(self.prev_action - action).mean()
        action_reward = np.exp(-5 * action_penalty)
        self.prev_action = action

        over_max = np.maximum(action - self.cfg.env.action_range_max, 0)
        under_min = np.maximum(self.cfg.env.action_range_min - action, 0)
        action_limit_reward = -np.sum(over_max + under_min)

        constant_reward = True
        termination_reward = -1 * failure

        reward = 0
        for key, value in locals().items():
            if "reward" in key and key in self.cfg.reward.scales:
                self.rewards[key] = value
                reward += self.cfg.reward.scales[key] * value
        self.rewards["reward"] = reward
        return reward


class CassieEnvVis(CassieEnv):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.vis = CassieVis(self.sim)
        self.terrain_generator.set_vis_ptr(self.vis.v)
        self.t = time.monotonic()

    def step(self, action: bool = None, no_actuation: bool = True):
        while self.vis.ispaused():
            self.render()

        if action is None:
            obs = self._random_step(no_actuation=no_actuation)
        else:
            obs = super().step(action)

        return obs

    def _random_step(self, no_actuation: bool):
        action = np.zeros(10) if no_actuation else self.action_space.sample()
        return super().step(action)

    def render(self):
        self.vis.draw(self.sim)

        def time_elapsed():
            return time.monotonic() - self.t

        while time_elapsed() < self.cfg.env.pd_sim_steps_per_policy_step/self.cfg.env.sim_rate_hz: # real time sync framerate
            time.sleep(0.0001)
        self.t = time.monotonic()


if __name__ == "__main__":
    env = CassieEnvVis(config=CassieBaseConfig)
    env.reset()
    env.exteroceptor.plot_height_sample_pattern()
    while True:
        for i in range(50):
            env.step(None)
            env.render()
        env.reset()
