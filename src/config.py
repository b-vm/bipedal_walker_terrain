import os
import inspect

import numpy as np


class BaseConfig:
    def __init__(self) -> None:
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        for key in dir(obj):
            if key=="__class__":
                continue
            var =  getattr(obj, key)
            if inspect.isclass(var):
                instantiated_var = var()
                setattr(obj, key, instantiated_var)
                BaseConfig.init_member_classes(instantiated_var)


class CassieBaseConfig(BaseConfig):
    class env:
        sim_rate_hz = 2000
        pd_sim_steps_per_policy_step = 50
        height_limit_min = 0.4
        height_limit_max = 3
        episode_step_limit = 300
        n_actuators = 10
        neutral_action = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968] * 2)
        action_range_min = np.array([-0.26179938779, -0.39269908169, -0.87266462599, -2.86233997327, -2.44346095279] * 2)
        action_range_max = np.array([ 0.39269908169,  0.39269908169,  1.3962634016,  -0.64577182323, -0.52359877559] * 2)
        action_scale = np.array([0.4, 0.4, 1, 1, 1] * 2)

        action_offset = True
        action_scaling = True

        init_velocity_commands = (1, 0, 0) # (v_x, v_y, omega_z)
        gait_period = 28
        gait_ground_ratios = (0.6, 0.6)
        phase_shifts = (0, 0.5)

        obs_clock_input = True # doesnt seem to be needed though
        curriculum_factor = 0
        cassie_model_file = "cassie2.xml"
        obs_fn = "exteroception" # proprioception, exteroception or privileged
        benchmarking_mode = False

    class control:
        P = np.array([100,  100,  88,  96,  50])
        D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

    class reward:
        scales = {
            "foot_frc_reward":      0.250,
            "foot_vel_reward":      0.250,
            "foot_airtime_reward":  0,
            "single_foot_reward":   0,
            "lin_vel_reward":       0.200,
            "lin_vel_mse_reward":   0,
            "ang_vel_reward":       0.200,
            "lin_ort_vel_reward":   0.050,
            "foot_orient_reward":   0.050,
            "pelvis_motion_reward": 0.050,
            "pelvis_orient_reward": 0.050,
            "torque_reward":        0.025,
            "action_reward":        0.025,
            "action_limit_reward":  0.100,
            "constant_reward":      0.200,
            "termination_reward":   0,
        }

        @staticmethod
        def switch_gait_reward(scales):
            scales["foot_frc_reward"] = 0
            scales["foot_vel_reward"] = 0
            scales["foot_airtime_reward"] = 1
            scales["single_foot_reward"] = 0.1
            scales["constant_reward"] = 0

        foot_force_normalize = 300
        foot_vel_normalize = 2.5
        incentive_clock = False

    class randomization:
        randomization = True

        class command:
            uniform = False
            # velocity_range = [0.5, 1.5] # TODO

        class domain: #values from http://www.roboticsproceedings.org/rss17/p061.pdf
            damping_range = [0.5, 3.5]
            mass_range = [0.5, 1.7]
            friction_range = [0.5, 1.1]

        class proprioception:
            add_noise = False # remove these subsettings because it is confusing
            joint_pos_range = [-0.01, 0.01]
            joint_vel_range = [-1.5, 1.5]
            pelvis_orientation_range = [-0.05, 0.05]
            pelvis_angular_velocity_range = [-0.2, 0.2]

        class exteroception:
            add_noise = False
            offset_noise_range = [-0.1, 0.1] # TODO, or consistent noise for whole episode?

    class terrain:
        generation = True

    class student:
        student_mode = False


class StudentTrainingConfig(CassieBaseConfig): # TODO
    class env(CassieBaseConfig.env):
        curriculum_factor = 1 # this is randomized anyway

    class student(CassieBaseConfig.student):
        student_mode = True


class BenchmarkConfig(CassieBaseConfig):
    class env(CassieBaseConfig.env):
        curriculum_factor = 1
        benchmarking_mode = True

    class randomization(CassieBaseConfig.randomization):
        randomization = False

    class terain(CassieBaseConfig.terrain):
        generation = False


class PPOConfig(CassieBaseConfig):
    class PPO:
        min_buffer_length = 50000
        learning_rate = 0.0001
        num_envs = os.cpu_count()

        log_dir = "teacher_log"

    class curriculum:
        terrain_start = 12e6
        terrain_end = 40e6
        gait_clock_reward_end = 10e6

        benchmarks_start = 40e6
        benchmarks_interval = 5e6
        datapoints_per_benchmark = 30