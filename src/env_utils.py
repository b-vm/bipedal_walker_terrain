import numpy as np
import pyfastnoisesimd

from cassie_mujoco_sim import pd_in_t
from config import CassieBaseConfig


def generate_perlin(frequency: float, size: tuple):
    perlin_generator = pyfastnoisesimd.Noise(numWorkers=1)
    perlin_generator.noiseType = pyfastnoisesimd.NoiseType.Perlin
    perlin_generator.frequency = frequency
    perlin_generator.seed = np.random.randint(-2147483648, 2147483647)
    return  perlin_generator.genAsGrid(size)


def get_pd_input(action, cfg):
    pd_input = pd_in_t()
    for i in range(5):
        pd_input.leftLeg.motorPd.pTarget[i] = action[i]
        pd_input.rightLeg.motorPd.pTarget[i] = action[i+5]
        for leg in [pd_input.leftLeg, pd_input.rightLeg]:
            leg.motorPd.torque[i] = 0 
            leg.motorPd.dTarget[i] = 0
            leg.motorPd.pGain[i] = cfg.control.P[i]
            leg.motorPd.dGain[i] = cfg.control.D[i]

    return pd_input


class BenchmarkLogger:
    def __init__(self, cfg: CassieBaseConfig):
        self.velocity_command_trajectory = None
        self.default_velocity_commands = cfg.env.init_velocity_commands

        self.benchmarks = ["v_x", "v_y", "omega_z", "torque"]
        self.benchmark_results = {}
        self.reset()

    def log(self, done, failure, state_estimate, qvel):
        self.benchmark_results["v_x"].append(qvel[0])
        self.benchmark_results["v_y"].append(qvel[1])
        self.benchmark_results["omega_z"].append(qvel[5])
        self.benchmark_results["torque"].append(np.array(state_estimate.motor.torque[:]).sum())

        if done:
            return {
                "done_fail": failure,
                "v_x": np.array(self.benchmark_results["v_x"]),
                "v_y": np.array(self.benchmark_results["v_y"]),
                "omega_z": np.array(self.benchmark_results["omega_z"]),
                "torque": np.array(self.benchmark_results["torque"]),
            }
        return {}

    def reset(self):
        self.velocity_command_trajectory = None
        for var in self.benchmarks:
            self.benchmark_results[var] = []

    def set_velocity_command_trajectory(self, velocity_command_trajectory):
        self.velocity_command_trajectory = velocity_command_trajectory

    def get_velocity_commands(self, step_counter):
        if not isinstance(self.velocity_command_trajectory, np.ndarray):
            return self.default_velocity_commands
        else:
            return self.velocity_command_trajectory[step_counter]