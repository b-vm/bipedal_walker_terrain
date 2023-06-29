import ctypes
from multiprocessing import Process, Array, Value

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from cassie_env import CassieEnv, CassieEnvVis
from config import CassieBaseConfig


def _make_cassie_env(config, **kwargs):
    def _init():
        env = CassieEnv(config, **kwargs)
        return env
    return _init


def get_vectorized_cassie_envs(config: CassieBaseConfig, n_envs: int, normalization: bool = False, **kwargs):
    cassie_envs_no_log = SubprocVecEnv([_make_cassie_env(config, **kwargs) for _ in range(n_envs)])
    cassie_envs = VecMonitor(cassie_envs_no_log)
    if normalization:
        cassie_envs = VecNormalize(cassie_envs, norm_obs_keys=["proprioception"])
    return cassie_envs


def get_cassie_vis(config: CassieBaseConfig, **kwargs):
    env = CassieEnvVis(config, **kwargs)
    env.vis.attach_cam("track")
    return env


class RewardPlotter():
    def __init__(self, max_time_steps: int, rewards: dict):
        self.max_time_steps = max_time_steps
        self.reward_names = list(rewards.keys())
        self.rewards_shape = (len(rewards), max_time_steps)

        self.rewards_mp = Array(ctypes.c_double, len(rewards) * max_time_steps)
        self.rewards = np.frombuffer(self.rewards_mp.get_obj()).reshape(self.rewards_shape)

        self.time_step_mp = Value("i", 0)

        self.render_process = Process(target=self._plot)
        self.render_process.start()

    def add_rewards(self, rewards: dict):
        for i, reward in enumerate(rewards.values()):
            self.rewards[i, self.time_step_mp.value] = reward
        if self.time_step_mp.value + 1 < self.max_time_steps:
            self.time_step_mp.value += 1

    def reset_rewards(self):
        self.rewards[:] = np.zeros(self.rewards_shape)
        self.time_step_mp.value = 0

    def __del__(self):
        self.render_process.terminate()

    def _plot(self):
        fig, axs = plt.subplots(len(self.reward_names), 1, sharex=True)
        plt.subplots_adjust(bottom=0.05, top=0.95)

        rewards = np.frombuffer(self.rewards_mp.get_obj()).reshape(self.rewards_shape)

        def update(_):
            for i, reward in enumerate(rewards):
                axs[i].clear()
                axs[i].plot(reward[:self.time_step_mp.value])
                axs[i].set_xlim(0, self.max_time_steps)
                if self.reward_names[i] in ["foot_frc_reward", "foot_vel_reward", "foot_airtime_reward"]:
                    axs[i].set_ylim(-1, 1)
                elif self.reward_names[i] in ["reward"]:
                    axs[i].set_ylim(0, 2)
                elif self.reward_names[i] in ["action_limit_reward"]:
                    axs[i].set_ylim(-2, 0)
                else:
                    axs[i].set_ylim(0, 1)
                axs[i].set_title(self.reward_names[i])
                axs[i].grid()

        animation = FuncAnimation(fig, update, interval=10)
        plt.show()


class ExteroceptionVis():
    def __init__(self, n_row, n_col, sizes, x_coords, y_coords, sample_coords, student_mode=False):
        self.n_row = n_row
        self.n_col = n_col
        self.sizes = sizes
        self.xv, self.yv = np.meshgrid(x_coords, y_coords)
        self.sample_coords = sample_coords
        self.student_mode = student_mode

        self.render_size = (80, 80)
        self.render_proc = None

        # allocate shared memory for vars that are plotted in a separate process
        self.left_samples_mp = Array(ctypes.c_double, 3 * self.sample_coords.shape[0])
        self.right_samples_mp = Array(ctypes.c_double, 3 * self.sample_coords.shape[0])
        self.left_samples_noisy_mp = Array(ctypes.c_double, 3 * self.sample_coords.shape[0])
        self.right_samples_noisy_mp = Array(ctypes.c_double, 3 * self.sample_coords.shape[0])
        self.left_reconstruction_mp = Array(ctypes.c_double, self.sample_coords.shape[0])
        self.right_reconstruction_mp = Array(ctypes.c_double, self.sample_coords.shape[0])
        self.hfield_mp = Array(ctypes.c_double, self.n_row*self.n_col)
        self.pelvis_coords_mp = Array(ctypes.c_double, 2)
        self.z_offset_mp = Value("f", 0)

        self.memory_pointers = {
            "left_samples": self.left_samples_mp,
            "right_samples": self.right_samples_mp,
            "left_samples_noisy": self.left_samples_noisy_mp,
            "right_samples_noisy": self.right_samples_noisy_mp,
            "left_reconstruction": self.left_reconstruction_mp,
            "right_reconstruction": self.right_reconstruction_mp,
            "hfield": self.hfield_mp,
            "pelvis_coords": self.pelvis_coords_mp,
            "z_offset": self.z_offset_mp
        }

    def get_memory_pointers(self):
        return self.memory_pointers

    def store_obs_reconstruction(self, obs_reconstruction):
        self.memory_pointers["left_reconstruction"][:] = obs_reconstruction["left"].flatten() + self.z_offset_mp.value
        self.memory_pointers["right_reconstruction"][:] = obs_reconstruction["right"].flatten() + self.z_offset_mp.value

    def render(self, qpos):
        self.pelvis_coords_mp[:] = np.array(qpos[1::-1]) * np.array([-1, 1]) + self.sizes[1]
        if not isinstance(self.render_proc, Process):
            # run the matplotlib in a separate process to prevent blocking of main process
            self.render_proc = Process(target=self._render)
            self.render_proc.start()

    def _render(self):
        hfield = np.flip(np.frombuffer(self.hfield_mp.get_obj()).reshape(self.n_row, self.n_col), 0)
        left_samples = np.frombuffer(self.left_samples_mp.get_obj()).reshape(3, self.sample_coords.shape[0])
        right_samples = np.frombuffer(self.right_samples_mp.get_obj()).reshape(3, self.sample_coords.shape[0])

        left_samples_noisy = np.frombuffer(self.left_samples_noisy_mp.get_obj()).reshape(3, self.sample_coords.shape[0])
        right_samples_noisy = np.frombuffer(self.right_samples_noisy_mp.get_obj()).reshape(3, self.sample_coords.shape[0])

        figure = plt.figure()
        ax = Axes3D(figure, computed_zorder=False, auto_add_to_figure=False)
        figure.add_axes(ax)

        self.plot_options = {
            "surface": lambda yv, xv, hfield_ : ax.plot_surface(yv, xv, hfield_, rstride=2, cstride=2, color='0.7', zorder=0),
            "left": lambda: ax.scatter(left_samples[0, :], left_samples[1, :], left_samples[2, :], c='g', depthshade=False, zorder=5, label="left"),
            "right": lambda: ax.scatter(right_samples[0, :], right_samples[1, :], right_samples[2, :], c='r', depthshade=False, zorder=10, label="right"),
            "left_noisy": lambda: ax.scatter(left_samples_noisy[0, :], left_samples_noisy[1, :], left_samples_noisy[2, :], c='c', depthshade=False, zorder=15, label="left_noisy"),
            "right_noisy": lambda: ax.scatter(right_samples_noisy[0, :], right_samples_noisy[1, :], right_samples_noisy[2, :], c='tab:orange', depthshade=False, zorder=20, label="right noisy"),
            "left_reconstruction": lambda: ax.scatter(left_samples[0, :], left_samples[1, :], self.left_reconstruction_mp[:], c='y', depthshade=False, zorder=25, label="left_reconstruction"),
            "right_reconstruction": lambda: ax.scatter(right_samples[0, :], right_samples[1, :], self.right_reconstruction_mp[:], c='m', depthshade=False, zorder=30, label="right_reconstruction"),
        }

        if self.student_mode:
            self.plots = ["surface", "left_noisy", "left_reconstruction"]
            # self.plots = ["surface", "left_noisy", "right_noisy"]
        else:
            self.plots = ["surface", "left", "right"]

        def update(_):
            yv, xv, _ = zoom_in()
            for i, plot in enumerate(self.plots):
                self.plot_objects[i].remove()
                if plot == "surface":
                    self.plot_objects[i] = self.plot_options[plot](*zoom_in())
                else:
                    self.plot_objects[i] = self.plot_options[plot]()

            ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
            ax.set_ylim(xv.min(), xv.max())
            ax.set_xlim(yv.min(), yv.max())
            return ax

        def zoom_in():
            a = int(self.pelvis_coords_mp[0]*self.n_row/self.sizes[1]/2)
            b = int(self.pelvis_coords_mp[1]*self.n_col/self.sizes[0]/2)
            origin = (min(max(a-self.render_size[0]//2, 0), self.n_row-self.render_size[0]), min(max(b-self.render_size[1]//2, 0), self.n_col-self.render_size[1]))

            hfield_zoom = hfield[origin[0]:origin[0]+self.render_size[0], origin[1]:origin[1]+self.render_size[0]]
            yv_zoom = self.yv[origin[0]:origin[0]+self.render_size[0], origin[1]:origin[1]+self.render_size[0]]
            xv_zoom = self.xv[origin[0]:origin[0]+self.render_size[0], origin[1]:origin[1]+self.render_size[0]]
            return yv_zoom, xv_zoom, hfield_zoom

        self.plot_objects = []
        for plot in self.plots:
            if plot == "surface":
                self.plot_objects.append(self.plot_options[plot](*zoom_in()))
            else:
                self.plot_objects.append(self.plot_options[plot]())

        ax.set_zlim(-1.5, 1.5)
        ax.view_init(elev=28, azim=0)
        # ax.view_init(elev=0, azim=0) # usefull for side screenshots
        ax.dist = 7
        ax.set_proj_type('persp')
        ax.grid(visible=False)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        figure.legend()
        animation = FuncAnimation(figure, update, interval=40)
        plt.show()

    def __del__(self):
        if isinstance(self.render_proc, Process):
            self.render_proc.terminate()


class KeyboardController():
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.stdscr.nodelay(True)
        self.cmds = {
            ord("w"): (1, 0, 0),
            ord("a"): (0, 1, 0),
            ord("s"): (-1, 0, 0),
            ord("d"): (0, -1, 0),
            ord("q"): (0, 0, 1),
            ord("e"): (0, 0, -1),
            -1: (0, 0, 0)
        }
        self.buffer_length = 30
        self.cmd_buffer = [-1] * self.buffer_length
        self.stop_cmd = 27 # Esc

    def get_cmd(self):
        cmd = self.stdscr.getch()
        self._flush_for_exit_cmd()
        if cmd == self.stop_cmd:
            exit()

        # key input buffering, get last command
        self.cmd_buffer.append(cmd)
        self.cmd_buffer.pop(0)
        cmds = [cmd for cmd in self.cmd_buffer if cmd > 0]
        if cmds:
            cmd = cmds[-1]

        self.stdscr.clear()
        if cmd in self.cmds:
            self.stdscr.addstr(str(self.cmds[cmd]))
            return self.cmds[cmd]
        else:
            self.stdscr.addstr(str(self.cmds[-1]))
            return self.cmds[-1]

    def _flush_for_exit_cmd(self):
        key = ""
        while key != -1:
            key = self.stdscr.getch()
            if key == self.stop_cmd:
                exit()