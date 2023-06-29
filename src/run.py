import argparse
import os
from curses import wrapper as curses_wrapper

from sb3_contrib import RecurrentPPO

from cassie_env import CassieEnvVis
from train_student import Model
from utils import RewardPlotter, ExteroceptionVis, KeyboardController, get_cassie_vis
from config import CassieBaseConfig


def run(model, env: CassieEnvVis, student_mode: bool = False, plot_reward: bool = False, keyboard_controller=None):
    if plot_reward:
        reward_plotter = RewardPlotter(env.cfg.env.episode_step_limit, env.get_rewards())
    else:
        exteroception_vis = ExteroceptionVis(*env.get_exteroception_config(), student_mode=student_mode)
        env.set_exteroception_memory(exteroception_vis.get_memory_pointers())
        exteroception_vis.render(env.get_pelvis_coords())

    while True:
        done = False
        obs = env.reset()
        hidden_state = None
        episode_start = True
        if plot_reward:
            reward_plotter.reset_rewards()
        if student_mode:
            env.set_student_exteroception_noise("nominal")

        while not done or keyboard_controller:
            if keyboard_controller:
                cmd = keyboard_controller.get_cmd()
                env.set_velocity_commands(cmd)
            action, hidden_state = model.predict(obs, state=hidden_state, episode_start=episode_start, deterministic=True)
            obs, _, done, _ = env.step(action)
            episode_start = done
            env.render()
            if plot_reward:
                reward_plotter.add_rewards(env.get_rewards())
            else:
                exteroception_vis.render(env.get_pelvis_coords())
                if student_mode:
                    exteroception_vis.store_obs_reconstruction(model.get_obs_reconstruction())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, choices=["student", "teacher"])
    parser.add_argument('observationFn', type=str, choices=["privileged", "exteroception", "proprioception"])
    parser.add_argument('name', type=str)
    parser.add_argument('epoch', type=int)
    parser.add_argument('curriculum_factor', type=float)
    parser.add_argument('-plotReward', action='store_true')
    parser.add_argument('-noActionOffset', action='store_false')
    parser.add_argument('-keyboardControl', action='store_true')
    args = parser.parse_args()

    if args.model_type == 'teacher':
        config = CassieBaseConfig()
        config.env.obs_fn = args.observationFn
        config.env.curriculum_factor = args.curriculum_factor
        config.env.action_offset = args.noActionOffset
        config.randomization.randomization = False
        config.terrain.generation = True

        env = get_cassie_vis(config)

        load_path = os.path.join(f"{args.model_type}_log", args.name, f"teacher_model_{args.epoch}k.zip")
        model = RecurrentPPO.load(load_path)

    elif args.model_type == 'student':
        config = CassieBaseConfig()
        config.env.obs_fn = args.observationFn
        config.env.curriculum_factor = args.curriculum_factor
        config.env.action_offset = args.noActionOffset
        config.randomization.randomization = False
        config.terrain.generation = True
        config.student.student_mode = True

        env = get_cassie_vis(config)

        load_path = os.path.join(f"{args.model_type}_log", args.name, f"model_{args.epoch}")
        model = Model.load(env, load_path)

    if args.keyboardControl:
        def run_wrapper(stdscr):
            keyboard_controller = KeyboardController(stdscr)
            run(model, env, config.student.student_mode, plot_reward=args.plotReward, keyboard_controller=keyboard_controller)
        curses_wrapper(run_wrapper)
    else:
        run(model, env, config.student.student_mode, plot_reward=args.plotReward)
