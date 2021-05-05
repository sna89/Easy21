import gym
from monte_carlo_control import MonteCarloControl


if __name__ == "__main__":
    env = gym.make("gym_easy21:easy21-v0")
    mc_control = MonteCarloControl(env, env.observation_space, env.action_space)
    mc_control.generate_episode()
