import gym
from monte_carlo_control import MonteCarloControl
import plotly.graph_objects as go
import pandas as pd
from os import path


def plot_value_function(value_function=None):
    if path.exists('value_f.csv'):
        df = pd.read_csv('value_f.csv')
        df = df.drop(columns=['Unnamed: 0'], axis=1)
        df.index = df.index + 1
    else:
        df = pd.DataFrame(columns=range(1, 11), index=range(1, 22))
        for state, value in value_function.items():
            df.at[state[0], state[1]] = value
        df.to_csv('value_f.csv')

    fig = go.Figure(data=[go.Surface(z=df.values, x=df.columns, y=df.index)])
    fig.update_layout(title='Value function', autosize=True)
    fig.show()


if __name__ == "__main__":
    env = gym.make("gym_easy21:easy21-v0")
    mc_control = MonteCarloControl(env, env.observation_space, env.action_space)
    _ = mc_control.run()
    value_function = mc_control.get_value_function()
    plot_value_function()
