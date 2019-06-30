import gym
import numpy as np
import neural_network as nn

env = gym.make('CartPole-v0')
print("Action Space: ",env.action_space.n)
print("Observation Space: ", env.observation_space)
print("Obs space bounds", env.observation_space.high)

