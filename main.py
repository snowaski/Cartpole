import gym
import numpy as np
import neural_network as nn

new_layer = [np.random.randn(3) * (1/3)**.5] * 5
print(new_layer)
list = [1,2,3,4]
for l, layer in enumerate(list[1:], start=1):
    print(l, " " ,layer)

env = gym.make('CartPole-v0')
print("Action Space: ",env.action_space.n)
print("Observation Space: ", env.observation_space)
print("Obs space bounds", env.observation_space.high)