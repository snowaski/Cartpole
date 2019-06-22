import numpy as np

#Each node holds its data and the weights from the previous layer
class Node:
    def __init__(self, d=0, weights=None):
        self.data = d
        self.w = weights

class Neural_Network:
    def __init__(self, inputs, bellman, lr):
        self.network = []
        input_layer = np.zeros(len(inputs))
        for i, val in enumerate(inputs):
            input_layer[i] = Node(val)
        self.network.append(input_layer)
        self.layers = 1
        self.act_func = lambda x: max(0, x)
        self.bellman = bellman
        self.lr = lr

    def add_layer(self, num_nodes):
        new_layer = np.zeros(num_nodes)
        connected_nodes = self.network[self.layers-1]
        for i in range(num_nodes):
            w = np.random.randn(connected_nodes) * (1/connected_nodes)**.5 
            new_layer[i] = Node(0, w)
        network.append(new_layer)
        self.layers += 1

    def epoch(self):
        for l, layer in enumerate(network[1:], start=1):
            for node in layer:
                sum = 0
                for i, w in enumerate(node.w):
                    sum += w*network[l-1][i]
                if l+1 == self.layers:
                    node.data = sum
                else:
                    node.data = self.act_func(sum)
        for node in network[self.layers-1]:
            for i, w in enumerate(node.w):
                gradient = 2*(self.act_func() - self.bellman) * self.act_func(node.data) * node.w[i]
                new_weight = w - (lr * gradient)
                
    