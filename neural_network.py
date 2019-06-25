import numpy as np

class Node:
    def __init__(self, z=0, w=None):
        """Creates a new node object.
        
        Parameters:
        z(float) -- the weighted sum of the previous layer
        weights(numpy array) -- the weights from the prvious layer connected to the node

        Defined Attributes:
        a(float) -- the value from putting z through an activation function
        """

        self.z = z
        self.w = w
        self.a = max(0, z)

class Neural_Network:
    def __init__(self, inputs, bellman, lr):
        """Initializes the network and adds the input layer.
        
        Parameters:
        inputs(numpy array) -- the input training data
        bellman(float) -- the optimal Q value? *********
        lr(float) -- the learning rate

        Defined Attributes:
        self.network(list) -- the list holding the network
        self.layers(int) -- the amount of layers in the network
        self.act_func(function) -- the activation function rel_u
        self.bellman(float) -- the optimal Q value? *******
        self.lr(float) -- the learning rate
        """
        #creates a Node for each input data
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
        """Adds a new layer to the network with weights randomized according to the normal
        distribution N(0, 1/n)

        Parameters:
        num_nodes(int) -- the number of nodes in the new layer
        """
        new_layer = np.zeros(num_nodes)
        prev_layer = self.network[self.layers-1]
        for i in range(num_nodes):
            #normalzies the weights
            w = np.random.randn(prev_layer) * (1/prev_layer)**.5 
            new_layer[i] = Node(0, w)
        network.append(new_layer)

        self.layers += 1

    def epoch(self):
        """Goes through one epoch of a training sample by feeding the data forward
        and then updating the weights with SGD.
        """
        #computes the weighted sum of the previous later for each node and 
        #puts is through the activation function
        for l, layer in enumerate(network[1:], start=1):
            for node in layer:
                sum = 0
                for i, w in enumerate(node.w):
                    sum += w*network[l-1][i]
                node.d = sum
        
        output_a = np.array([node.a for node in network[-1]])

        #find the error in the output layer
        errors = np.zeros(self.layers)
        errors[-1] = (output_a - bellman).dot(output_a)

        l = self.layers - 2
        while(i > 0):
            layer_a = np.array([node.a for node in network[l]])
            errors[i] = ((network[l+1]).T*errors[l+1]).dot(layer_a)
        
        for l, layer in enumerate(network[1:], start=1):
            for node in layer:
                for i,w in enumerate(node):
                    node[i] = w - (self.lr * np.array([node.a for node in network[l-1]]) * errors[l])
        

                
    