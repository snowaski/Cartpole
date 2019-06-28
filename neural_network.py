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
        self.weights = []
        self.values = []
        self.weights.append(np.array([None for _ in range(len(inputs))]))
        self.values.append(np.array(inputs)) 

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
        prev_layer_length = len(self.values[self.layers-1])
        self.values.append(np.zeros(num_nodes))
        for i in range(num_nodes):
            #normalzies the weights
            w = np.random.randn(prev_layer_length) * (1/prev_layer_length)**.5 
            new_layer[i] = w
        self.weights.append(new_layer)

        self.layers += 1

    def epoch(self):
        """Goes through one epoch of a training sample by feeding the data forward
        and then updating the weights with SGD.
        """
        #computes the weighted sum of the previous later for each node and 
        #puts is through the activation function
        for l, layer in enumerate(self.values[1:], start=1):
            for n, z in enumerate(layer):
                sum = 0
                for i, w in enumerate(self.weights[l][n]):
                    sum += w*self.values[l-1][i]
                self.values[l][n] = self.act_func(sum)
        
        errors = np.zeros(self.layers)

        #find the error in the output layer
        errors[-1] = (self.values[-1] - bellman).dot(self.values[-1])

        #finds the errors of the other layers by calculating the loss
        l = self.layers - 2
        while(l > 0):
            errors[i] = ((self.weights[l+1]).T*errors[l+1]).dot(values[l])
        
        #updates the weights
        for l, layer in enumerate(self.values[1:], start=1):
            for n, z in enumerate(layer):
                for i,w in enumerate(self.weights[l][n]):
                    self.weights[l][i] = w - (self.lr * errors[l] * values[l-1])
        

                
    