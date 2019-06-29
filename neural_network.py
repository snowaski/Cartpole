import numpy as np

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
        self.weights = []
        self.values = []
        self.biases = []
        self.weights.append(np.array([None for _ in range(len(inputs))]))
        self.values.append(np.array(inputs)) 
        self.biases.append(np.array([None for _ in range(len(inputs))]))

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
        self.values.append(np.array([np.zeros(num_nodes) for _ in range(num_nodes)]))
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
        z_values = []
        #computes the weighted sum with a bias of the previous later for each node
        for l, layer in enumerate(self.values[1:], start=1):
            z = self.weights[l].dot(self.values[l-1]) + self.biases[l]
            z_values.append(z)

        #puts the z values through an activation function
        for i, v in enumerate(self.values[1:], start=1):
            self.values[i] = np.maximum(v, np.zeros(len(v)))
        
        errors = np.zeros(self.layers)

        #find the error in the output layer
        errors[-1] = (self.values[-1] - bellman).dot(self.values[-1])

        #finds the errors of the other layers by calculating the loss
        l = self.layers - 2
        while(l > 0):
            errors[i] = ((self.weights[l+1]).T*errors[l+1]).dot(values[l])
        
        #updates the weights and biases 
        for l, layer in enumerate(self.values[1:], start=1):
            for n, z in enumerate(layer):
                for i, pair in enumerate(zip(list(self.weights[l][n]), list(self.biases[l][n]))):
                    self.weights[l][n][i] = pair[0] - (self.lr * errors[l] * values[l-1])
                    self.biases[l][n][i] = pair[1] -  self.lr * errors[l]
        

                
    