import numpy as np
import copy

class Neural_Network:
    def __init__(self, input_length, lr, discount):
        """Initializes the network and adds the input layer.
        
        Parameters:
        input_length(int) -- the number of nodes in the input layer
        lr(float) -- the learning rate
        discount(float) -- the discount rate

        Defined Attributes:
        self.weights(list of numpy arrays) -- the weights for the policy network
        self.t_weights(list of numpy arrays) -- the weights for the target network
        self.biases(list of numpy arrays) -- the biases for the policy network
        self.t_biases(list of numpy arrays) -- the biases for the target network
        self.layers(int) -- the amount of layers in the network
        self.act_func(function) -- the activation function rel_u
        self.lr(float) -- the learning rate
        self.discount(float) -- the discount rate
        """
        #initializes the policy network
        self.weights = []
        self.biases = []
        self.weights.append(np.array([None for _ in range(input_length)]))
        self.biases.append(np.array([None for _ in range(input_length)]))

        #initializes the target network as a copy of the policy network
        self.t_weights = copy.deepcopy(self.weights)
        self.t_biases = copy.deepcopy(self.biases)

        self.layers = 1
        self.act_func = lambda x: max(0, x)
        self.lr = lr
        self.discount = discount

    def add_layer(self, num_nodes):
        """Adds a new layer to the network with weights randomized according to the normal
        distribution N(0, 1/n) and biases set to 0. Increments self.layers.

        Parameters:
        num_nodes(int) -- the number of nodes in the new layer
        """
        new_layer = np.zeros(num_nodes)
        prev_layer_length = len(self.values[self.layers-1])
        self.biases.append(np.array([np.zeros(num_nodes) for _ in range(num_nodes)]))
        for i in range(num_nodes):
            #normalzies the weights
            w = np.random.randn(prev_layer_length) * (1/prev_layer_length)**.5 
            new_layer[i] = w
        self.weights.append(new_layer)

        #updates the target network
        self.t_weights = copy.deepcopy(self.weights)
        self.t_biases = copy.deepcopy(self.biases)

        self.layers += 1

    def update_target_network(self):
        self.t_weights = copy.deepcopy(self.weights)
        self.t_biases = copy.deepcopy(self.biases)

    def epoch(self, batches):
        """Goes through one epoch of a training sample by feeding the data forward
        and then updating the weights with SGD.

        Parameters:
        batches(list of tuples of (st, at, rt+1, st+1)) -- a batch of replay memory
        """
        activations_for_batch = []
        errors_for_batch = []
        for batch, data_set in enumerate(batches):
            #sets the z and activation values of the first layer for the policy network
            z_values = [[x[0] for x in data_set]]
            a_values = [[act_func(x[0]) for x in data_set]]
            #sets the z and activation values of the first layer for the target network
            z_values_t = [[x[0] for x in data_set]]
            a_values_t = [[act_func(x[0]) for x in data_set]]

            #computes the weighted sum with a bias of the previous later for each node in the policy and target networks
            for l in range(1, self.layers):
                z_policy = self.weights[l].dot(a_values[l-1]) + self.biases[l]
                z_target = self.t_weights[l].dot(a_values_t[l-1]) + self.t_biases[l]

                z_values.append(z_policy)
                z_values_t.append(z_target)
            
                a_values.append(np.maximum(z_policy, np.zeros(len(z_policy))))
                a_values_t.append(np.maximum(z_target, np.zeros(len(z_target))))

            q_optimal = np.amax(z_values[-1])
            bellman = data_set[2] + self.discount * q_optimal
            errors = np.zeros(self.layers)

            #find the error in the output layer
            errors[-1] = (a_values[-1] - bellman).dot(a_values[-1])

            #finds the errors of the other layers by calculating the loss
            l = self.layers - 2
            while(l > 0):
                errors[i] = ((self.weights[l+1]).T*errors[l+1]).dot(a_values[l])
            
            errors_for_batch.append(errors)
            activations_for_batch.append(a_values)
        
        #updates the weights and biases 
        l = self.layers-1
        while(l > 0):
            sum_weights = 0
            sum_biases = 0
            for x, training_set in enumerate(batches):
                sum_weights += errors_for_batch[x][l] * activations_for_batch[x][l-1]
                sum_biases += errors_for_batch[x][l]
            self.weights[l] = self.weights[l] - self.lr * sum_weights / len(batches)
            self.biases[l] = self.biases[l] - self.lr * sum_biases / len(batches)
        
    def generate_batches(self, inputs, batch_size):
        batches = [input[i * batch_size:(i + 1) * batch_size] for i in range((len(inputs) + batch_size - 1) // batch_size)]

                
    