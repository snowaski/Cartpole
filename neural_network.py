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
        self.greedy(int) -- the action with the higest q value
        """
        #initializes the policy network
        self.weights = []
        self.biases = []
        # self.weights.append(np.array([None for _ in range(input_length)]))
        # self.biases.append(np.array([None for _ in range(input_length)]))

        #initializes the target network as a copy of the policy network
        self.t_weights = copy.deepcopy(self.weights)
        self.t_biases = copy.deepcopy(self.biases)

        self.layers = 1
        self.act_func = lambda x: max(0, x)
        self.lr = lr
        self.discount = discount
        self.input_length = input_length


    def add_layer(self, num_nodes):
        """Adds a new layer to the network with weights randomized according to the normal
        distribution N(0, 1/n) and biases set to 0. Increments self.layers.

        Parameters:
        num_nodes(int) -- the number of nodes in the new layer
        """
        new_layer = [[] for _ in range(num_nodes)]
        if self.layers == 1:
            prev_layer_length = self.input_length
        else:
            prev_layer_length = len(self.weights[self.layers-2])
        self.biases.append(np.zeros(num_nodes))
        for i in range(num_nodes):
            #normalzies the weights
            w = np.random.randn(prev_layer_length) * (1/prev_layer_length)**.5 
            new_layer[i] = w
        self.weights.append(np.array(new_layer))

        #updates the target network
        self.t_weights = copy.deepcopy(self.weights)
        self.t_biases = copy.deepcopy(self.biases)

        self.layers += 1

    def update_target_network(self):
        self.t_weights = copy.deepcopy(self.weights)
        self.t_biases = copy.deepcopy(self.biases)

    def epoch(self, batch):
        """Goes through one epoch of a training sample by feeding the data forward
        and then updating the weights with SGD.

        Parameters:
        batches(list of tuples of (s_t, a_t, r_t+1, s_t+1)) -- a batch of replay memory
        """
        # activations_for_batch = []
        # errors_for_batch = []

        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        for b, data_set in enumerate(batch):
            delta_gradient_b, delta_gradient_w = self.backprop(data_set)
            gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
            gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]
        
        self.weights = [w-(self.lr/len(batch)) * nw for w, nw in zip(self.weights, gradient_w)]
        self.biases = [b-(self.lr/len(batch)) * nb for b, nb in zip(self.biases, gradient_b)]
            # state = data_set[0]
            # #sets the z and activation values of the first layer for the policy network
            # z_values = [state]
            # a_values = [np.maximum(state, np.zeros(len(state)))]

            # #computes the weighted sum with a bias of the previous later for each node in the policy and target networks
            # for l in range(1, self.layers):
            #     z_policy = self.weights[l].dot(a_values[l-1]) + self.biases[l]

            #     z_values.append(z_policy)
            
            #     a_values.append(np.maximum(z_policy, np.zeros(z_policy.shape)))
            
            # #determining the training labels by passing the next state to the target network,
            # #determining the action with the max q value and setting that label to the bellman
            # #equation and all others to their previous q values for an error of 0
            # next_state_output = self.return_target_q(data_set[3])
            # max_q_index = np.argmax(next_state_output)
            # y = np.zeros(len(next_state_output))
            # for i in range(len(next_state_output)):
            #     if i == max_q_index:
            #         y[i] = data_set[2] + self.discount * next_state_output[i]
            #     else:
            #         y[i] = z_values[-1][i]

            # errors = [[] for _ in range(self.layers)]
            # delta_gradient_b = [np.zeros(b.shape) for b in self.biases]
            # delta_gradient_w = [np.zeros(w.shape) for w in self.weights]

            # #finds the error in the output layer
            # errors[-1] = (z_values[-1] - y) * self.relu_prime(z_values[-1])


            # #finds the errors of the other layers by calculating the loss
            # l = self.layers - 2
            # while(l > 0):
            #     errors[l] = (self.weights[l+1].T.dot(errors[l+1])) * self.relu_prime(z_values[l])
            #     l -= 1

            # errors_for_batch.append(errors)
            # activations_for_batch.append(a_values)
        
        #updates the weights and biases 
        # l = self.layers-1
        # while(l > 0):
        #     sum_weights = 0
        #     sum_biases = 0
        #     for x, training_set in enumerate(batch):
        #         sum_weights += errors_for_batch[x][l].reshape(errors_for_batch[x][l].shape[0],1).dot((activations_for_batch[x][l-1].reshape(1, activations_for_batch[x][l-1].shape[0])))
        #         sum_biases += errors_for_batch[x][l]
        #     self.weights[l] = self.weights[l] - self.lr * sum_weights / len(batch)
        #     self.biases[l] = self.biases[l] - self.lr * sum_biases / len(batch)
        #     l -= 1

    def backprop(self, x):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        # state = x[0]
        # #sets the z and activation values of the first layer for the policy network
        # activation = state
        # a_values = [state]
        # z_values = []

        # #computes the weighted sum with a bias of the previous later for each node in the policy and target networks
        # for l in range(1, self.layers):
        #     z_policy = self.weights[l].dot(activation) + self.biases[l]

        #     z_values.append(z_policy)
        #     activation = np.maximum(z_policy, np.zeros(z_policy.shape))
        #     a_values.append(activation)
        a_values, z_values = self.feedforward(x[0])

        #determins the training labels by passing the next state to the target network,
        #determining the action with the max q value and setting that label to the bellman
        #equation and all others to their previous q values for an error of 0
        next_state_output = self.return_target_q(x[3])
        max_q_index = np.argmax(next_state_output)
        y = np.zeros(len(next_state_output))
        for i in range(len(next_state_output)):
            if i == max_q_index:
                y[i] = x[2] + self.discount * next_state_output[i]
            else:
                y[i] = z_values[-1][i]
        # if x[4]:
        #     y = -1
        # else:
        #     y = x[2] + self.discount * np.amax(next_state_output)

        # y = x[2] + self.discount * np.amax(next_state_output)
        #backward pass
        error = (z_values[-1] - y) * self.relu_prime(z_values[-1])

        gradient_b[-1] = error
        gradient_w[-1] = np.dot(error.reshape(error.shape[0], 1), a_values[-2].reshape(1, a_values[-2].shape[0]))

        l = self.layers - 3
        while l >= 0:
            error = (self.weights[l+1].T.dot(error)) * self.relu_prime(z_values[l])
            gradient_b[l] = error
            gradient_w[l] = np.dot(error.reshape(error.shape[0], 1), a_values[l].reshape(1, a_values[l].shape[0]))
            l-=1

        return gradient_b, gradient_w
        

    def relu_prime(self, z):
        """
        Applies the derivative of relu to every element in an array
        """
        deriv = np.zeros(len(z))
        for i, d in enumerate(z):
            if d > 0:
                deriv[i] = 1
            else:
                deriv[i] = 0
        return deriv

    def feedforward(self, state, network="policy"):
        if network == "policy":
            w = self.weights
            b = self.biases
        else:
            w = self.t_weights
            b = self.t_biases
        activation = state
        a_values = [state]
        z_values = []

        #computes the weighted sum with a bias of the previous later for each node in the policy and target networks
        for l in range(self.layers-1):
            z_policy = w[l].dot(activation) + b[l]

            z_values.append(z_policy)
            activation = np.maximum(z_policy, np.zeros(z_policy.shape))
            a_values.append(activation)

        return a_values, z_values

    def return_max_q(self, state):
        """
        Runs the state through the network and returns the output
        Parameters:
        state(np array) -- the current state

        Returns:
        output(np array) -- the q values of the various actions
        """
        a_values, z_values = self.feedforward(state)

        return np.argmax(z_values[-1])

    def return_target_q(self, state):
        a_values, z_values = self.feedforward(state, "target")

        return np.array(z_values[-1])

                
    