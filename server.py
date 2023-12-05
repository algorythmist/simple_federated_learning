import tensorflow as tf


class FederatedServer:
    """
    The central server coordinates training by aggregating model weights from clients.
    """

    def __init__(self, model, clients, batch_size = 1, client_epochs=5):
        """
        :param model: The model maintained by the server
        :param clients: An array of ClientNode objects
        :param client_epochs: the number of epochs to train on each client
        """
        self.model = model
        self.clients = clients
        self.client_epochs = client_epochs
        self.batch_size = batch_size

    def train(self, iterations=10, evaluate_fn=None):
        """
        Train the model for the given number of iterations.
        :param iterations: The number of iterations to train for.
        :param evaluate_fn: An optional function to evaluate the model after each iteration.
        :return:
        """
        for iteration in range(iterations):
            print(f"Server iteration {iteration}")
            local_weights = self.training_step()
            scaled_weights = self.__scale_weights(local_weights.values())
            average_weights = self.__aggregate_scaled_weights(scaled_weights)
            # update global model
            self.model.set_weights(average_weights)
            if evaluate_fn:
                evaluate_fn(self.model)

    def __scale_weights(self, local_weights):
        total_samples = float(sum([size for size, _ in local_weights]))
        return [self.__scale_model_weights(size / total_samples, weights) for size, weights in local_weights]

    @staticmethod
    def __scale_model_weights(scalar, weights):
        return [scalar * w for w in weights]

    #TODO: Simplify weighted average calculation
    @staticmethod
    def __aggregate_scaled_weights(scaled_weights):
        """
        Return the sum of the listed scaled weights.
        The is equivalent to scaled average of the weights
        """
        average_weights = []
        # get the average grad across all client gradients
        for weight_list_tuple in zip(*scaled_weights):
            layer_mean = tf.math.reduce_sum(weight_list_tuple, axis=0)
            average_weights.append(layer_mean)
        return average_weights

    def training_step(self):
        global_weights = self.model.get_weights()
        local_weights = {}
        for client in self.clients:
            print(f"Fitting local model for client {client.name}")
            local_weights[client.name] = client.train(global_weights,
                                                      batch_size = self.batch_size,
                                                      epochs=self.client_epochs)
        return local_weights
