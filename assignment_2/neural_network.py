from tqdm import tqdm


class Network:
    """ Network Class"""
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.error = []

    def add_layer(self, layer):
        """ Add layer to the network. """
        self.layers.append(layer)

    def set_loss_function(self, loss: callable, loss_prime: callable):
        """ Set loss function to be used in backward propagation."""
        self.loss = loss
        self.loss_prime = loss_prime

    def fit(self, x: list[list], y: list[list], epochs, learning_rate):
        """ Train the network.
        Args:
        x (list): training sample features.
        y (list): training sample results.
        epochs (int): number of steps in learning process.
        learning_rate (float): size of each step.
        """
        # sample dimension first
        samples = len(x)

        # training epochs over training sample.
        for _ in tqdm(range(epochs)):
            self.error.append(0)

            for j in range(samples):
                # forward propagation
                output = x[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # append loss to actual error
                self.error[-1] += self.loss(y[j], output)

                # backward propagation
                error = self.loss_prime(y[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # sum(error) -> avg(error)
            self.error[-1] /= samples

    def predict(self, input_data: list):
        """ predict output for given <input_data> using forward propagation. """

        result = []
        for i, sample in enumerate(input_data):
            output = sample
            # process the sample.
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result
