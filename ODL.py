import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ODL(nn.Module):
    """
    Online Deep Learning (ODL) implementation with dynamically weighted ensemble of models
    """

    def __init__(
        self,
        features_size,
        max_num_hidden_layers,
        qtd_neuron_per_hidden_layer,
        n_classes,
        batch_size=1,
        b=0.99,
        n=0.01,
        s=0.2,
        use_cuda=False,
    ):
        super().__init__()  # Simplified parent class call

        # Setup device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        # Store parameters
        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.batch_size = batch_size

        # Convert hyperparameters to PyTorch parameters and move to device
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)

        # Initialize network architecture
        self._build_network()

        # Initialize tracking variables
        self.loss_array = []
        self.prediction = []

    def _build_network(self):
        """Build network architecture with hidden and output layers"""
        # Create lists for hidden and output layers
        hidden_layers = []
        output_layers = []

        # First hidden layer connects input features to hidden neurons
        hidden_layers.append(
            nn.Linear(self.features_size, self.qtd_neuron_per_hidden_layer)
        )

        # Additional hidden layers (hidden to hidden)
        for _ in range(self.max_num_hidden_layers - 1):
            hidden_layers.append(
                nn.Linear(
                    self.qtd_neuron_per_hidden_layer, self.qtd_neuron_per_hidden_layer
                )
            )

        # Create output layers for each hidden layer
        for _ in range(self.max_num_hidden_layers):
            output_layers.append(
                nn.Linear(self.qtd_neuron_per_hidden_layer, self.n_classes)
            )

        # Convert to ModuleLists and move to device
        self.hidden_layers = nn.ModuleList(hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(output_layers).to(self.device)

        # Initialize alpha weights (equal weighting initially)
        self.alpha = Parameter(
            torch.full((self.max_num_hidden_layers,), 1 / self.max_num_hidden_layers),
            requires_grad=False,
        ).to(self.device)

    def zero_grad(self):
        """Reset gradients for all network parameters"""
        for i in range(self.max_num_hidden_layers):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def forward(self, X):
        """Forward pass through the network"""
        # Convert input to PyTorch tensor
        X = torch.from_numpy(X).float().to(self.device)

        # Process through hidden layers with ReLU activation
        hidden_connections = []
        x = F.relu(self.hidden_layers[0](X))
        hidden_connections.append(x)

        for i in range(1, self.max_num_hidden_layers):
            hidden_connections.append(
                F.relu(self.hidden_layers[i](hidden_connections[i - 1]))
            )

        # Get output from each layer
        output_class = [
            self.output_layers[i](hidden_connections[i])
            for i in range(self.max_num_hidden_layers)
        ]

        # Stack outputs from all layers
        pred_per_layer = torch.stack(output_class)
        return pred_per_layer

    def update_weights(self, X, Y, show_loss):
        """Update network weights using ODL algorithm"""
        # Convert target to tensor
        Y = torch.from_numpy(Y).to(self.device)

        # Get predictions from all layers
        predictions_per_layer = self.forward(X)

        # Calculate weighted ensemble prediction
        alpha_expanded = (
            self.alpha.view(self.max_num_hidden_layers, 1)
            .repeat(1, self.batch_size)
            .view(self.max_num_hidden_layers, self.batch_size, 1)
        )
        real_output = torch.sum(torch.mul(alpha_expanded, predictions_per_layer), 0)

        # Store prediction for evaluation
        self.prediction.append(torch.argmax(real_output).item())

        # Calculate loss for final output
        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(
            real_output.view(self.batch_size, self.n_classes),
            Y.view(self.batch_size).long(),
        )
        self.loss_array.append(loss.detach().numpy())

        # Display loss if requested
        if show_loss and (len(self.loss_array) % 1000) == 0:
            print(
                "WARNING: Set 'show_loss' to 'False' when not debugging. "
                "It will deteriorate the fitting performance."
            )
            loss = np.mean(self.loss_array[-1000:])
            print(f"Alpha: {self.alpha.data.cpu().numpy()}")
            print(f"Training Loss: {loss}")

        # Calculate loss for each layer's output
        losses_per_layer = []
        for out in predictions_per_layer:
            layer_loss = criterion(
                out.view(self.batch_size, self.n_classes),
                Y.view(self.batch_size).long(),
            )
            losses_per_layer.append(layer_loss)

        # Prepare accumulators for hidden layer gradients
        w = [None] * len(losses_per_layer)
        b = [None] * len(losses_per_layer)

        with torch.no_grad():
            # Update each layer based on its contribution
            for i in range(len(losses_per_layer)):
                # Backpropagate for this layer
                losses_per_layer[i].backward(retain_graph=True)

                # Update output layer weights
                self.output_layers[i].weight.data -= (
                    self.n * self.alpha[i] * self.output_layers[i].weight.grad.data
                )
                self.output_layers[i].bias.data -= (
                    self.n * self.alpha[i] * self.output_layers[i].bias.grad.data
                )

                # Accumulate gradients for hidden layers
                for j in range(i + 1):
                    if w[j] is None:
                        w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                    else:
                        w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                # Reset gradients
                self.zero_grad()

            # Update hidden layers with accumulated gradients
            for i in range(len(losses_per_layer)):
                self.hidden_layers[i].weight.data -= self.n * w[i]
                self.hidden_layers[i].bias.data -= self.n * b[i]

            # Update alpha values based on layer performance
            for i in range(len(losses_per_layer)):
                # Decrease alpha for layers with higher loss
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                # Ensure minimum contribution from each layer
                self.alpha[i] = torch.max(
                    self.alpha[i], self.s / self.max_num_hidden_layers
                )

            # Normalize alpha values to sum to 1
            z_t = torch.sum(self.alpha)
            self.alpha = Parameter(self.alpha / z_t, requires_grad=False).to(
                self.device
            )

    def validate_input_X(self, data):
        """Validate input features dimension"""
        if len(data.shape) != 2:
            raise ValueError("X data should have exactly two dimensions")

    def validate_input_Y(self, data):
        """Validate target labels dimension"""
        if len(data.shape) != 1:
            raise ValueError("Y data should have exactly one dimension")

    def partial_fit(self, X_data, Y_data, show_loss=False):
        """Train model on one batch of data"""
        self.validate_input_X(X_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, Y_data, show_loss)
