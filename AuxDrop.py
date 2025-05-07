# Libraries required
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim


# Optimized AuxDrop with ODL (Online Dense Learning)
class AuxDrop_ODL(nn.Module):
    def __init__(
        self,
        features_size,
        max_num_hidden_layers,
        qtd_neuron_per_hidden_layer,
        n_classes,
        aux_layer,
        n_neuron_aux_layer,
        batch_size=1,
        b=0.99,
        n=0.01,
        s=0.2,
        dropout_p=0.5,
        n_aux_feat=3,
        use_cuda=False,
    ):
        super(AuxDrop_ODL, self).__init__()

        # Set up device configuration
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        # Store model configuration
        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.batch_size = batch_size
        self.n_aux_feat = n_aux_feat

        # Initialize training parameters
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(dropout_p), requires_grad=False).to(self.device)

        # Create network layers
        hidden_layers = []
        # First layer: input features → hidden neurons
        hidden_layers.append(nn.Linear(features_size, qtd_neuron_per_hidden_layer))

        # Configure remaining hidden layers with aux layer handling
        for i in range(max_num_hidden_layers - 1):
            if i + 2 == aux_layer:  # Layer before aux layer
                hidden_layers.append(
                    nn.Linear(
                        n_aux_feat + qtd_neuron_per_hidden_layer, n_neuron_aux_layer
                    )
                )
            elif i + 1 == aux_layer:  # Aux layer
                hidden_layers.append(
                    nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer)
                )
            else:  # Regular hidden layer
                hidden_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer)
                )

        # Configure output layers (one per layer except first and aux)
        output_layers = []
        for _ in range(max_num_hidden_layers - 2):
            output_layers.append(nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        # Register layers as module lists
        self.hidden_layers = nn.ModuleList(hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(output_layers).to(self.device)

        # Initialize alpha values with equal weights
        self.alpha = Parameter(
            torch.ones(self.max_num_hidden_layers - 2)
            / (self.max_num_hidden_layers - 2),
            requires_grad=False,
        ).to(self.device)

        # Initialize tracking variables
        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

    def zero_grad(self):
        """Reset all gradients to zero"""
        for i in range(self.max_num_hidden_layers - 2):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def update_weights(self, X, aux_feat, aux_mask, Y, show_loss):
        """Update network weights based on predictions and targets"""
        Y = torch.from_numpy(Y).to(self.device)

        # Get predictions from forward pass
        predictions_per_layer = self.forward(X, aux_feat, aux_mask)

        # Calculate weighted ensemble prediction
        real_output = torch.sum(
            torch.mul(
                self.alpha.view(self.max_num_hidden_layers - 2, 1, 1).expand(
                    -1, self.batch_size, self.n_classes
                ),
                predictions_per_layer,
            ),
            dim=0,
        )
        self.prediction.append(real_output)

        # Calculate loss
        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(
            real_output.view(self.batch_size, self.n_classes),
            Y.view(self.batch_size).long(),
        )
        self.loss_array.append(loss.detach().numpy())

        # Display loss if requested
        if show_loss and len(self.loss_array) % 1000 == 0:
            print("WARNING: Set 'show_loss' to 'False' when not debugging.")
            loss_mean = np.mean(self.loss_array[-1000:])
            print(f"Alpha: {self.alpha.data.cpu().numpy()}")
            print(f"Training Loss: {loss_mean}")

        # Calculate per-layer losses
        losses_per_layer = [
            criterion(
                out.view(self.batch_size, self.n_classes),
                Y.view(self.batch_size).long(),
            )
            for out in predictions_per_layer
        ]

        # Prepare weight update arrays
        w = [None] * (len(losses_per_layer) + 2)
        b = [None] * (len(losses_per_layer) + 2)

        with torch.no_grad():
            # Process each layer's loss
            for i in range(len(losses_per_layer)):
                losses_per_layer[i].backward(retain_graph=True)

                # Update output layer weights directly
                self.output_layers[i].weight.data -= (
                    self.n * self.alpha[i] * self.output_layers[i].weight.grad.data
                )
                self.output_layers[i].bias.data -= (
                    self.n * self.alpha[i] * self.output_layers[i].bias.grad.data
                )

                # Accumulate gradients for hidden layers with special aux_layer handling
                if i < self.aux_layer - 2:
                    for j in range(i + 2):
                        if w[j] is None:
                            w[j] = (
                                self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            )
                            b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                        else:
                            w[j] += (
                                self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            )
                            b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                if i > self.aux_layer - 3:
                    for j in range(i + 3):
                        if w[j] is None:
                            w[j] = (
                                self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            )
                            b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                        else:
                            w[j] += (
                                self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            )
                            b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad()

            # Update hidden layer weights
            for i in range(self.max_num_hidden_layers):
                if w[i] is not None:
                    self.hidden_layers[i].weight.data -= self.n * w[i]
                    self.hidden_layers[i].bias.data -= self.n * b[i]

            # Update alpha values based on layer performance
            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(self.alpha[i], self.s / len(losses_per_layer))

            # Normalize alpha values to sum to 1
            self.alpha.data = self.alpha.data / torch.sum(self.alpha.data)

        # Track losses and alphas
        self.layerwise_loss_array.append(
            np.array([loss.detach().numpy() for loss in losses_per_layer])
        )
        self.alpha_array.append(self.alpha.detach().numpy())

    def forward(self, X, aux_feat, aux_mask):
        """Forward pass through the network with directed dropout in aux layer"""
        # Convert inputs to tensors
        X = torch.from_numpy(X).float().to(self.device)
        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Process layers with activations
        hidden_connections = []

        # First layer
        hidden_connections.append(F.relu(self.hidden_layers[0](X)))

        # Process remaining layers
        for i in range(1, self.max_num_hidden_layers):
            if i == self.aux_layer - 1:
                # Aux layer processing with auxiliary features
                combined = torch.cat((aux_feat, hidden_connections[i - 1]), dim=1)
                layer_output = F.relu(self.hidden_layers[i](combined))

                # Calculate dropout mask based on auxiliary inputs
                aux_p = (
                    self.p * self.n_neuron_aux_layer
                    - (aux_mask.size()[1] - torch.sum(aux_mask))
                ) / (self.n_neuron_aux_layer - aux_mask.size()[1])
                binomial = torch.distributions.binomial.Binomial(probs=1 - aux_p)
                non_aux_mask = binomial.sample(
                    [1, self.n_neuron_aux_layer - aux_mask.size()[1]]
                )
                mask = torch.cat((aux_mask, non_aux_mask), dim=1)

                # Apply dropout with scaling factor
                hidden_connections.append(layer_output * mask * (1.0 / (1 - self.p)))
            else:
                # Regular layer processing
                hidden_connections.append(
                    F.relu(self.hidden_layers[i](hidden_connections[i - 1]))
                )

        # Get outputs from each layer (skipping aux layer)
        output_class = []
        for i in range(self.max_num_hidden_layers - 1):
            if i < self.aux_layer - 2:
                output_class.append(
                    F.softmax(self.output_layers[i](hidden_connections[i + 1]), dim=1)
                )
            if i > self.aux_layer - 2:
                output_class.append(
                    F.softmax(
                        self.output_layers[i - 1](hidden_connections[i + 1]), dim=1
                    )
                )

        return torch.stack(output_class)

    def validate_input_X(self, data):
        """Validate X input dimensions"""
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions."
            )

    def validate_input_Y(self, data):
        """Validate Y input dimensions"""
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions."
            )

    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        """Fit the model on a single batch of data"""
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, aux_data, aux_mask, Y_data, show_loss)


# Optimized AuxDrop where the AuxLayer is the first layer
class AuxDrop_ODL_AuxLayer1stlayer(nn.Module):
    def __init__(
        self,
        features_size,
        max_num_hidden_layers,
        qtd_neuron_per_hidden_layer,
        n_classes,
        aux_layer,
        n_neuron_aux_layer,
        batch_size=1,
        b=0.99,
        n=0.01,
        s=0.2,
        dropout_p=0.5,
        n_aux_feat=3,
        use_cuda=False,
    ):
        super(AuxDrop_ODL_AuxLayer1stlayer, self).__init__()

        # Set up device configuration
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        # Store model configuration
        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.batch_size = batch_size
        self.n_aux_feat = n_aux_feat

        # Initialize training parameters
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(dropout_p), requires_grad=False).to(self.device)

        # Create network layers
        hidden_layers = []
        # First layer combines input features with auxiliary features
        hidden_layers.append(nn.Linear(features_size + n_aux_feat, n_neuron_aux_layer))
        # Second layer converts from aux layer to standard size
        hidden_layers.append(nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer))
        # Remaining hidden layers
        for _ in range(max_num_hidden_layers - 2):
            hidden_layers.append(
                nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer)
            )

        # Output layers - one per hidden layer except first/aux
        output_layers = []
        for _ in range(max_num_hidden_layers - 1):
            output_layers.append(nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        # Register layers as module lists
        self.hidden_layers = nn.ModuleList(hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(output_layers).to(self.device)

        # Initialize alpha values with equal weights
        self.alpha = Parameter(
            torch.ones(self.max_num_hidden_layers - 1)
            / (self.max_num_hidden_layers - 1),
            requires_grad=False,
        ).to(self.device)

        # Initialize tracking variables
        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

    def zero_grad(self):
        """Reset all gradients to zero"""
        for i in range(self.max_num_hidden_layers - 1):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def update_weights(self, X, aux_feat, aux_mask, Y, show_loss):
        """Update network weights based on predictions and targets"""
        Y = torch.from_numpy(Y).to(self.device)

        # Get predictions from forward pass
        predictions_per_layer = self.forward(X, aux_feat, aux_mask)

        # Calculate weighted ensemble prediction
        real_output = torch.sum(
            torch.mul(
                self.alpha.view(self.max_num_hidden_layers - 1, 1, 1).expand(
                    -1, self.batch_size, self.n_classes
                ),
                predictions_per_layer,
            ),
            dim=0,
        )
        self.prediction.append(real_output)

        # Calculate loss
        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(
            real_output.view(self.batch_size, self.n_classes),
            Y.view(self.batch_size).long(),
        )
        self.loss_array.append(loss.detach().numpy())

        # Display loss if requested
        if show_loss and len(self.loss_array) % 1000 == 0:
            print("WARNING: Set 'show_loss' to 'False' when not debugging.")
            loss_mean = np.mean(self.loss_array[-1000:])
            print(f"Alpha: {self.alpha.data.cpu().numpy()}")
            print(f"Training Loss: {loss_mean}")

        # Calculate per-layer losses
        losses_per_layer = [
            criterion(
                out.view(self.batch_size, self.n_classes),
                Y.view(self.batch_size).long(),
            )
            for out in predictions_per_layer
        ]

        # Prepare weight update arrays
        w = [None] * (len(losses_per_layer) + 1)
        b = [None] * (len(losses_per_layer) + 1)

        with torch.no_grad():
            # Process each layer's loss
            for i in range(len(losses_per_layer)):
                losses_per_layer[i].backward(retain_graph=True)

                # Update output layer weights directly
                self.output_layers[i].weight.data -= (
                    self.n * self.alpha[i] * self.output_layers[i].weight.grad.data
                )
                self.output_layers[i].bias.data -= (
                    self.n * self.alpha[i] * self.output_layers[i].bias.grad.data
                )

                # Accumulate gradients for hidden layers
                for j in range(i + 2):
                    if w[j] is None:
                        w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                    else:
                        w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad()

            # Update hidden layer weights
            for i in range(self.max_num_hidden_layers):
                if w[i] is not None:
                    self.hidden_layers[i].weight.data -= self.n * w[i]
                    self.hidden_layers[i].bias.data -= self.n * b[i]

            # Update alpha values based on layer performance
            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(self.alpha[i], self.s / len(losses_per_layer))

            # Normalize alpha values to sum to 1
            self.alpha.data = self.alpha.data / torch.sum(self.alpha.data)

        # Track losses and alphas
        self.layerwise_loss_array.append(
            np.array([loss.detach().numpy() for loss in losses_per_layer])
        )
        self.alpha_array.append(self.alpha.detach().numpy())

    def forward(self, X, aux_feat, aux_mask):
        """Forward pass through the network with aux layer as first layer"""
        # Convert inputs to tensors
        X = torch.from_numpy(X).float().to(self.device)
        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Process layers with activations
        hidden_connections = []

        # First layer (aux layer) - combine input and auxiliary features
        combined = torch.cat((aux_feat, X), dim=1)
        layer_output = F.relu(self.hidden_layers[0](combined))

        # Apply dropout to aux layer
        aux_p = (
            self.p * self.n_neuron_aux_layer
            - (aux_mask.size()[1] - torch.sum(aux_mask))
        ) / (self.n_neuron_aux_layer - aux_mask.size()[1])
        binomial = torch.distributions.binomial.Binomial(probs=1 - aux_p)
        non_aux_mask = binomial.sample(
            [1, self.n_neuron_aux_layer - aux_mask.size()[1]]
        )
        mask = torch.cat((aux_mask, non_aux_mask), dim=1)
        hidden_connections.append(layer_output * mask * (1.0 / (1 - self.p)))

        # Process remaining layers
        for i in range(1, self.max_num_hidden_layers):
            hidden_connections.append(
                F.relu(self.hidden_layers[i](hidden_connections[i - 1]))
            )

        # Get outputs from each layer
        output_class = [
            F.softmax(self.output_layers[i](hidden_connections[i + 1]), dim=1)
            for i in range(self.max_num_hidden_layers - 1)
        ]

        return torch.stack(output_class)

    def validate_input_X(self, data):
        """Validate X input dimensions"""
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions."
            )

    def validate_input_Y(self, data):
        """Validate Y input dimensions"""
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions."
            )

    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        """Fit the model on a single batch of data"""
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, aux_data, aux_mask, Y_data, show_loss)


# Optimized AuxDrop with OGD (Online Gradient Descent)
class AuxDrop_OGD(nn.Module):
    def __init__(
        self,
        features_size,
        max_num_hidden_layers=5,
        qtd_neuron_per_hidden_layer=100,
        n_classes=2,
        aux_layer=3,
        n_neuron_aux_layer=100,
        batch_size=1,
        n_aux_feat=3,
        n=0.01,
        dropout_p=0.5,
    ):
        super(AuxDrop_OGD, self).__init__()

        # Store model configuration
        self.features_size = features_size
        self.max_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.n_aux_feat = n_aux_feat
        self.n_classes = n_classes
        self.p = dropout_p
        self.batch_size = batch_size
        self.n = n

        # Create network layers
        hidden_layers = []
        # First layer: input features → hidden neurons
        hidden_layers.append(
            nn.Linear(features_size, qtd_neuron_per_hidden_layer, bias=True)
        )

        # Configure remaining layers with aux layer handling
        for i in range(max_num_hidden_layers - 1):
            if i + 2 == aux_layer:  # Layer before aux layer
                hidden_layers.append(
                    nn.Linear(
                        n_aux_feat + qtd_neuron_per_hidden_layer,
                        n_neuron_aux_layer,
                        bias=True,
                    )
                )
            elif i + 1 == aux_layer:  # Aux layer
                hidden_layers.append(
                    nn.Linear(
                        n_neuron_aux_layer, qtd_neuron_per_hidden_layer, bias=True
                    )
                )
            else:  # Regular hidden layer
                hidden_layers.append(
                    nn.Linear(
                        qtd_neuron_per_hidden_layer,
                        qtd_neuron_per_hidden_layer,
                        bias=True,
                    )
                )

        self.hidden_layers = nn.ModuleList(hidden_layers)

        # Output layer takes final hidden layer output
        self.output_layer = nn.Linear(qtd_neuron_per_hidden_layer, n_classes, bias=True)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Tracking variables
        self.prediction = []
        self.loss_array = []

    def forward(self, X, aux_feat, aux_mask):
        """Forward pass through the network"""
        # Convert inputs to tensors
        X = torch.from_numpy(X).float()
        aux_feat = torch.from_numpy(aux_feat).float()
        aux_mask = torch.from_numpy(aux_mask).float()

        # Process first layer
        inp = F.relu(self.hidden_layers[0](X))

        # Process remaining layers
        for i in range(1, self.max_layers):
            if i == self.aux_layer - 1:
                # Aux layer processing with auxiliary features
                combined = torch.cat((aux_feat, inp), dim=1)
                inp = F.relu(self.hidden_layers[i](combined))

                # Apply dropout in aux layer
                aux_p = (
                    self.p * self.n_neuron_aux_layer
                    - (aux_mask.size()[1] - torch.sum(aux_mask))
                ) / (self.n_neuron_aux_layer - aux_mask.size()[1])
                binomial = torch.distributions.binomial.Binomial(probs=1 - aux_p)
                non_aux_mask = binomial.sample(
                    [1, self.n_neuron_aux_layer - aux_mask.size()[1]]
                )
                mask = torch.cat((aux_mask, non_aux_mask), dim=1)
                inp = inp * mask * (1.0 / (1 - self.p))
            else:
                # Regular layer processing
                inp = F.relu(self.hidden_layers[i](inp))

        # Final output with softmax
        out = F.softmax(self.output_layer(inp), dim=1)

        return out

    def validate_input_X(self, data):
        """Validate X input dimensions"""
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions."
            )

    def validate_input_Y(self, data):
        """Validate Y input dimensions"""
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions."
            )

    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        """Fit the model on a single batch of data using SGD"""
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)

        # Configure optimizer with learning rate
        optimizer = optim.SGD(self.parameters(), lr=self.n)
        optimizer.zero_grad()

        # Forward pass
        y_pred = self.forward(X_data, aux_data, aux_mask)
        self.prediction.append(y_pred)

        # Calculate loss
        loss = self.loss_fn(y_pred, torch.tensor(Y_data))
        self.loss_array.append(loss.item())

        # Backward pass and update
        loss.backward()
        optimizer.step()

        if show_loss:
            print("Loss is: ", loss.item())


# Optimized AuxDrop with directed dropout in aux layer and random in other layers
class AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer(nn.Module):
    def __init__(
        self,
        features_size,
        max_num_hidden_layers,
        qtd_neuron_per_hidden_layer,
        n_classes,
        aux_layer,
        n_neuron_aux_layer,
        batch_size=1,
        b=0.99,
        n=0.01,
        s=0.2,
        dropout_p=0.5,
        n_aux_feat=3,
        use_cuda=False,
    ):
        super(AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer, self).__init__()

        # Set up device configuration
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        # Store model configuration
        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.batch_size = batch_size
        self.n_aux_feat = n_aux_feat

        # Initialize training parameters
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(dropout_p), requires_grad=False).to(self.device)

        # Create network layers
        hidden_layers = []
        # First layer: input features → hidden neurons
        hidden_layers.append(nn.Linear(features_size, qtd_neuron_per_hidden_layer))

        # Configure remaining layers with aux layer handling
        for i in range(max_num_hidden_layers - 1):
            if i + 2 == aux_layer:  # Layer before aux layer
                hidden_layers.append(
                    nn.Linear(
                        n_aux_feat + qtd_neuron_per_hidden_layer, n_neuron_aux_layer
                    )
                )
            elif i + 1 == aux_layer:  # Aux layer
                hidden_layers.append(
                    nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer)
                )
            else:  # Regular hidden layer
                hidden_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer)
                )

        # Output layers
        output_layers = []
        for _ in range(max_num_hidden_layers - 2):
            output_layers.append(nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        # Register layers as module lists
        self.hidden_layers = nn.ModuleList(hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(output_layers).to(self.device)

        # Initialize alpha values with equal weights
        self.alpha = Parameter(
            torch.ones(self.max_num_hidden_layers - 2)
            / (self.max_num_hidden_layers - 2),
            requires_grad=False,
        ).to(self.device)

        # Initialize tracking variables
        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

    def zero_grad(self):
        """Reset all gradients to zero"""
        for i in range(self.max_num_hidden_layers - 2):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def forward(self, X, aux_feat, aux_mask, training):
        """Forward pass with directed dropout in aux layer and random dropout elsewhere"""
        # Convert inputs to tensors
        X = torch.from_numpy(X).float().to(self.device)
        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Process layers with activations
        hidden_connections = []

        # First layer
        first_output = F.relu(self.hidden_layers[0](X))
        hidden_connections.append(first_output)

        # Process remaining layers with different dropout strategies
        for i in range(1, self.max_num_hidden_layers):
            if i == self.aux_layer - 1:
                # Aux layer with directed dropout
                combined = torch.cat((aux_feat, hidden_connections[i - 1]), dim=1)
                layer_output = F.relu(self.hidden_layers[i](combined))

                # Apply directed dropout in aux layer
                aux_p = (
                    self.p * self.n_neuron_aux_layer
                    - (aux_mask.size()[1] - torch.sum(aux_mask))
                ) / (self.n_neuron_aux_layer - aux_mask.size()[1])
                binomial = torch.distributions.binomial.Binomial(probs=1 - aux_p)
                non_aux_mask = binomial.sample(
                    [1, self.n_neuron_aux_layer - aux_mask.size()[1]]
                )
                mask = torch.cat((aux_mask, non_aux_mask), dim=1)
                hidden_connections.append(layer_output * mask * (1.0 / (1 - self.p)))
            else:
                # Regular layer with random dropout
                layer_output = F.relu(self.hidden_layers[i](hidden_connections[i - 1]))
                binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
                mask = binomial.sample([1, self.qtd_neuron_per_hidden_layer])
                hidden_connections.append(layer_output * mask * (1.0 / (1 - self.p)))

        # Get outputs from each layer (skipping aux layer)
        output_class = []
        for i in range(self.max_num_hidden_layers - 1):
            if i < self.aux_layer - 2:
                output_class.append(
                    F.softmax(self.output_layers[i](hidden_connections[i + 1]), dim=1)
                )
            if i > self.aux_layer - 2:
                output_class.append(
                    F.softmax(
                        self.output_layers[i - 1](hidden_connections[i + 1]), dim=1
                    )
                )

        return torch.stack(output_class)

    def update_weights(self, X, aux_feat, aux_mask, Y, show_loss):
        """Update network weights based on predictions and targets"""
        Y = torch.from_numpy(Y).to(self.device)

        # Get predictions in training mode (applying dropout)
        predictions_per_layer = self.forward(X, aux_feat, aux_mask, training=True)

        # Calculate weighted ensemble prediction
        real_output = torch.sum(
            torch.mul(
                self.alpha.view(self.max_num_hidden_layers - 2, 1, 1).expand(
                    -1, self.batch_size, self.n_classes
                ),
                predictions_per_layer,
            ),
            dim=0,
        )
        self.prediction.append(real_output)

        # Calculate loss
        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(
            real_output.view(self.batch_size, self.n_classes),
            Y.view(self.batch_size).long(),
        )
        self.loss_array.append(loss.detach().numpy())

        # Display loss if requested
        if show_loss and len(self.loss_array) % 1000 == 0:
            print("WARNING: Set 'show_loss' to 'False' when not debugging.")
            loss_mean = np.mean(self.loss_array[-1000:])
            print(f"Alpha: {self.alpha.data.cpu().numpy()}")
            print(f"Training Loss: {loss_mean}")

        # Calculate per-layer losses
        losses_per_layer = [
            criterion(
                out.view(self.batch_size, self.n_classes),
                Y.view(self.batch_size).long(),
            )
            for out in predictions_per_layer
        ]

        # Prepare weight update arrays
        w = [None] * (len(losses_per_layer) + 2)
        b = [None] * (len(losses_per_layer) + 2)

        with torch.no_grad():
            # Apply weight updates with special aux_layer handling
            for i in range(len(losses_per_layer)):
                losses_per_layer[i].backward(retain_graph=True)

                # Update output layer weights directly
                self.output_layers[i].weight.data -= (
                    self.n * self.alpha[i] * self.output_layers[i].weight.grad.data
                )
                self.output_layers[i].bias.data -= (
                    self.n * self.alpha[i] * self.output_layers[i].bias.grad.data
                )

                # Accumulate gradients for hidden layers
                if i < self.aux_layer - 2:
                    for j in range(i + 2):
                        if w[j] is None:
                            w[j] = (
                                self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            )
                            b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                        else:
                            w[j] += (
                                self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            )
                            b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                if i > self.aux_layer - 3:
                    for j in range(i + 3):
                        if w[j] is None:
                            w[j] = (
                                self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            )
                            b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                        else:
                            w[j] += (
                                self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            )
                            b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad()

            # Update hidden layer weights
            for i in range(self.max_num_hidden_layers):
                if w[i] is not None:
                    self.hidden_layers[i].weight.data -= self.n * w[i]
                    self.hidden_layers[i].bias.data -= self.n * b[i]

            # Update alpha values based on layer performance
            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(self.alpha[i], self.s / len(losses_per_layer))

            # Normalize alpha values to sum to 1
            self.alpha.data = self.alpha.data / torch.sum(self.alpha.data)

        # Track losses and alphas
        self.layerwise_loss_array.append(
            np.array([loss.detach().numpy() for loss in losses_per_layer])
        )
        self.alpha_array.append(self.alpha.detach().numpy())

    def validate_input_X(self, data):
        """Validate X input dimensions"""
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions."
            )

    def validate_input_Y(self, data):
        """Validate Y input dimensions"""
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions."
            )

    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=True):
        """Fit the model on a single batch of data"""
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, aux_data, aux_mask, Y_data, show_loss)


# Optimized AuxDrop with random dropout in all layers
class AuxDrop_ODL_RandomAllLayer(nn.Module):
    def __init__(
        self,
        features_size,
        max_num_hidden_layers,
        qtd_neuron_per_hidden_layer,
        n_classes,
        aux_layer,
        n_neuron_aux_layer,
        batch_size=1,
        b=0.99,
        n=0.01,
        s=0.2,
        dropout_p=0.5,
        n_aux_feat=3,
        use_cuda=False,
    ):
        super(AuxDrop_ODL_RandomAllLayer, self).__init__()

        # Set up device configuration
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        # Store model configuration
        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.batch_size = batch_size
        self.n_aux_feat = n_aux_feat

        # Initialize training parameters
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(dropout_p), requires_grad=False).to(self.device)

        # Create network layers with the same pattern as DirectedInAuxLayer
        hidden_layers = []
        # First layer: input features → hidden neurons
        hidden_layers.append(nn.Linear(features_size, qtd_neuron_per_hidden_layer))

        # Configure remaining layers with aux layer handling
        for i in range(max_num_hidden_layers - 1):
            if i + 2 == aux_layer:  # Layer before aux layer
                hidden_layers.append(
                    nn.Linear(
                        n_aux_feat + qtd_neuron_per_hidden_layer, n_neuron_aux_layer
                    )
                )
            elif i + 1 == aux_layer:  # Aux layer
                hidden_layers.append(
                    nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer)
                )
            else:  # Regular hidden layer
                hidden_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer)
                )

        # Output layers
        output_layers = []
        for _ in range(max_num_hidden_layers - 2):
            output_layers.append(nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        # Register layers as module lists
        self.hidden_layers = nn.ModuleList(hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(output_layers).to(self.device)

        # Initialize alpha values with equal weights
        self.alpha = Parameter(
            torch.ones(self.max_num_hidden_layers - 2)
            / (self.max_num_hidden_layers - 2),
            requires_grad=False,
        ).to(self.device)

        # Initialize tracking variables
        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

    def zero_grad(self):
        """Reset all gradients to zero"""
        for i in range(self.max_num_hidden_layers - 2):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def forward(self, X, aux_feat, aux_mask, training):
        """Forward pass with random dropout in all layers"""
        # Convert inputs to tensors
        X = torch.from_numpy(X).float().to(self.device)
        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Process layers with activations
        hidden_connections = []

        # First layer
        first_output = F.relu(self.hidden_layers[0](X))
        binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
        mask = binomial.sample([1, self.qtd_neuron_per_hidden_layer])
        hidden_connections.append(first_output * mask * (1.0 / (1 - self.p)))

        # Process remaining layers with random dropout in all layers
        for i in range(1, self.max_num_hidden_layers):
            if i == self.aux_layer - 1:
                # Aux layer with random dropout
                combined = torch.cat((aux_feat, hidden_connections[i - 1]), dim=1)
                layer_output = F.relu(self.hidden_layers[i](combined))

                binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
                mask = binomial.sample([1, self.n_neuron_aux_layer])
                hidden_connections.append(layer_output * mask * (1.0 / (1 - self.p)))
            else:
                # Regular layer with random dropout
                layer_output = F.relu(self.hidden_layers[i](hidden_connections[i - 1]))
                binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
                mask = binomial.sample([1, self.qtd_neuron_per_hidden_layer])
                hidden_connections.append(layer_output * mask * (1.0 / (1 - self.p)))

        # Get outputs from each layer (skipping aux layer)
        output_class = []
        for i in range(self.max_num_hidden_layers - 1):
            if i < self.aux_layer - 2:
                output_class.append(
                    F.softmax(self.output_layers[i](hidden_connections[i + 1]), dim=1)
                )
            if i > self.aux_layer - 2:
                output_class.append(
                    F.softmax(
                        self.output_layers[i - 1](hidden_connections[i + 1]), dim=1
                    )
                )

        return torch.stack(output_class)

    # Use the same update_weights, validate_input_X/Y, and partial_fit methods as AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer
    update_weights = AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer.update_weights
    validate_input_X = AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer.validate_input_X
    validate_input_Y = AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer.validate_input_Y
    partial_fit = AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer.partial_fit


# Optimized AuxDrop with random dropout only in the aux layer
class AuxDrop_ODL_RandomInAuxLayer(nn.Module):
    def __init__(
        self,
        features_size,
        max_num_hidden_layers,
        qtd_neuron_per_hidden_layer,
        n_classes,
        aux_layer,
        n_neuron_aux_layer,
        batch_size=1,
        b=0.99,
        n=0.01,
        s=0.2,
        dropout_p=0.5,
        n_aux_feat=3,
        use_cuda=False,
    ):
        super(AuxDrop_ODL_RandomInAuxLayer, self).__init__()

        # Set up device configuration
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        # Store model configuration
        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.batch_size = batch_size
        self.n_aux_feat = n_aux_feat

        # Initialize training parameters
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(dropout_p), requires_grad=False).to(self.device)

        # Create network layers with the same pattern as previous classes
        hidden_layers = []
        # First layer: input features → hidden neurons
        hidden_layers.append(nn.Linear(features_size, qtd_neuron_per_hidden_layer))

        # Configure remaining layers with aux layer handling
        for i in range(max_num_hidden_layers - 1):
            if i + 2 == aux_layer:  # Layer before aux layer
                hidden_layers.append(
                    nn.Linear(
                        n_aux_feat + qtd_neuron_per_hidden_layer, n_neuron_aux_layer
                    )
                )
            elif i + 1 == aux_layer:  # Aux layer
                hidden_layers.append(
                    nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer)
                )
            else:  # Regular hidden layer
                hidden_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer)
                )

        # Output layers
        output_layers = []
        for _ in range(max_num_hidden_layers - 2):
            output_layers.append(nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        # Register layers as module lists
        self.hidden_layers = nn.ModuleList(hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(output_layers).to(self.device)

        # Initialize alpha values with equal weights
        self.alpha = Parameter(
            torch.ones(self.max_num_hidden_layers - 2)
            / (self.max_num_hidden_layers - 2),
            requires_grad=False,
        ).to(self.device)

        # Initialize tracking variables
        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

    def zero_grad(self):
        """Reset all gradients to zero"""
        for i in range(self.max_num_hidden_layers - 2):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def forward(self, X, aux_feat, aux_mask, training):
        """Forward pass with random dropout only in aux layer"""
        # Convert inputs to tensors
        X = torch.from_numpy(X).float().to(self.device)
        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Process layers with activations
        hidden_connections = []

        # First layer - no dropout
        hidden_connections.append(F.relu(self.hidden_layers[0](X)))

        # Process remaining layers - dropout only in aux layer
        for i in range(1, self.max_num_hidden_layers):
            if i == self.aux_layer - 1:
                # Aux layer with random dropout
                combined = torch.cat((aux_feat, hidden_connections[i - 1]), dim=1)
                layer_output = F.relu(self.hidden_layers[i](combined))

                binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
                mask = binomial.sample([1, self.n_neuron_aux_layer])
                hidden_connections.append(layer_output * mask * (1.0 / (1 - self.p)))
            else:
                # Regular layer - no dropout
                hidden_connections.append(
                    F.relu(self.hidden_layers[i](hidden_connections[i - 1]))
                )

        # Get outputs from each layer (skipping aux layer)
        output_class = []
        for i in range(self.max_num_hidden_layers - 1):
            if i < self.aux_layer - 2:
                output_class.append(
                    F.softmax(self.output_layers[i](hidden_connections[i + 1]), dim=1)
                )
            if i > self.aux_layer - 2:
                output_class.append(
                    F.softmax(
                        self.output_layers[i - 1](hidden_connections[i + 1]), dim=1
                    )
                )

        return torch.stack(output_class)

    # Use the same update_weights, validate_input_X/Y, and partial_fit methods as previous classes
    update_weights = AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer.update_weights
    validate_input_X = AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer.validate_input_X
    validate_input_Y = AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer.validate_input_Y
    partial_fit = AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer.partial_fit


# Optimized AuxDrop with random dropout in first layer and all features passed to it
class AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst(nn.Module):
    def __init__(
        self,
        features_size,
        max_num_hidden_layers,
        qtd_neuron_per_hidden_layer,
        n_classes,
        aux_layer,
        n_neuron_aux_layer,
        batch_size=1,
        b=0.99,
        n=0.01,
        s=0.2,
        dropout_p=0.5,
        n_aux_feat=3,
        use_cuda=False,
    ):
        super(AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst, self).__init__()

        # Set up device configuration
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        # Store model configuration
        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.batch_size = batch_size
        self.n_aux_feat = n_aux_feat

        # Initialize training parameters
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(dropout_p), requires_grad=False).to(self.device)

        # Create network layers with different architecture (first layer has input + aux features)
        hidden_layers = []
        # First layer combines input features with auxiliary features
        hidden_layers.append(nn.Linear(features_size + n_aux_feat, n_neuron_aux_layer))
        # Second layer converts from aux layer to standard size
        hidden_layers.append(nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer))
        # Remaining hidden layers
        for _ in range(max_num_hidden_layers - 2):
            hidden_layers.append(
                nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer)
            )

        # Output layers - one per hidden layer except first/aux
        output_layers = []
        for _ in range(max_num_hidden_layers - 1):
            output_layers.append(nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        # Register layers as module lists
        self.hidden_layers = nn.ModuleList(hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(output_layers).to(self.device)

        # Initialize alpha values with equal weights
        self.alpha = Parameter(
            torch.ones(self.max_num_hidden_layers - 1)
            / (self.max_num_hidden_layers - 1),
            requires_grad=False,
        ).to(self.device)

        # Initialize tracking variables
        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

    def zero_grad(self):
        """Reset all gradients to zero"""
        for i in range(self.max_num_hidden_layers - 1):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def forward(self, X, aux_feat, aux_mask):
        """Forward pass with random dropout in first layer and all features passed to it"""
        # Convert inputs to tensors
        X = torch.from_numpy(X).float().to(self.device)
        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Process layers with activations
        hidden_connections = []

        # First layer - combine input and auxiliary features with random dropout
        combined = torch.cat((aux_feat, X), dim=1)
        layer_output = F.relu(self.hidden_layers[0](combined))

        binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
        mask = binomial.sample([1, self.n_neuron_aux_layer])
        hidden_connections.append(layer_output * mask * (1.0 / (1 - self.p)))

        # Process remaining layers - no dropout
        for i in range(1, self.max_num_hidden_layers):
            hidden_connections.append(
                F.relu(self.hidden_layers[i](hidden_connections[i - 1]))
            )

        # Get outputs from each layer
        output_class = [
            F.softmax(self.output_layers[i](hidden_connections[i + 1]), dim=1)
            for i in range(self.max_num_hidden_layers - 1)
        ]

        return torch.stack(output_class)

    # The update_weights method needs to be adapted for the different alpha size
    def update_weights(self, X, aux_feat, aux_mask, Y, show_loss):
        """Update network weights based on predictions and targets"""
        Y = torch.from_numpy(Y).to(self.device)

        # Get predictions from forward pass
        predictions_per_layer = self.forward(X, aux_feat, aux_mask)

        # Calculate weighted ensemble prediction
        real_output = torch.sum(
            torch.mul(
                self.alpha.view(self.max_num_hidden_layers - 1, 1, 1).expand(
                    -1, self.batch_size, self.n_classes
                ),
                predictions_per_layer,
            ),
            dim=0,
        )
        self.prediction.append(real_output)

        # Calculate loss
        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(
            real_output.view(self.batch_size, self.n_classes),
            Y.view(self.batch_size).long(),
        )
        self.loss_array.append(loss.detach().numpy())

        # Display loss if requested
        if show_loss and len(self.loss_array) % 1000 == 0:
            print("WARNING: Set 'show_loss' to 'False' when not debugging.")
            loss_mean = np.mean(self.loss_array[-1000:])
            print(f"Alpha: {self.alpha.data.cpu().numpy()}")
            print(f"Training Loss: {loss_mean}")

        # Calculate per-layer losses
        losses_per_layer = [
            criterion(
                out.view(self.batch_size, self.n_classes),
                Y.view(self.batch_size).long(),
            )
            for out in predictions_per_layer
        ]

        # Prepare weight update arrays
        w = [None] * (len(losses_per_layer) + 1)
        b = [None] * (len(losses_per_layer) + 1)

        with torch.no_grad():
            # Process each layer's loss
            for i in range(len(losses_per_layer)):
                losses_per_layer[i].backward(retain_graph=True)

                # Update output layer weights directly
                self.output_layers[i].weight.data -= (
                    self.n * self.alpha[i] * self.output_layers[i].weight.grad.data
                )
                self.output_layers[i].bias.data -= (
                    self.n * self.alpha[i] * self.output_layers[i].bias.grad.data
                )

                # Accumulate gradients for hidden layers
                for j in range(i + 2):
                    if w[j] is None:
                        w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                    else:
                        w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad()

            # Update hidden layer weights
            for i in range(self.max_num_hidden_layers):
                if w[i] is not None:
                    self.hidden_layers[i].weight.data -= self.n * w[i]
                    self.hidden_layers[i].bias.data -= self.n * b[i]

            # Update alpha values based on layer performance
            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(self.alpha[i], self.s / len(losses_per_layer))

            # Normalize alpha values to sum to 1
            self.alpha.data = self.alpha.data / torch.sum(self.alpha.data)

        # Track losses and alphas
        self.layerwise_loss_array.append(
            np.array([loss.detach().numpy() for loss in losses_per_layer])
        )
        self.alpha_array.append(self.alpha.detach().numpy())

    # Use the same validate_input_X/Y, and partial_fit methods as previous classes
    validate_input_X = AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer.validate_input_X
    validate_input_Y = AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer.validate_input_Y
    partial_fit = AuxDrop_ODL_AuxLayer1stlayer.partial_fit
