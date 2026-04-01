#######################
#
# ACTIVATION FUNCTIONS
#
#######################


import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(1337)


class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))))


class DeepNeuralNetwork(nn.Module):

    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
                    nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
                    GELU()),
                    nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
                    GELU()),
                    nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
                    GELU()),
                    nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
                    GELU()),
                    nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
                    GELU())
                    ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape[-1] == layer_output.shape[-1]:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_example_dnn():

    x = torch.rand(2, 3, 768)
    ffn = DeepNeuralNetwork(layer_sizes=[768, 1024, 768, 1024, 768, 1024], use_shortcut=True)
    print(f"EXAMPLE DNN SHAPE: {ffn(x).shape}")


def plot_relu_vs_gelu():

    gelu, relu = GELU(), nn.ReLU()
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)
    plt.figure(figsize=(8, 3))

    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):

        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)
        plt.tight_layout()

        plt.show()


def print_gradients_example():
    layer_sizes = [3, 3, 3, 3, 3, 1]

    _print_gradients(DeepNeuralNetwork(layer_sizes, use_shortcut=False))
    _print_gradients(DeepNeuralNetwork(layer_sizes, use_shortcut=True))


def _print_gradients(model):

    x = torch.tensor([[1., 0., -1.]])

    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

