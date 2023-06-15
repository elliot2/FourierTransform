import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

batch_size = 128
sample_size = 512
net_size = sample_size * 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def vector_transform(inputs, num_steps=19, tune=3.0, init_displacement=0.0, scaling_factor=1.0):
    batch_size, N, num_channels = inputs.shape
    dt = (inputs + tune) / num_steps

    y = torch.zeros_like(inputs, device=device) + init_displacement  # Initial displacement
    v = dt  # Initial velocity

    y = y.to(device)
    v = v.to(device)
    dt = dt.to(device)

    for _ in range(num_steps):
        y, v = y + v, v - y * dt

    return (y * scaling_factor).view(batch_size, -1, num_channels)


class CustomLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tune = nn.Parameter(torch.tensor(3.0, device=device))
        self.init_displacement = nn.Parameter(torch.tensor(0.0, device=device))
        self.scaling_factor = nn.Parameter(torch.tensor(1.0, device=device))

    def forward(self, x):
        x = vector_transform(x, tune=self.tune, init_displacement=self.init_displacement, scaling_factor=self.scaling_factor)
        return x



class Net(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.fc1_input_dim = sample_size * input_channels
        self.fc1 = nn.Linear(self.fc1_input_dim, net_size)
        self.act = nn.ReLU()
        self.fc1b = nn.Linear(net_size, net_size)
        self.custom = CustomLayer()
        self.fc2 = nn.Linear(net_size, sample_size * output_channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc1b(x)
        x = self.act(x)
        x = x.view(x.size(0), -1, self.input_channels)  # Reshape the tensor to match the custom layer input shape
        x = self.custom(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor before passing it to the linear layer
        x = self.fc2(x)
        x = x.view(x.size(0), -1, self.output_channels)  # Reshape the output tensor
        return x


# Create the neural networks
forward_net = Net(input_channels=1, output_channels=2).to(device)
inverse_net = Net(input_channels=2, output_channels=1).to(device)

# Load the saved weights into the new instances
forward_net.load_state_dict(torch.load("forward_net_model.pth"))
inverse_net.load_state_dict(torch.load("inverse_net_model.pth"))


# Test the neural network on the input signal
input_signal = np.random.rand(sample_size)
true_fourier = np.fft.fft(input_signal)
inputs = torch.tensor(input_signal, dtype=torch.float32, device=device).unsqueeze(0)

targets_real = torch.tensor(true_fourier.real, dtype=torch.float32).unsqueeze(0).to(device)
targets_imag = torch.tensor(true_fourier.imag, dtype=torch.float32).unsqueeze(0).to(device)
targets = torch.cat((targets_real, targets_imag), dim=1)

test_outputs = forward_net(inputs)

plt.plot(true_fourier.real, label='True Fourier')
plt.plot(true_fourier.imag, label='True Fourier')
plt.plot(test_outputs.detach().cpu().numpy().squeeze(), label='Predicted Fourier')
plt.legend()
plt.show()
