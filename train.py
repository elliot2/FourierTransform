import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity

import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# Profiler fix
# C:\dev\anaconda3\envs\pytorch2b\Lib\site-packages\torch_tb_profiler\profiler\data.py
# @staticmethod
# def parse(worker, span, path, cache_dir):
#     trace_path, trace_json = RunProfileData._preprocess_file(path, cache_dir)

#     profile = RunProfileData.from_json(worker, span, trace_json.replace(b"\\", b"\\\\"))
#     profile.trace_file_path = trace_path
#     return profile
# Profile with reduced epochs and batch size
# epochs = 100
# batch_size = 32
use_profiling = False


# run_name = 'sample_size_X8'
# learning_rate = 1e-5
# epochs = 200000
# batch_size = 128
# sample_size = 512
# net_size = sample_size * 8

run_name = 'sample_size_X8_continue'
learning_rate = 1e-5
epochs = 200000
batch_size = 128
sample_size = 512
net_size = sample_size * 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Before device change in vector_transform
# Epoch: 99500, Forward Loss: 6.990011024754494e-05, Inverse Loss: 2.425016646157019e-05

# After device change in vector_transform
# Epoch: 99500, Forward Loss: 0.015215392224490643, Inverse Loss: 0.0025944700464606285

# 256 batch 256 sample size
# Epoch: 99500, Forward Loss: 0.005948793143033981, Inverse Loss: 0.00020423822570592165



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


# Define the loss function and the optimizers
criterion = torch.nn.MSELoss()
forward_optimizer = optim.Adam(forward_net.parameters(), lr=learning_rate)
inverse_optimizer = optim.Adam(inverse_net.parameters(), lr=learning_rate)

writer = SummaryWriter('./log/' + run_name)


prof = torch.profiler.profile(
         activities=[
         ProfilerActivity.CPU,
         ProfilerActivity.CUDA,
     ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/' + run_name),
        record_shapes=True,
        with_stack=True)

if (use_profiling) :
    prof.start()


for epoch in range(epochs):
    # Generate random input signal
    inputs_batch = torch.randn(batch_size, sample_size, device=device)


    # Calculate the true Fourier transform
    true_fourier = torch.fft.fft(inputs_batch)

    # Convert input signal to a PyTorch tensor
    # inputs = torch.tensor(inputs_batch, dtype=torch.float32, device=device).unsqueeze(-1)  # Add a channel dimension
    # targets_real = torch.tensor(true_fourier.real, dtype=torch.float32, device=device).unsqueeze(-1)
    # targets_imag = torch.tensor(true_fourier.imag, dtype=torch.float32, device=device).unsqueeze(-1)
 
    # Add a channel dimension
    inputs = inputs_batch.unsqueeze(-1)
    targets_real = true_fourier.real.unsqueeze(-1)
    targets_imag = true_fourier.imag.unsqueeze(-1)
    
    targets = torch.cat((targets_real, targets_imag), dim=-1)  # Concatenate real and imaginary parts along the channel dimension

    # print(f'inputs shape: {inputs.shape}')

    # Train the forward_net
    forward_outputs = forward_net(inputs)
    # print(f'Forward_outputs shape: {forward_outputs.shape}')
    # print(f'Targets shape: {targets.shape}')
    forward_loss = criterion(forward_outputs, targets)
    forward_optimizer.zero_grad()
    forward_loss.backward()
    forward_optimizer.step()

    # Make a prediction on the Fourier transformed output
    predicted_fourier = forward_outputs.detach()

    # Train the inverse_net
    inverse_outputs = inverse_net(predicted_fourier)
    inverse_loss = criterion(inverse_outputs, inputs)

    inverse_optimizer.zero_grad()
    inverse_loss.backward()
    inverse_optimizer.step()

    # Print progress
    if epoch % 500 == 0:
        print(f'Epoch: {epoch}, Forward Loss: {forward_loss.item()}, Inverse Loss: {inverse_loss.item()}')
        writer.add_scalar("Forward Loss:", forward_loss.item(), epoch)
        writer.add_scalar("Inverse Loss:", inverse_loss.item(), epoch)

        if epoch % 5000 == 0:
            for name, param in forward_net.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'forward_net/{name}_grad', param.grad, epoch)

            for name, param in inverse_net.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'inverse_net/{name}_grad', param.grad, epoch)


    # Update the profiler at the end of each iteration
    if (use_profiling) :
        prof.step()

if (use_profiling) :
    prof.stop()

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

# Save the forward_net model
# torch.save(forward_net.state_dict(), "forward_net_model.pth")

# Save the inverse_net model
# torch.save(inverse_net.state_dict(), "inverse_net_model.pth")
