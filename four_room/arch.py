from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
import torch.nn as nn
import numpy as np
import gymnasium as gym
import math

def orthogonal_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer

def kaiming_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    if layer.bias is not None:
        fan_in, _ = th.nn.init._calculate_fan_in_and_fan_out(layer.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            th.nn.init.uniform_(layer.bias, -bound, bound)
    return layer

class ResidualBlock(nn.Module):
    def __init__(self, channels, init_function='orthogonal'):
        super().__init__()
        if init_function == 'orthogonal':
            self.conv0 = orthogonal_layer_init(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1))
            self.conv1 = orthogonal_layer_init(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1))
        elif init_function == 'kaiming':
            self.conv0 = kaiming_layer_init(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1))
            self.conv1 = kaiming_layer_init(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1))
    
    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, max_pool=True, init_function='orthogonal'):
        super().__init__()
        self.max_pool = max_pool
        self._input_shape = input_shape
        self._out_channels = out_channels
        if init_function == 'orthogonal':
            self.conv = orthogonal_layer_init(nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1))
        elif init_function == 'kaiming':
            self.conv = kaiming_layer_init(nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1))

        self.res_block0 = ResidualBlock(self._out_channels, init_function=init_function)
        self.res_block1 = ResidualBlock(self._out_channels, init_function=init_function)

    def forward(self, x):
        x = self.conv(x)
        if self.max_pool:
            x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape(), f"{x.shape[1:]} != {self.get_output_shape()}"
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        if self.max_pool:
            return (self._out_channels, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self._out_channels, h, w)

class CNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64, load_file = None, freeze_linear = False, init_function='orthogonal'):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.image_normaliser = 10
        
        # conv_seq1 = ConvSequence(observation_space.shape, 16, max_pool=True)
        # conv_seq2 = ConvSequence(conv_seq1.get_output_shape(), 32, max_pool=False)
        # self.cnn = nn.Sequential(conv_seq1, conv_seq2)
        self.cnn = ConvSequence(observation_space.shape, 64, max_pool=True, init_function=init_function)


        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = np.prod(self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1:])

        if init_function == 'orthogonal':
            self.linear = nn.Sequential(orthogonal_layer_init(nn.Linear(n_flatten, features_dim)), nn.ReLU())
        elif init_function == 'kaiming':
            self.linear = nn.Sequential(kaiming_layer_init(nn.Linear(n_flatten, features_dim)), nn.ReLU())


    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations / self.image_normaliser
        x = self.cnn(observations)
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(x)
        return self.linear(x)
