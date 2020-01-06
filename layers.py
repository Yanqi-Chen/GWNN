"""GWNN layers."""

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class GWNNLayer(nn.Module):

	def __init__(self, node_num, in_channels, out_channels, wavelets, wavelets_inv):
		super(GWNNLayer, self).__init__()
		self.node_num = node_num
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.wavelets = wavelets
		self.wavelets_inv = wavelets_inv

		self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
		self.filter = nn.Parameter(torch.Tensor(self.node_num))

		init.uniform_(self.filter, 0.9, 1.1)
		init.xavier_uniform_(self.weight_matrix)

	def forward(self, features):
		transformed_features = torch.mm(features, self.weight_matrix)
		output = torch.mm(torch.mm(self.wavelets, torch.diag(self.filter)),
						  torch.mm(self.wavelets_inv, transformed_features))
		return output
