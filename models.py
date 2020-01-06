from layers import GWNNLayer
from torch import nn
import torch.nn.functional as F

class GraphWaveletNeuralNetwork(nn.Module):
	def __init__(self, node_cnt, feature_dims, hidden_dims, output_dims, wavelets, wavelets_inv, dropout_rate):
		super(GraphWaveletNeuralNetwork, self).__init__()
		self.node_cnt = node_cnt
		self.feature_dims = feature_dims
		self.hidden_dims = hidden_dims
		self.output_dims = output_dims
		self.wavelets = wavelets
		self.wavelets_inv = wavelets_inv
		self.dropout_rate = dropout_rate

		self.conv_1 = GWNNLayer(self.node_cnt, 
								self.feature_dims, 
								self.hidden_dims, 
								self.wavelets, 
								self.wavelets_inv)

		self.conv_2 = GWNNLayer(self.node_cnt, 
								self.hidden_dims, 
								self.output_dims, 
								self.wavelets, 
								self.wavelets_inv)

	def forward(self, input):
		output_1 = F.dropout(F.relu(self.conv_1(input)), 
							 training=self.training, 
							 p=self.dropout_rate)
		output_2 = self.conv_2(output_1)
		pred = F.log_softmax(output_2, dim=1)
		return pred
