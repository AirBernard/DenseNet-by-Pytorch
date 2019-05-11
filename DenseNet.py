import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class DenseNet(nn.Module):
	def __init__(self, cls_num, depth=100, growth_rate=12,
				use_bottleneck=True, compress_factor=0.5):
		super(DenseNet, self).__init__()
		num_dense_block = (depth-4) // 3
		if use_bottleneck:
			num_dense_block //= 2
		channels = 2 * growth_rate
		self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)

		# DenseBlock1
		self.dense1 = self.make_dense_block(channels, growth_rate, 
						num_dense_block, use_bottleneck)
		channels += num_dense_block * growth_rate
		outChannels = int(math.floor(channels * compress_factor))
		self.trans1 = Transition(channels, outChannels)

		# DenseBlock2
		channels = outChannels
		self.dense2 = self.make_dense_block(channels, growth_rate,
						num_dense_block, use_bottleneck)
		channels += num_dense_block * growth_rate
		outChannels = int(math.floor(channels * compress_factor))
		self.trans2 = Transition(channels, outChannels)

		# DenseBlock3
		channels = outChannels
		self.dense3 = self.make_dense_block(channels, growth_rate,
						num_dense_block, use_bottleneck)
		channels += num_dense_block * growth_rate
		
		# Output layer
		self.output_layer = nn.Sequential(
			nn.BatchNorm2d(channels),
			nn.ReLU(True),
			nn.Conv2d(channels, cls_num, kernel_size=1, bias=False),
			nn.AdaptiveAvgPool2d(1)
			)

		# Params initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, input_x):
		x = self.conv1(input_x)
		x = self.trans1(self.dense1(x))
		x = self.trans2(self.dense2(x))
		x = self.dense3(x)
		x = torch.squeeze(self.output_layer(x))
		x = F.log_softmax(x, dim=1)
		return x

	def make_dense_block(self, channels, growth_rate, num_dense_block, use_bottleneck):
		layers = []
		for i in range(int(num_dense_block)):
			if use_bottleneck:
				layers.append(Bottleneck(channels, growth_rate))
			else:
				layers.append(SingleLayer(channels, growth_rate))
			channels += growth_rate
		return nn.Sequential(*layers)


class Bottleneck(nn.Module):
	def __init__(self, channels, growth_rate):
		super(Bottleneck, self).__init__()
		inter_channels = 4 * growth_rate
		self.bn1 = nn.BatchNorm2d(channels)
		self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=1, 
								padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(inter_channels)
		self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3,
								padding=1, bias=False)

	def forward(self, input_x):
		x = self.conv1(F.relu(self.bn1(input_x)))
		x = self.conv2(F.relu(self.bn2(x)))
		x = torch.cat((input_x, x), 1)
		return x

class SingleLayer(nn.Module):
	def __init__(self, channels, growth_rate):
		super(SingleLayer, self).__init__
		self.bn1 = nn.BatchNorm2d(channels)
		self.conv1 = nn.Conv2d(channels, growth_rate, kernel_size=3,
								padding=1, bias=False)

	def forward(self, input_x):
		x = self.conv1(F.relu(self.bn1(input_x)))
		x = torch.cat((x, input_x), 1)
		return x

class Transition(nn.Module):
	def __init__(self, input_channels, out_channels):
		super(Transition, self).__init__()
		self.bn1 = nn.BatchNorm2d(input_channels)
		self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=1,
						padding=0, bias=False)

	def forward(self, input_x):
		x = self.conv1(self.bn1(input_x))
		x = F.avg_pool2d(x, 2)
		return x


