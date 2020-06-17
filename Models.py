import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sd_layer import *


class BasicConv(nn.Module):
	def __init__(self, in_f, out_f, kernel_size, stride=1, padding=1):
		super(BasicConv, self).__init__()
		self.conv = nn.Conv2d(in_f, out_f, kernel_size, stride, padding)
		self.bn = nn.BatchNorm2d(out_f)
		self.relu = nn.ReLU()
		self._initialize_weights()

	def forward(self, inputs):
		out = self.relu(self.bn(self.conv(inputs)))
		return out

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.ConvTranspose2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))

			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

class network(nn.Module):
	def __init__(self, image_name, gpu_no):
		super(network, self).__init__()
		self.sd_layer = sd_layer_pytorch_modular_dct_no_threshold_trainable(image_name, gpu_no)
		layers = []
		layers.append(BasicConv(3,16,5,2,1))
		layers.append(BasicConv(16,16,3,1,1))
		layers.append(nn.MaxPool2d(2))
		layers.append(BasicConv(16,32,3,1,1))
		layers.append(nn.MaxPool2d(2))
		layers.append(BasicConv(32,48,3,1,1))
		layers.append(nn.MaxPool2d(2))
		layers.append(BasicConv(48,64,3,1,1))
		layers.append(nn.MaxPool2d(2))
		layers.append(BasicConv(64,64,3,1,1))
		self.layers = nn.Sequential(*layers)
		self.gpu_no = gpu_no
		self.linear = nn.Linear(64**2, 2)
		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.ConvTranspose2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))

			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def forward(self, inputs):
		out = self.sd_layer(inputs).cuda(self.gpu_no)
		out = self.layers(out)
		out_t = out.view(out.shape[0], out.shape[1],-1).transpose(1,2)
		out = out.view(out.shape[0], out.shape[1], -1)
		out = torch.bmm(out,out_t)
		out = out.view(out.shape[0], -1)
		out = torch.sign(out) * torch.sqrt( torch.abs(out) + 1e-5)
		out = torch.nn.functional.normalize(out)
		out = self.linear(out)
		out = F.log_softmax(out, dim=1)
		return out
