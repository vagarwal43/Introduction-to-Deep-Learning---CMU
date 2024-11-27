import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import GCNConv, ChebConv



def get_dataset(save_path):
	'''
	read data from .npy file 
	no need to modify this function
	'''
	raw_data = np.load(save_path, allow_pickle=True)
	dataset = []
	for i, (node_f, edge_index, edge_attr, y)in enumerate(raw_data):
		sample = Data(
			x=torch.tensor(node_f, dtype=torch.float),
			y=torch.tensor([y], dtype=torch.float),
			edge_index=torch.tensor(edge_index, dtype=torch.long),
			edge_attr=torch.tensor(edge_attr, dtype=torch.float)
		)
		dataset.append(sample)
	return dataset


class GraphNet(nn.Module):
	'''
	Graph Neural Network class
	'''
	def __init__(self, n_features):
		'''
		n_features: number of features from dataset, should be 37
		'''
		super(GraphNet, self).__init__()

		self.embed = nn.Linear(n_features, 64)
		self.conv1 = GCNConv(64, 64, normalize=True)
		self.bn1 = nn.BatchNorm1d(64)
		self.conv2 = GCNConv(64, 64, normalize=True)
		self.bn2 = nn.BatchNorm1d(64)

		self.fc1 = nn.Linear(64,32)
		self.fc2 = nn.Linear(32,16)
		self.fc3 = nn.Linear(16,1)
		self.act = F.relu
		self.dropout = F.dropout
		
	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = self.embed(x.type(torch.FloatTensor))

		x = self.conv1(x,edge_index)
		x = self.bn1(x)
		x = self.act(x)
		x = self.dropout(x)

		x = self.conv2(x,edge_index)
		x = self.bn2(x)
		x = self.act(x)
		x = self.dropout(x)
		
		x = global_max_pool(x, data.batch)

		x = self.fc1(x)
		x = self.act(x)
		x = self.fc2(x)
		x = self.act(x)
		x = self.fc3(x)

		return x


def main():
	# load data and build the data loader
	train_set = get_dataset('train_set.npy')
	test_set = get_dataset('test_set.npy')
	train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

	# number of features in the dataset
	# no need to change the value
	n_features = 37

	# build your GNN model
	model = GraphNet(n_features)

	# define your loss and optimizer
	loss_func = F.mse_loss
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

	print(model)

	hist = {"train_loss":[], "test_loss":[]}
	num_epoch = 100
	for epoch in range(1, 1+num_epoch):
		model.train()
		loss_all = 0
		for data in train_loader:
			# your codes for training the model
			# ...
			optimizer.zero_grad()
			#print(len(data))
			pred = model(data)
			label = data.y

			loss = loss_func(pred.squeeze(), label)
			loss.backward()
			
			optimizer.step()
			loss_all += loss.item() * data.num_graphs
		train_loss = loss_all / len(train_set)

		with torch.no_grad():
			model.eval()
			loss_all = 0
			for data in test_loader:
				# your codes for validation on test set
				# ...

				pred = model(data)
				label = data.y

				loss = loss_func(pred.squeeze(), label)
				loss_all += loss.item() * data.num_graphs
			test_loss = loss_all / len(test_set)

			hist["train_loss"].append(train_loss)
			hist["test_loss"].append(test_loss)
			print(f'Epoch: {epoch}, Train loss: {train_loss:.3}, Test loss: {test_loss:.3}')

	# test on test set to get prediction 
	with torch.no_grad():
		model.eval()
		prediction = np.zeros(len(test_set))
		label = np.zeros(len(test_set))
		idx = 0
		for data in test_loader:
			output = model(data)
			prediction[idx:idx+len(output)] = output.squeeze().detach().numpy()
			label[idx:idx+len(output)] = data.y.detach().numpy()
			idx += len(output)
		prediction = np.array(prediction).squeeze()
		label = np.array(label).squeeze()
		total_mse = np.sum((prediction - label)**2)
		print(f'Final summed square error on test set: {total_mse:.3}')

	# save the model weights
	torch.save(model.state_dict(), 'gnn_weights.pt')

	# visualization
	# plot loss function
	ax = plt.subplot(1,1,1)
	ax.plot([e for e in range(1,1+num_epoch)], hist["train_loss"], label="train loss")
	ax.plot([e for e in range(1,1+num_epoch)], hist["test_loss"], label="test loss")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	ax.legend()
	plt.show()

	# plot prediction vs. label
	x = np.linspace(np.min(label), np.max(label))
	y = np.linspace(np.min(label), np.max(label))
	ax = plt.subplot(1,1,1)
	ax.scatter(prediction, label, marker='+', c='red')
	ax.plot(x, y, '--')
	plt.xlabel("prediction")
	plt.ylabel("label")
	plt.show()


if __name__ == "__main__":
	main()
