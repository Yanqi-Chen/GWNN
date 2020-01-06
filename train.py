from utils import *
from data import *
from models import *
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train GWNN.")

parser.add_argument("--epochs",
					type=int,
					default=200,
					help="Number of training epochs. Default is 200.")

parser.add_argument("--dataset",
					default='cora',
					choices=['cora', 
							 'citeseer', 
							 'pubmed'],
					help="Which dataset to use. Default is 'cora'.")

parser.add_argument("--hidden", 
					type=int, 
					default=16, 
					help="Number of units in hidden layer. Default is 16.")

parser.add_argument("--learning-rate", 
					type=float, 
					default=0.01, 
					help="Learning rate. Default is 0.01.")

parser.add_argument("--dropout",
					type=float,
					default=0.5,
					help="Dropout probability. Default is 0.5.")

parser.add_argument("--scale",
					type=float,
					default=1.0,
					help="Scaling parameter. Default is 1.0.")

parser.add_argument("--weight-decay",
					type=float,
					default=5e-4,
					help="Adam weight decay. Default is 1e-5.")

parser.add_argument("--threshold",
					type=float,
					default=1e-4,
					help="Sparsification parameter. Default is 1e-4.")

parser.add_argument("--save-path",
					nargs="?",
					default="./models",
					help="Target classes csv.")

parser.add_argument("--fast",
					action='store_true',
					help="Use fast graph wavelets with Chebyshev polynomial approximation")

parser.add_argument("--approximation-order",
					type=int,
					default=4,
					help="Order of Chebyshev polynomial. Default is 4.")

args = parser.parse_args()

def train(times):
	# Delete old models
	print('Deleting old models...')
	path = args.save_path
	if not os.path.exists(path):
		os.makedirs(path)
	else:
		for root, dirs, files in os.walk(path):
			for file in files:
				if file.startswith(args.dataset):
					os.remove(os.path.join(root, file))
					print ("Delete File: " + os.path.join(root, file))
	print('Old models deleted')

	print('Loading data...')
	adj, features, y_train, y_val, y_test, classes_num, train_mask, val_mask, test_mask = load_data(args.dataset)
	print('Loading data complete')

	# features = preprocess_features(features)

	train_num, val_num, test_num = y_train.shape[0], y_val.shape[0], y_test.shape[0]
	print('Calculating wavelet basis...')
	if args.fast:
		wavelets, wavelet_inv = fast_wavelet_basis(adj, args.scale, args.threshold, args.approximation_order)
	else:
		wavelets, wavelet_inv = wavelet_basis(adj, args.scale, args.threshold)

	print('Wavelet basis complete')

	#debug
	#exit(0)

	wavelets, wavelet_inv = (torch.from_numpy(wavelets.toarray())).float().to(device), (torch.from_numpy(wavelet_inv.toarray())).float().to(device)
	if times == 0:
		writer = SummaryWriter(f'runs/{args.dataset}-lr={args.learning_rate}-s={args.scale}')
	print('Building model...')
	model = GraphWaveletNeuralNetwork((adj.shape)[0], 
									(features.shape)[1], 
									args.hidden,
									classes_num,
									wavelets,
									wavelet_inv,
									args.dropout)

	optimizer = torch.optim.Adam(model.parameters(),
									lr=args.learning_rate,
									weight_decay=args.weight_decay)

	loss_fn = nn.NLLLoss(reduction='mean')

	features = (torch.from_numpy(features.toarray())).float().to(device)

	print(f'Shape of feature: {features.shape}')

	y_train = (torch.from_numpy(y_train)).long().to(device)
	y_val = (torch.from_numpy(y_val)).long().to(device)
	y_test = (torch.from_numpy(y_test)).long().to(device)

	model.to(device)

	model.train()
	max_val_acc = 0
	test_acc = 0

	print('Training start')
	for epoch in range(1, args.epochs + 1):
		optimizer.zero_grad()

		y_pred = model(features)

		loss = loss_fn(y_pred[train_mask], y_train)

		loss.backward()
		optimizer.step()
		with torch.no_grad():
			model.eval()

			y_pred = model(features)
			val_loss = loss_fn(y_pred[val_mask], y_val)
			_, y_pred = y_pred[val_mask].max(dim=1)
			

			correct = y_pred.eq(y_val).sum().item()
			accuracy = correct/int(val_num)

			if times == 0:
				writer.add_scalar('validation acc', accuracy, epoch)

			save_model = False

			if accuracy >= max_val_acc:
				max_val_acc = accuracy
				torch.save(model.state_dict(), f'./models/{args.dataset}-{epoch}.pth')
				save_model = True

			y_pred = model(features)
			test_loss = loss_fn(y_pred[test_mask], y_test)
			_, y_pred = y_pred[test_mask].max(dim=1)
			

			correct = y_pred.eq(y_test).sum().item()
			accuracy2 = correct/int(test_num)

			if times == 0:
				writer.add_scalar('training loss', loss, epoch)

			print(f"Epoch {epoch}: Train loss: {loss}, Val loss: {val_loss}, Val acc: {accuracy:.4f}, Test loss: {test_loss}, Test acc: {accuracy2:.4f}")

			if times == 0:
				writer.add_scalar('test acc', accuracy2, epoch)

			if save_model:
				test_acc = accuracy2

	filename = f'{args.dataset}-lr={args.learning_rate}-s={args.scale}.txt'
	with open(filename, 'a') as file_object:
		file_object.write(f'{test_acc:.4f}\n')
	print(f"Final test accuracy: {test_acc:.4f}")
	print(args)

	if times == 0:
		writer.add_graph(model, (features,))
		writer.close()

for times in range(1):
	train(times)