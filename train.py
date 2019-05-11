import torch
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os
import argparse
from DenseNet import DenseNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def getParams():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--epochs", type=int, default=300)
	parser.add_argument("--optimizer", type=str, default="sgd",
						choices=("sgd", "adam", "rmsprop"))
	parser.add_argument("--save_dir", type=str, default="models/cifar10")
	parser.add_argument("--data_dir", type=str, default="data/cifar10")
	parser.add_argument("--display_step", type=int, default=20)
	args = parser.parse_args()
	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)
	if not os.path.exists(args.data_dir):
		os.mkdir(args.data_dir)
	return args

def main():
	args = getParams()

	normMean = [0.49139968, 0.48215827, 0.44653124]
	normStd = [0.24703233, 0.24348505, 0.26158768]
	normTransform = transforms.Normalize(normMean, normStd)

	trainTransform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normTransform
	])
	testTransform = transforms.Compose([
		transforms.ToTensor(),
		normTransform
	])

	trainData = datasets.CIFAR10(root=args.data_dir, train=True, download=True,
								transform=trainTransform)
	testData = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
								transform=testTransform)

	trainLoader = DataLoader(trainData, batch_size=args.batch_size,
							shuffle=True, num_workers=8, pin_memory=True)
	testLoader = DataLoader(testData, batch_size=args.batch_size,
							shuffle=False, num_workers=8, pin_memory=True)

	net = DenseNet(cls_num=10, depth=100, growth_rate=12, 
					compress_factor=0.5, use_bottleneck=True)
	print('  + Number of params: {}'.format(
		sum([p.data.nelement() for p in net.parameters()])))

	net.to(device)

	if args.optimizer == 'sgd':
		optimizer = optim.SGD(net.parameters(), lr=1e-1,
						momentum=0.9, weight_decay=1e-4)
	elif args.optimizer == 'adam':
		optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
	elif args.optimizer == 'rmsprop':
		optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

	train_log = os.path.join(args.save_dir, "train_log.txt")
	test_log = os.path.join(args.save_dir, "test_log.txt")

	for epoch in range(1, args.epochs+1):
		if args.optimizer == 'sgd':
			adjust_lr(optimizer, epoch)
		train(args, epoch, net, trainLoader, optimizer, train_log)
		test(args, epoch, net, testLoader, optimizer, test_log)
		torch.save(net, os.path.join(args.save_dir, "last.pth"))

def adjust_lr(optimizer, epoch):
	if epoch < 150:
		lr = 1e-1
	elif epoch == 150:
		lr = 1e-2
	elif epoch == 225:
		lr = 1e-3
	else:
		return 
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def train(args, epoch, net, trainLoader, optimizer, train_log):
	net.train()
	runing_loss = 0.0
	for batch_idx, data in enumerate(trainLoader):
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)
		# inputs, labels = Variable(inputs), Variable(labels)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = F.nll_loss(outputs, labels)
		loss.backward()
		optimizer.step()
		runing_loss += loss.item()
        if ((batch_idx+1) % args.display_step) == 0:
            print("Train Epoch: %d/%d,  Batch Num: %d/%d, Avg Loss: %.6f" %\
                (epoch, args.epochs, batch_idx+1, 
                len(trainLoader),runing_loss/args.display_step))
            runing_loss = 0.0
        # write the loss to train log file
        with open(train_log, 'a') as f:
            f.write("{},{}\n".format(
                epoch+(batch_idx+1.)/len(trainLoader),loss.item()))
		

def test(args, epoch, net, testLoader, optimizer, test_log):
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size()[0]
            correct += (predicted == labels).sum().item()
    acc = 100.*correct/total
    print "Test Epoch: %d/%d,  Accuracy: %.2f %%" % (epoch, args.epochs, acc)
    with open(test_log, 'a') as f:
        f.write("{},{}\n".format(epoch,acc))


if __name__ == "__main__":
    main()
