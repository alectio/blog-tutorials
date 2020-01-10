'''
entropy-based active learning
'''
import random
import torch
import torch.nn as nn
import torch.optim as optim


from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, DataLoader

from model import AlexNet
from query_strategy import EntropySelectQuery


num_loops = 10 # num of loops
num_epochs = 5# num of epochs to train in each loop
init_samples = 10000
n_rec = 2000 # num of samples to be labeled by oracle in each loop 

# directory to save CIFAR images
data_dir = '/home/ubuntu/DataLake/Data/CIFAR10' 

# training hyperparamters
batch_size = 512
learning_rate = 1e-4
weight_decay = 1e-4


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# setup dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914,0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010))
    ])


cifar_train = CIFAR10(root=data_dir, download=False, train=True, transform=transform)
cifar_test = CIFAR10(root=data_dir, download=False, train=False, transform=transform)

# setup model
model = AlexNet().to(device)

# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate,
    weight_decay=weight_decay)

# initialize our entropy based select query
select_fn = EntropySelectQuery(model, cifar_train)


# main training loop
unlabeled, labeled = [i for i in range(len(cifar_train))], []
for loop in range(num_loops):

    if loop == 0:
        # randomly select <init_samples> many samples
        selected = random.sample(unlabeled, init_samples) 
    else:
        # select based on entropy
        selected = select_fn(unlabeled, n_rec)

    
    # removed selected samples from the pool of unlabeled data
    unlabeled = list(set(unlabeled) - set(selected))
    
    # add selected samples to the pool of labeled data
    labeled.extend(selected)

    # train the model using labeled data
    model.train()
    for epoch in range(num_epochs):
        sampler = SubsetRandomSampler(labeled)
        dataloader = DataLoader(
                cifar_train, 
                batch_size=batch_size,
                sampler=sampler,
                pin_memory=True)

        step = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print("Epoch: {}, Step: {}, Loss: {}".format(epoch, step, loss.item()))
            step+=1 

    # validate the model on test set at the end of each loop
    model.eval()
    # average and number of correct prediction
    avg_loss, num_correct = 0.0, 0.0 
    dataloader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)
    step=0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            avg_loss += loss.item()

            pred = torch.argmax(outputs, dim=1)
            num_correct+= (pred == labels).float().sum().item()

            step+=1
        
        # compute average loss and accuracy
        avg_loss, accuracy = avg_loss/step, num_correct/len(cifar_test)


    print("Validation result after loop {}".format(loop))
    print("Loss: {}, Accuracy: {}".format(avg_loss, accuracy))







