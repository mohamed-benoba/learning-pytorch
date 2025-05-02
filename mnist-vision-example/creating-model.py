import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


##### TRAINING PIPELINE #####

# The training pipeline consists of the following steps:
# 1. Load the data

# --- CREATE THE MODEL
# 2. Create hyperparameters
# 3. Create a model
# 4. Define a loss function
# 5. Define an optimizer
# 6. Train the model

# 7. Test the model
# 8. Save the model

# --- USE MODEL
# 9. Load the model
# 10. Make predictions
# 11. Visualize the results

# 12. Evaluate the model
# 13. Tune the hyperparameters

# 14. Deploy the model
# 15. Monitor the model
# 16. Update the model


# -----

# Downloed the Fashion MNIST dataset 
training_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True, # Once downloaded, it will not be downloaded again
    transform = ToTensor() # Convert the data to PyTorch tensors
)

test_data = datasets.FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor()
)

# hyperparameters
batch_size = 64

# The batch size is the number of samples that will be propagated through the network in one forward/backward pass.
# The larger the batch size, the more memory space you'll need.
# A smaller batch size means more updates to the model parameters, which can lead to a more accurate model.
# However, smaller batch sizes can also lead to longer training times.
# The batch size is a hyperparameter that you can tune to find the best performance for your model.

# What is a hyperparameter?
# A hyperparameter is a parameter whose value is used to control the learning process.
# Hyperparameters are different from model parameters, which are learned during training.
# Common hyperparameters include the learning rate, batch size, and number of epochs.
# Hyperparameters are typically set before the training process begins and are not updated during training.
# Hyperparameters are often tuned using techniques such as grid search or random search.
# Hyperparameters are important because they can have a significant impact on the performance of the model.


# -----

# DataLoader is a PyTorch class that returns an iterable over the dataset.
# It allows you to easily iterate over the dataset in batches, shuffle the data, and load the data in parallel using multiprocessing workers.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
# Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28]) -> 64 images, 1 channel, 28x28 pixels
# Shape of y: torch.Size([64]) torch.int64 -> 64 labels


# -----

device = device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# To define a neural network in PyTorch, we create a class that inherits from nn.Module
class NeuralNetwork(nn.Module):
    ##  Initialize layers & components of the network in __init__
    def __init__(self):
        super().__init__()
        # Flatten layer
        self.flatten = nn.Flatten() # Converts multi-dimensional input tensors into a 1D tensor. Simply, it converts the image into its pixels.
        # Here you create the layers of the neural network (Input, hidden, output)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # Linear layer that takes the input of size 28*28 (784) and outputs 512 features
            nn.ReLU(), # Relu is an activation function that introduces non-linearity into the model. if feature is not important it outputs 0
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10) # predict 10 labels
        )

    ## Specify how the data flows in the network in the forward function (feedforward is passing data from one node to another) 
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print("-----")
print(model)


# To train a model, we need a loss function and an optimizer.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
