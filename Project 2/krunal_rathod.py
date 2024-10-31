"""
Assignment 2
Student: Krunal Rathod
"""
# *** Packges ***
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.nn import Dropout
from rich import print
from rich.console import Console
from rich.table import Table
from art import text2art
import copy
import os

debug = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# *** Functions ***
def imshow(img, dir, name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(name)
    plt.savefig(dir + name + ".png")


def calculate_accuracy(output, labels):
    _, prediction = torch.max(output.data, 1)
    correct = (prediction == labels).sum().item()
    acuracy = correct / batch_size
    return acuracy


if __name__ == "__main__":
    print(text2art("DLL: Assignment 2"))

    # Write your code here
    if torch.cuda.is_available():
        print("GPU is available, and device is:", torch.cuda.get_device_name(0))
    else:
        print("GPU is not available, CPU is used")

    """
    DON'T MODIFY THE SEED
    """
    # Set the seed for reproducibility
    manual_seed = 42
    torch.manual_seed(manual_seed)

    print("-------------------------")
    print("[bold cyan]QUESTION 1.1: DATA [/bold cyan]")
    print("-------------------------")

    print("[bold green]Question 1.1.1 (5pts) [/bold green]")

    # Loading the Data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 32

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # Observing the data
    first_images = {}

    for images, labels in trainset:
        class_name = classes[labels]
        if class_name not in first_images:
            first_images[class_name] = images

        if len(first_images) == len(classes):
            break

    dir1 = "./figures/1.1/"
    for class_name, image in first_images.items():
        imshow(image, dir1, class_name)
    print(
        "First images of each class are saved in the [bold yellow]figures/1.1[bold yellow] folder"
    )

    # Creating Histogram to show the distribution of the data

    ## Initializing the counts
    train_counts = {class_name: 0 for class_name in classes}
    test_counts = {class_name: 0 for class_name in classes}

    for images, labels in trainset:
        class_name = classes[labels]
        train_counts[class_name] += 1

    for images, labels in testset:
        class_name = classes[labels]
        test_counts[class_name] += 1

    ## Plotting the histogram
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(train_counts.keys(), train_counts.values(), color="g")
    plt.title("Train Dataset")
    plt.xlabel("Classes")
    plt.ylabel("Counts")

    plt.subplot(1, 2, 2)
    plt.bar(test_counts.keys(), test_counts.values(), color="b")
    plt.title("Test Dataset")
    plt.xlabel("Classes")
    plt.ylabel("Counts")

    plt.tight_layout()

    if not os.path.exists(dir1):
        os.makedirs(dir1)
    plt.savefig(dir1 + "histogram.png")

    print(
        "Histogram of the data is saved in the [bold yellow]figures/1.1[bold yellow] folder"
    )

    print("\n[bold green]Question 1.1.2 (5pts) [/bold green]")

    # Creating the dataloader to convert the data to tensor

    trainloader1 = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader1 = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Getting a sample from the dataset to check if the data is correct type
    data_sample, label_sample = next(iter(trainloader1))

    print("Data type of the sample is:", data_sample.dtype)

    if not isinstance(data_sample, torch.Tensor):
        data_sample = torch.tensor(data_sample)
        print("Data type of the sample is changed to:", data_sample.dtype)

    if not isinstance(label_sample, torch.Tensor):
        label_sample = torch.tensor(label_sample)
        print("Data type of the sample is changed to:", label_sample.dtype)

    # Print Dimension of the tensor
    print("Dimension of the data tensor is: ", data_sample.shape)

    print("\n[bold green]Question 1.1.3 (5pts) [/bold green]")
    transformNorm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0, 0, 0), (1, 1, 1))]
    )

    trainset.transform = transformNorm
    testset.transform = transformNorm

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)

    pixels_red = []
    pixels_green = []
    pixels_blue = []

    for data, _ in trainset:
        red, green, blue = data.split(1)
        pixels_red.extend(red.flatten().tolist())
        pixels_green.extend(green.flatten().tolist())
        pixels_blue.extend(blue.flatten().tolist())

    # Calculate the mean and standard deviation for each channel
    mean_red = sum(pixels_red) / len(pixels_red)
    std_red = (sum((i - mean_red) ** 2 for i in pixels_red) / len(pixels_red)) ** 0.5
    mean_green = sum(pixels_green) / len(pixels_green)
    std_green = (
        sum((i - mean_green) ** 2 for i in pixels_green) / len(pixels_green)
    ) ** 0.5
    mean_blue = sum(pixels_blue) / len(pixels_blue)
    std_blue = (
        sum((i - mean_blue) ** 2 for i in pixels_blue) / len(pixels_blue)
    ) ** 0.5

    console = Console()

    print("Before Normalization")
    # Create a table
    table = Table(show_header=True, header_style="bold magenta")

    # Add columns to the table

    table.add_column("Channel", style="cyan")
    table.add_column("Mean", justify="center", style="green")
    table.add_column("Std Deviation", justify="center", style="blue")

    # Add data to the table
    table.add_row("Red", f"{mean_red:.4f}", f"{std_red:.4f}")
    table.add_row("Green", f"{mean_green:.4f}", f"{std_green:.4f}")
    table.add_row("Blue", f"{mean_blue:.4f}", f"{std_blue:.4f}")

    # Print the table
    console.print(table)

    # Convert mean and std to tensors
    mean_tensor = torch.tensor([mean_red, mean_green, mean_blue])
    std_tensor = torch.tensor([std_red, std_green, std_blue])

    # Use the calculated mean and std for normalization
    normalize = transforms.Normalize(mean_tensor, std_tensor)

    # Combine the existing transformations with the new normalization transform
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Apply the composed transform to the training and test datasets
    trainset.transform = transform
    testset.transform = transform

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    if debug:
        # Calculating the mean and std again
        # Initialize lists to store pixel values for each channel
        pixels_red = []
        pixels_green = []
        pixels_blue = []

        # Iterate over the entire training set
        for data, _ in trainset:
            red, green, blue = data.split(1)
            pixels_red.extend(red.flatten().tolist())
            pixels_green.extend(green.flatten().tolist())
            pixels_blue.extend(blue.flatten().tolist())

        # Convert lists to tensors
        pixels_red = torch.tensor(pixels_red)
        pixels_green = torch.tensor(pixels_green)
        pixels_blue = torch.tensor(pixels_blue)

        # Calculate the mean and standard deviation for each channel
        new_mean_red = torch.mean(pixels_red)
        new_std_red = torch.std(pixels_red)
        new_mean_green = torch.mean(pixels_green)
        new_std_green = torch.std(pixels_green)
        new_mean_blue = torch.mean(pixels_blue)
        new_std_blue = torch.std(pixels_blue)

        print("After Normalization")
        # Create a table
        table = Table(show_header=True, header_style="bold magenta")

        # Add columns to the table
        table.add_column("Channel", style="cyan")
        table.add_column("Mean", justify="center", style="green")
        table.add_column("Std Deviation", justify="center", style="blue")

        # Add data to the table
        table.add_row("Red", f"{new_mean_red:.4f}", f"{new_std_red:.4f}")
        table.add_row("Green", f"{new_mean_green:.4f}", f"{new_std_green:.4f}")
        table.add_row("Blue", f"{new_mean_blue:.4f}", f"{new_std_blue:.4f}")

        # Print the table
        console.print(table)

    print("[bold green]Question 1.1.4 (5pts) [/bold green]")

    train_indices, val_indices = train_test_split(
        list(range(len(trainset))), test_size=0.2, random_state=manual_seed
    )

    trainset_new = Subset(trainset, train_indices)
    valset = Subset(trainset, val_indices)

    trainloader = torch.utils.data.DataLoader(
        trainset_new, batch_size=batch_size, shuffle=True, num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # Print size of valloader
    dataSampleVal, labelsampee = next(iter(valloader))
    dataSampleTrain, labelsampee = next(iter(trainloader))
    dataSampleTest, labelsampee = next(iter(testloader))
    print("Size of the validation set:", dataSampleVal.shape)
    print("Size of the training set:", dataSampleTrain.shape)
    print("Size of the test set:", dataSampleTest.shape)

    print("-------------------------")
    print("[bold cyan]QUESTION 1.2: MODEL (10pts)[/bold cyan]")
    print("-------------------------")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
            self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
            self.conv3 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))

            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = Net().to(device)
    print(model)

    print("-------------------------")
    print("[bold cyan]QUESTION 1.3: TRAINING (60 pts)[/bold cyan]")
    print("-------------------------")

    print("[bold green]Question 1.3.1 (15pts) [/bold green]")

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 15

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    if debug:
        print("Training the model")
        # Training Loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_loss_final = 0.0
            correct = 0
            total = 0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                # Forward Pass
                outputs = model(inputs)
                loss_train = criterion(outputs, labels)

                # Backward and Optimize
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                running_loss += loss_train.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total

                if i % 1000 == 999:
                    print(
                        "[Epoch: %d, Batch: %5d] loss: %.3f accuracy: %.3f"
                        % (epoch + 1, i + 1, running_loss / 100, accuracy)
                    )
                    running_loss_final = running_loss
                    running_loss = 0.0
            train_losses.append(running_loss_final / len(trainloader))
            train_accuracies.append(100 * correct / total)

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data in valloader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    loss_val = criterion(outputs, labels)

                    val_loss += loss_val.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(valloader)
            val_accuracy = 100 * val_correct / val_total
            print("Validation loss: %.3f accuracy: %.3f" % (val_loss, val_accuracy))
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            test_loss = 0
            test_correct = 0
            test_total = 0

            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    test_loss += loss.item()

                test_loss = test_loss / len(testloader)
                test_accuracy = 100 * test_correct / test_total

        print("[bold green]Question 1.3.2 (13pts) [/bold green]")
        print("Training done in 15 epochs")
        print(
            "Test Loss: {:.3f}, Test Accuracy: {:.3f}".format(test_loss, test_accuracy)
        )
        print("Final training loss:", train_losses[-1])
        print("Final training accuracy:", train_accuracies[-1])

        print("[bold green]Question 1.3.3 (2pts) [/bold green]")
        torch.save(model.state_dict(), "harkeerat_sawhney_1.pt")
        print("[bold green]Question 1.3.4 (10pts) [/bold green]")

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(val_accuracies, label="Val Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        if not os.path.exists("./figures/1.3/"):
            os.makedirs("./figures/1.3/")
        plt.savefig("./figures/1.3/loss_accuracy.png")

    print("[bold green]Question 1.3.5 (18pts) [/bold green]")

    batch_size = 32

    trainloader1 = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader1 = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Getting a sample from the dataset to check if the data is correct type
    data_sample, label_sample = next(iter(trainloader1))

    print("Data type of the sample is:", data_sample.dtype)

    if not isinstance(data_sample, torch.Tensor):
        data_sample = torch.tensor(data_sample)
        print("Data type of the sample is changed to:", data_sample.dtype)

    if not isinstance(label_sample, torch.Tensor):
        label_sample = torch.tensor(label_sample)
        print("Data type of the sample is changed to:", label_sample.dtype)

    # Print Dimension of the tensor
    print("Dimension of the data tensor is: ", data_sample.shape)

    transformNorm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0, 0, 0), (1, 1, 1))]
    )

    trainset.transform = transformNorm
    testset.transform = transformNorm

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)

    pixels_red = []
    pixels_green = []
    pixels_blue = []

    for data, _ in trainset:
        red, green, blue = data.split(1)
        pixels_red.extend(red.flatten().tolist())
        pixels_green.extend(green.flatten().tolist())
        pixels_blue.extend(blue.flatten().tolist())

    # Calculate the mean and standard deviation for each channel
    mean_red = sum(pixels_red) / len(pixels_red)
    std_red = (sum((i - mean_red) ** 2 for i in pixels_red) / len(pixels_red)) ** 0.5
    mean_green = sum(pixels_green) / len(pixels_green)
    std_green = (
        sum((i - mean_green) ** 2 for i in pixels_green) / len(pixels_green)
    ) ** 0.5
    mean_blue = sum(pixels_blue) / len(pixels_blue)
    std_blue = (
        sum((i - mean_blue) ** 2 for i in pixels_blue) / len(pixels_blue)
    ) ** 0.5

    # Convert mean and std to tensors
    mean_tensor = torch.tensor([mean_red, mean_green, mean_blue])
    std_tensor = torch.tensor([std_red, std_green, std_blue])

    # Use the calculated mean and std for normalization
    normalize = transforms.Normalize(mean_tensor, std_tensor)

    # Combine the existing transformations with the new normalization transform
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Apply the composed transform to the training and test datasets
    trainset.transform = transform
    testset.transform = transform

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    class CustomCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
            self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
            self.conv3 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
            self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout1 = Dropout(0.25)
            self.dropout2 = Dropout(0.5)
            self.fc1 = nn.Linear(512 * 1 * 1, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.gelu(self.conv1(x)))
            x = self.pool(F.gelu(self.conv2(x)))
            x = self.pool(F.gelu(self.conv3(x)))
            x = self.pool(F.gelu(self.conv4(x)))
            x = self.pool(F.gelu(self.conv5(x)))
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = F.gelu(self.fc1(x))
            x = self.dropout2(x)
            x = F.gelu(self.fc2(x))
            x = self.fc3(x)
            return x

    print("[bold green]Question CustomCNN (5pts) [/bold green]")

    modelCustom = CustomCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(modelCustom.parameters(), lr=0.003, momentum=0.95)

    print(modelCustom)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    num_epochs_custom = 20
    for epoch in range(num_epochs_custom):
        train_loss, train_correct, train_total = 0, 0, 0
        val_loss, val_correct, val_total = 0, 0, 0

        modelCustom.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = modelCustom(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()

        modelCustom.eval()
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = modelCustom(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()

        train_loss = train_loss / len(trainloader)
        train_accuracy = 100 * train_correct / train_total
        val_loss = val_loss / len(valloader)
        val_accuracy = 100 * val_correct / val_total

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            "Epoch: {}, Train Loss: {:.3f}, Train Accuracy: {:.3f}, Val Loss: {:.3f}, Val Accuracy: {:.3f}".format(
                epoch, train_loss, train_accuracy, val_loss, val_accuracy
            )
        )

    test_loss = 0
    test_correct = 0
    test_total = 0

    modelCustom.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = modelCustom(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_loss += loss.item()

    test_loss = test_loss / len(testloader)
    test_accuracy = 100 * test_correct / test_total

    print("Test Loss: {:.3f}, Test Accuracy: {:.3f}".format(test_loss, test_accuracy))

    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    if not os.path.exists("./figures/1.4/"):
        os.makedirs("./figures/1.4/")
    plt.savefig("./figures/1.4/loss_accuracy.png")

    print("[bold green]Question 1.3.6 (2pts) [/bold green]")
    torch.save(modelCustom.state_dict(), "harkeerat_sawhney_2.pt")

    """
    Code for bonus question
    """
    for seed in range(10):
        torch.manual_seed(seed)
        # Train the models here
