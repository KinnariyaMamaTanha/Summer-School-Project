import time
from torch import nn
import torch
import torch.utils
from Model import AudioClassifier
import numpy as np
from sklearn.model_selection import KFold
from Testing import testing


def kfold_training(model: AudioClassifier, train_ds, num_epochs: int, device: torch.device):
    # Number of epochs to train per fold
    epoch_per_fold = 1
    epoch = 0
    # Create an array of indices for the dataset
    data_indices = np.arange(len(train_ds))
    # Initialize KFold with 10 splits
    kfold = KFold(n_splits=10)

    losses = []
    accuracies = []
    validation_accuracies = []
    stop_training = False

    while not stop_training:
        # Split the dataset into training and validation sets
        for i, (train_indices, validation_indices) in enumerate(kfold.split(data_indices)):
            train_data = torch.utils.data.dataset.Subset(
                train_ds, train_indices)
            val_data = torch.utils.data.dataset.Subset(
                train_ds, validation_indices)

            train_dl = torch.utils.data.DataLoader(
                train_data, batch_size=16, shuffle=True)
            val_dl = torch.utils.data.DataLoader(
                val_data, batch_size=16, shuffle=True)

            for fold_epoch in range(epoch_per_fold):
                # Train the model for one epoch
                fold_losses, fold_accuracies = training(
                    model, train_dl, 1, device, base_epoch=epoch, from_kfold=True)

                losses.append(fold_losses)
                accuracies.append(fold_accuracies)

                print(f"[Validation Epoch {epoch + 1} Fold {i + 1}]")
                # Test the model and append validation accuracies
                validation_accuracies.append(
                    testing(model, val_dl, device, from_kfold=True))

                if epoch % 5 == 4:
                    c = int(input("Continue training? 0: break, 1: continue\n"))
                    if c == 0:
                        stop_training = True
                        break

                epoch += 1
                if epoch >= num_epochs:
                    stop_training = True
                    break

            if stop_training:
                break

    np.save("data/losses.npy", arr=np.array(losses))
    np.save("data/accuracies.npy", arr=np.array(accuracies))
    np.save("data/validation_acc.npy", arr=np.array(validation_accuracies))


def training(model: AudioClassifier, train_dl, num_epochs: int, device: torch.device, base_epoch: int = 0, from_kfold=False):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=0.001,
        steps_per_epoch=int(len(train_dl)),
        epochs=num_epochs,
        anneal_strategy="linear",
    )

    losses = []
    accuracies = []

    # Repeat for each epoch
    for epoch in range(num_epochs):
        t0 = time.perf_counter()
        running_loss: float = 0.0
        correct_prediction: int = 0
        total_prediction: int = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs: torch.Tensor
            labels: torch.Tensor
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the gradiants
            optimizer.zero_grad()

            # forward, backward, and optimize
            outputs: torch.Tensor
            outputs = model(inputs)
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()  # type: ignore
            total_prediction += prediction.shape[0]

            if i % 10 == 0:  # print every 10 mini-batches
                print("[%d, %5d] loss: %.3f" % (base_epoch + epoch + 1, i + 1, running_loss / 10))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        t1 = time.perf_counter()
        print(
            f"Epoch: {base_epoch + epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, Total time: {(t1 - t0):.2f}s"
        )
        if (base_epoch + epoch) % 5 == 4 and from_kfold == False:
            c = int(input("Continue training? 0: break, 1: continue\n"))
            if c == 0:
                break

        # Save losses and accuracies for ploting
        losses.append(avg_loss)
        accuracies.append(acc)

    if from_kfold == False: 
        np.save("data/losses.npy", arr=np.array(losses))
        np.save("data/accuracies.npy", arr=np.array(accuracies))
    else:
        return losses, accuracies
