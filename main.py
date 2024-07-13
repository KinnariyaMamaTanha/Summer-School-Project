import torch.utils.data
from torch.utils.data import random_split
from CustomDataLoader import SoundDS
from Testing import testing
from Model import AudioClassifier
from Training import training, kfold_training
import time
from MetadataLoader import MetadataLoader

# -------------------------------------------------------------
# ---- load metadata ----
# -------------------------------------------------------------

print("Begin loading metadata...")

download_path, df = MetadataLoader()

print("Metadata already loaded!\n\n")

# -------------------------------------------------------------
# ---- batch split ----
# -------------------------------------------------------------

print("Begin loading training and testing batches...")

data_path = download_path / "audio"

myds = SoundDS(df, data_path)

# Random split of 80:20 between training and testing
num_items = len(myds)

num_train = round(num_items * 0.8)
num_test = num_items - num_train
train_ds, test_ds = random_split(myds, [num_train, num_test])

# Create training and testing data loaders
# train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=True)

print("Batches already loaded!\n\n")

# -------------------------------------------------------------
# ---- model ----
# -------------------------------------------------------------

print("Begin initializing model...")

# Create model and try to move it to gpu if available
model = AudioClassifier()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("The model will be running on GPU.")
else:
    device = torch.device("cpu")
    print("The model will be running on CPU.")
model = model.to(device)

# Check that it is on Cuda
next(model.parameters()).device

print("Model already initialized!\n\n")

# -------------------------------------------------------------
# ---- training ----
# -------------------------------------------------------------

print("Begin training...")

t0 = time.perf_counter()

num_epochs = 50
# training(model, train_dl, num_epochs, device)
kfold_training(model, train_ds, num_epochs, device)

# Save model
torch.save(model.state_dict(), "data/model.pt")

t1 = time.perf_counter()

print(f"Training completed! Total time: {(t1 - t0):.2f}s\n\n")

# -------------------------------------------------------------
# ---- Testing ----
# -------------------------------------------------------------

print("Begin testing...")

t0 = time.perf_counter()

testing(model, test_dl, device)

t1 = time.perf_counter()

print(f"Testing completed! Total time: {(t1 - t0):.2f}s\n\n")
