import numpy as np
import matplotlib.pyplot as plt

losses = np.load("data/losses.npy")
accuracies = np.load("data/accuracies.npy")

plt.plot(losses, color="b", label="Losses")
plt.plot(accuracies, color="g", label="Accuracies")
plt.title("Training Losses And Accuracies Plot")
plt.xlabel("Epoches")
plt.ylabel("Losses And Accuracies")
plt.grid(visible=True, which="major", axis="both", linewidth=1, linestyle="--")
plt.legend()
plt.savefig("img/losses.jpg")
