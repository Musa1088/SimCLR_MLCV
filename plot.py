import re
import matplotlib.pyplot as plt

log_path = "May27_08-44-15_kirk/training.log"  # Update path if needed

epochs = []
losses = []
accuracies = []

with open(log_path, "r") as f:
    for line in f:
        # Match lines like: DEBUG:root:Epoch: 0	Loss: 4.53	Top1 accuracy: 11.32
        match = re.search(r"Epoch:\s*(\d+).*Loss:\s*([\d\.]+).*Top1 accuracy:\s*([\d\.]+)", line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            acc = float(match.group(3))
            epochs.append(epoch)
            losses.append(loss)
            accuracies.append(acc)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracies, label="Top1 Accuracy", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Top1 Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("training_loss_accuracy_finetuned.png", dpi=200)
plt.show()