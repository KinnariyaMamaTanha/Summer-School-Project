import torch
import Model


def testing(model: Model.AudioClassifier, test_dl, device: torch.device):
    correct_prediction: int = 0
    total_prediction: int = 0
    with torch.no_grad():
        for data in test_dl:
            inputs: torch.Tensor
            labels: torch.Tensor
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize inputs
            inputs = (inputs - torch.mean(inputs)) / torch.std(inputs)

            # Get predictions
            outputs = model(inputs)
            _, prediction = torch.max(outputs, 1)

            # Count of predictions that matched the target label
            correct_prediction += (labels == prediction).sum().item()  # type: ignore
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f"Accuracy: {acc:.2f}, Total items: {total_prediction}")
