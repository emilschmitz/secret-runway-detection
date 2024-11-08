# Imports
import numpy as np
import torch



def compute_baseline_accuracy(val_dataloader):
    """
    Computes the baseline pixel-wise accuracy by predicting all pixels as zero.

    Args:
        val_dataloader (DataLoader): DataLoader for the validation set.

    Returns:
        baseline_accuracy (float): The accuracy when predicting all zeros.
    """
    total_pixels = 0
    correct_predictions = 0

    for inputs, labels in val_dataloader:
        # Move labels to CPU and convert to NumPy array
        labels_np = labels.cpu().numpy()

        # Count how many labels are zero
        correct_predictions += np.sum(labels_np == 0)

        # Accumulate total number of pixels
        total_pixels += labels_np.size

    # Calculate baseline accuracy
    baseline_accuracy = correct_predictions / total_pixels

    return baseline_accuracy

def compute_validation_accuracy(model, val_dataloader, device):
    """
    Computes the pixel-wise accuracy over the validation set for multiple thresholds.

    Args:
        model: The trained model.
        val_dataloader: DataLoader for the validation set.
        device: The device (CPU or GPU) to perform computations on.

    Returns:
        best_accuracy (float): The highest accuracy achieved across thresholds.
        best_threshold (float): The threshold corresponding to the best accuracy.
    """
    model.eval()
    total_pixels = 0
    best_accuracy = 0.0
    best_threshold = 0.0
    thresholds = np.linspace(0.0, 1.0, 11)  # Thresholds from 0.0 to 1.0, inclusive
    threshold_correct = {threshold: 0 for threshold in thresholds}

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze(1)  # Adjust dimensions if necessary

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            # Move tensors to CPU and flatten for numpy operations
            probs_np = probs.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()

            total_pixels += labels_np.size

            # Evaluate at multiple thresholds
            for threshold in thresholds:
                preds = (probs_np >= threshold).astype(np.uint8)
                correct = (preds == labels_np).sum()
                threshold_correct[threshold] += correct

    # Compute average accuracy for each threshold
    for threshold in thresholds:
        accuracy = threshold_correct[threshold] / total_pixels
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_accuracy, best_threshold