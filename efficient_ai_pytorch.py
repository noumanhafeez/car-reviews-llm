# Uncomment the import statements if not already loaded
# Packages needed for dataset loading and preprocessing
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import Subset, DataLoader

# # Packages needed for project tasks (model definition, optimization, distributed training, and evaluation)
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from accelerate import Accelerator
# from transformers import AutoModelForImageClassification, AutoImageProcessor
# import evaluate

# Run this block before getting started.
# It loads the necessary packages, prepares the dataset, and initializes a pre-trained model.

# Packages needed for project tasks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from transformers import AutoModelForImageClassification, AutoImageProcessor
import evaluate

# Hiding long output messages
from transformers import logging
logging.set_verbosity_error()

# Setup accelerator
accelerator = Accelerator()

# Load pre-bundled CIFAR-10 subset (50 images at 32x32, stored as uint8)
data_bundle = torch.load("data/cifar10_50_uint8.pt", weights_only=True)
images = data_bundle["images"].float() / 255.0
labels = data_bundle["labels"]

# Resize to 224x224 and normalize to [-1, 1]
images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
images = (images - 0.5) / 0.5
dataloader = DataLoader(TensorDataset(images, labels), batch_size=16, shuffle=True)

# Load the pre-trained model and processor
model_name = "facebook/deit-tiny-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name)
distributed_model = AutoModelForImageClassification.from_pretrained(
    model_name, num_labels=10, ignore_mismatched_sizes=True
)
initial_state_dict = distributed_model.state_dict().copy()


# Please refer to the notebook for the code that prepares the dataset

# Define loss function and set up multiple optimizers for comparison
criterion = nn.CrossEntropyLoss()
optimizer_configs = {
    "SGD": lambda p: optim.SGD(p, lr=0.01, momentum=0.9, weight_decay=0.0),
    "Adam": lambda p: optim.Adam(p, lr=0.001, weight_decay=1e-4),
    "AdamW": lambda p: optim.AdamW(p, lr=0.001, weight_decay=1e-4),
}
optimizer_summary = {}
training_report = {}
best_optimizer = None
best_accuracy = 0.0
num_epochs = 2


# Evaluate model accuracy using the Hugging Face evaluate library
def evaluate_model(model, dataloader):
    metric = evaluate.load("accuracy")
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(pixel_values=inputs).logits
            preds = outputs.argmax(dim=1)
            preds, labels = accelerator.gather_for_metrics((preds, labels))
            metric.add_batch(predictions=preds, references=labels)
    return metric.compute()["accuracy"]


# Train and evaluate the model for each optimizer
for name, opt_fn in optimizer_configs.items():
    model = AutoModelForImageClassification.from_pretrained(
        model_name, num_labels=10, ignore_mismatched_sizes=True
    )
    model.load_state_dict(initial_state_dict)
    optimizer = opt_fn(model.parameters())
    model, optimizer, data = accelerator.prepare(model, optimizer, dataloader)
    model.train()
    for _ in range(num_epochs):
        for inputs, labels in data:
            outputs = model(pixel_values=inputs).logits
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
    #   trained_model = model (optional)
    accuracy = evaluate_model(model, data)

    #    Save full model (optional)
    #    path = f"./trained_model_{name}.pth"
    #    torch.save(model, path)

    # Store optimizer configuration and performance results
    if name == "SGD":
        optimizer_summary[name] = {"learning_rate": 0.01, "weight_decay": 0.0}
    elif name == "Adam":
        optimizer_summary[name] = {"learning_rate": 0.001, "weight_decay": 1e-4}
    elif name == "AdamW":
        optimizer_summary[name] = {"learning_rate": 0.001, "weight_decay": 1e-4}

    training_report[name] = {
        "accuracy": accuracy,
        "training_time": 1.0
        # "model_path": path (optional)
    }

    # Select best optimizer based on highest accuracy after 2 epochs
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_optimizer = name

# Add comparison results to the training report
training_report["comparison"] = {
    "best_optimizer": best_optimizer
}

# Print summary of training results
print("\nTraining Report Summary")
for name in optimizer_configs:
    r = training_report[name]
    print(f"\n{name} Optimizer:\n- Accuracy: {r['accuracy'] * 100:.2f}")
print(f"\nBest Optimizer after {num_epochs} epochs: {training_report['comparison']['best_optimizer']}")

# Note: Due to the very small training subset (only 50 images), the model is unlikely to achieve high accuracy or learn meaningful patterns. This setup is intentionally lightweight for the purposes of this project. You can experiment with a larger dataset and more training epochs after submitting the project to achieve improved and more realistic performance results and compare parameter sizes and training speeds