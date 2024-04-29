import fire
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from examples.data_loader import MnistDataLoader
from pytagi import HRCSoftmaxMetric
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    LayerNorm,
    Linear,
    OutputUpdater,
    ReLU,
    Sequential,
)

FNN = Sequential(
    Linear(784, 100),
    ReLU(),
    Linear(100, 100),
    ReLU(),
    Linear(100, 11),
)

FNN_BATCHNORM = Sequential(
    Linear(784, 100),
    ReLU(),
    BatchNorm2d(100),
    Linear(100, 100),
    ReLU(),
    BatchNorm2d(100),
    Linear(100, 11),
)

FNN_LAYERNORM = Sequential(
    Linear(784, 100, bias=False),
    ReLU(),
    LayerNorm((100,)),
    Linear(100, 100, bias=False),
    ReLU(),
    LayerNorm((100,)),
    Linear(100, 11),
)

CNN = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28),
    ReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5),
    ReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100),
    ReLU(),
    Linear(100, 11),
)

CNN_BATCHNORM = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False),
    ReLU(),
    BatchNorm2d(16),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False),
    ReLU(),
    BatchNorm2d(32),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100),
    ReLU(),
    Linear(100, 11),
)

CNN_LAYERNORM = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False),
    LayerNorm((16, 27, 27)),
    ReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False),
    LayerNorm((32, 9, 9)),
    ReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100),
    ReLU(),
    Linear(100, 11),
)

def plot_accuracy(accuracies_train=None, accuracies_valid=None, save_path=None):
    assert accuracies_train is not None or accuracies_valid is not None, "Cannot plot because neither training nor validation accuracies provided."

    plt.figure(figsize=(8, 6))

    if accuracies_train:
        plt.plot(accuracies_train, label="Training")
    if accuracies_valid:
        plt.plot(accuracies_valid, label="Validation")

    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_images_per_label(images: np.ndarray, labels: np.ndarray):
    """
    Visualize a grid of images for each label.

    Parameters:
    - images: numpy.ndarray, array of shape (num_images, 28, 28) containing the images
    - labels: numpy.ndarray, array of shape (num_images,) containing the corresponding labels
    """
    # Create a list to store images for each label
    label_images = [[] for _ in range(10)]

    # Iterate through images and organize them by label
    for img, lbl in zip(images, labels):
        # Convert the image array to a list
        img_list = img.tolist()
        label_images[lbl].append(img_list)

    # Plot images for each label in a 2 row, 5 column grid
    plt.figure(figsize=(15, 6))
    for i in range(2):  # Iterate over rows
        for j in range(5):  # Iterate over columns
            label_idx = i * 5 + j  # Calculate the label index
            if label_idx < 10:  # Check if the label index is within the range
                images = label_images[label_idx]
                for k in range(10):  # Display 10 images per label
                    if k < len(images):  # Check if there are enough images for the label
                        plt.subplot(2, 5, i * 5 + j + 1)  # Subplot position
                        img_array = np.array(images[k])
                        plt.imshow(img_array, cmap='gray')
                        plt.axis('off')
                    else:  # If not, leave the subplot empty
                        plt.subplot(2, 5, i * 5 + j + 1)
                        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main(num_epochs: int = 10, batch_size: int = 512, sigma_v: float = 2.0):
    """
    Run classification training on the MNIST dataset using a custom neural model.

    Parameters:
    - num_epochs: int, number of epochs for training
    - batch_size: int, size of the batch for training
    """
    # Load dataset
    train_dtl = MnistDataLoader(
        x_file="data/mnist/train-images-idx3-ubyte",
        y_file="data/mnist/train-labels-idx1-ubyte",
        num_images=60000,
    )
    test_dtl = MnistDataLoader(
        x_file="data/mnist/t10k-images-idx3-ubyte",
        y_file="data/mnist/t10k-labels-idx1-ubyte",
        num_images=10000,
    )

    # Extract images and labels from train_dtl
    print("Visualizing images per label before training...")
    mnist_images, _, _, mnist_labels = train_dtl.dataset["value"]
    visualize_images_per_label(mnist_images, mnist_labels)

    # Hierachical Softmax
    metric = HRCSoftmaxMetric(num_classes=10)

    # Network configuration
    net = CNN
    # net.to_device("cuda")
    # net.set_threads(16)
    out_updater = OutputUpdater(net.device)

    # Training
    error_rates = []
    accuracy_train = []
    accuracy_test = []
    var_y = np.full(
        (batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32
    )
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size=batch_size)
        for x, y, y_idx, label in batch_iter:
            # Feedforward and backward pass
            m_pred, v_pred = net(x)

            # Update output layers based on targets
            out_updater.update_using_indices(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                selected_idx=y_idx,
                delta_states=net.input_delta_z_buffer,
            )

            # Update parameters
            net.backward()
            net.step()

            # Training metric
            error_rate = metric.error_rate(m_pred, v_pred, label)
            error_rates.append(error_rate)

        # Averaged error
        avg_error_rate = sum(error_rates[-100:])
        train_error_rate = sum(error_rates) / len(error_rates)
        accuracy_train.append((1 - train_error_rate))

        # Testing
        test_error_rates = []
        test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=False)
        for x, _, _, label in test_batch_iter:
            m_pred, v_pred = net(x)

            # Training metric
            error_rate = metric.error_rate(m_pred, v_pred, label)
            test_error_rates.append(error_rate)

        test_error_rate = sum(test_error_rates) / len(test_error_rates)
        accuracy_test.append(1-test_error_rate)

        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | training error: {avg_error_rate:.2f}% | test error: {test_error_rate * 100:.2f}%",
            refresh=True,
        )
    print("Training complete.")

    plot_accuracy(accuracies_train=accuracy_train, accuracies_valid=accuracy_test)

if __name__ == "__main__":
    fire.Fire(main)
