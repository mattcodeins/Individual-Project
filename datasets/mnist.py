import torchvision.datasets as d
import torchvision.transforms as t
import torch
import matplotlib.pyplot as plt


def import_n_mnist(batch_size_train, batch_size_test):
    """
    Import MNIST normalised.
    """
    train_loader = torch.utils.data.DataLoader(
        d.MNIST('datasets/', train=True, download=True,
                transform=t.Compose([
                    t.ToTensor(),
                    t.Normalize(
                        (0.1307,), (0.3081,))
                ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        d.MNIST('datasets/', train=False, download=True,
                transform=t.Compose([
                    t.ToTensor(),
                    t.Normalize(
                        (0.1307,), (0.3081,))
                ])),
        batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader


def plot_examples(test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
