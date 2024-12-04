import torch
from matplotlib import pyplot as plt

from disorder.diversity import diversity


def main():
    torch.manual_seed(123)
    a = torch.rand(10, 1)
    a = a / a.sum()
    x = torch.rand(100, 32)
    dist = torch.cdist(x, x)
    similarity = torch.exp(-dist)
    viewpoints = torch.arange(0.98, 1.0, 0.00001)
    d = [
        diversity(
            abundance=a,
            similarity=similarity,
            viewpoint=viewpoints,
            measure="alpha",
            normalize=True,
        )
        for viewpoints in viewpoints
    ]
    plt.plot(viewpoints, d)
    plt.savefig("diversity.png")


if __name__ == "__main__":
    main()
