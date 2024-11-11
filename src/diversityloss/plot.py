import matplotlib.pyplot as plt
import torch

from diversityloss.diversity import diversity

if __name__ == "__main__":
    torch.manual_seed(123)
    a = torch.rand(100, 16)
    a = a / a.sum()
    x = torch.rand(100, 32)
    dist = torch.cdist(x, x)
    sim = torch.exp(-dist)
    viewpoints = torch.arange(0.9, 1.1, 0.00001)
    d = [
        diversity(
            abundance=a,
            # similarity=sim,
            viewpoint=viewpoints,
            measure="alpha",
            normalize=True,
        )
        for viewpoints in viewpoints
    ]
    plt.plot(viewpoints, d)
    plt.savefig("diversity.png")
