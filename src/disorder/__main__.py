import torch

from disorder.community import metacommunity


def main():
    torch.manual_seed(123)
    a = torch.rand(10, 1)
    a = a / a.sum()
    x = torch.rand(10, 32)
    dist = torch.cdist(x, x)
    similarity = torch.exp(-dist)
    viewpoints = torch.arange(0.0, 10.0, 0.5)
    d = [
        metacommunity(
            abundance=a,
            similarity=similarity,
            viewpoint=viewpoint,
            measure="alpha",
            normalize=True,
        )
        for viewpoint in viewpoints
    ]
    print(d)


if __name__ == "__main__":
    main()
