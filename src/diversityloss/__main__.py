from diversityloss.diversity import Diversity, MEASURES
import torch


def main():
    counts = torch.tensor([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
    abundance = counts / counts.sum()
    similarity = torch.tensor(
        [
            [1.0, 0.5, 0.5, 0.7, 0.7, 0.7],
            [0.5, 1.0, 0.5, 0.7, 0.7, 0.7],
            [0.5, 0.5, 1.0, 0.7, 0.7, 0.7],
            [0.7, 0.7, 0.7, 1.0, 0.5, 0.5],
            [0.7, 0.7, 0.7, 0.5, 1.0, 0.5],
            [0.7, 0.7, 0.7, 0.5, 0.5, 1.0],
        ]
    )
    for measure in MEASURES.keys():
        diversity = Diversity(viewpoint=1.0, measure=measure[0], normalize=measure[1])
        D = diversity(abundance)
        print(measure, D)
    print()
    for measure in MEASURES.keys():
        diversity = Diversity(viewpoint=1.0, measure=measure[0], normalize=measure[1])
        D_Z = diversity(abundance, similarity)
        print(measure, D_Z)


if __name__ == "__main__":
    main()
