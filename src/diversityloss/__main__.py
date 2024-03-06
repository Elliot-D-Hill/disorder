from diversityloss.diversity import (
    FrequencySensitiveDiversity,
    SimilaritySensitiveDiversity,
)
import torch
from diversityloss.diversity import FREQUENCY_MEASURES, SIMILARITY_MEASURES


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
    for measure in FREQUENCY_MEASURES.keys():
        diversity = FrequencySensitiveDiversity(
            viewpoint=1.0, measure=measure[0], normalize=measure[1]
        )
        d = diversity(abundance)
        print(measure, d)
    print()
    for measure in SIMILARITY_MEASURES.keys():
        diversity = SimilaritySensitiveDiversity(
            viewpoint=1.0, measure=measure[0], normalize=measure[1]
        )
        d = diversity(abundance, similarity)
        print(measure, d)


if __name__ == "__main__":
    main()
