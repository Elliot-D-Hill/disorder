from dataclasses import dataclass

import pytest
import torch


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(0)
    yield


counts_6by2 = torch.tensor(
    [
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
    ]
)
abundance_6by2 = counts_6by2 / counts_6by2.sum()
similarity_6by2 = torch.tensor(
    [
        [1.0, 0.5, 0.5, 0.7, 0.7, 0.7],
        [0.5, 1.0, 0.5, 0.7, 0.7, 0.7],
        [0.5, 0.5, 1.0, 0.7, 0.7, 0.7],
        [0.7, 0.7, 0.7, 1.0, 0.5, 0.5],
        [0.7, 0.7, 0.7, 0.5, 1.0, 0.5],
        [0.7, 0.7, 0.7, 0.5, 0.5, 1.0],
    ]
)
counts_3by2 = torch.tensor(
    [
        [1, 5],
        [3, 0],
        [0, 1],
    ]
)
abundance_3by2 = counts_3by2 / counts_3by2.sum()
similarity_3by2 = torch.tensor(
    [
        [1.0, 0.5, 0.1],
        [0.5, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ]
)


@dataclass
class CommunityTestCase:
    expected: float
    abundance: torch.Tensor = abundance_6by2
    viewpoint: float = 1.0
    measure: str = "alpha"
    normalize: bool = False
    similarity: torch.Tensor | None = None
    id: str | None = None

    def __post_init__(self):
        similarity = None if self.similarity is None else tuple(self.similarity.size())
        self.id = (
            f"{tuple(self.abundance.size())}_{self.measure}"
            f"_{self.viewpoint}_{self.normalize}_{similarity}"
        )


COMMUNITY_CASES = [
    CommunityTestCase(6.0, viewpoint=0.0),
    CommunityTestCase(6.0, viewpoint=1.0001),
    CommunityTestCase(1.0, measure="rho"),
    CommunityTestCase(1.0, viewpoint=2.0, measure="beta"),
    CommunityTestCase(6.0, viewpoint=3.0, measure="gamma"),
    CommunityTestCase(3.0, viewpoint=4.0, normalize=True),
    CommunityTestCase(0.5, viewpoint=5.0, measure="rho", normalize=True),
    CommunityTestCase(2.0, viewpoint=6.0, measure="beta", normalize=True),
    CommunityTestCase(6.0, viewpoint=7.0, measure="gamma", normalize=True),
    CommunityTestCase(3.0, viewpoint=8.0, similarity=similarity_6by2),
    CommunityTestCase(2.05, viewpoint=9.0, measure="rho", similarity=similarity_6by2),
    CommunityTestCase(
        0.487805, viewpoint=10.0, measure="beta", similarity=similarity_6by2
    ),
    CommunityTestCase(
        1.463415,
        viewpoint=11.0,
        measure="gamma",
        normalize=True,
        similarity=similarity_6by2,
    ),
    CommunityTestCase(
        1.5, viewpoint=float("inf"), normalize=True, similarity=similarity_6by2
    ),
    CommunityTestCase(
        1.025,
        viewpoint=float("-inf"),
        measure="rho",
        normalize=True,
        similarity=similarity_6by2,
    ),
    CommunityTestCase(
        0.97561,
        viewpoint=float("inf"),
        measure="beta",
        normalize=True,
        similarity=similarity_6by2,
    ),
    CommunityTestCase(
        1.463415,
        viewpoint=float("-inf"),
        measure="gamma",
        normalize=True,
        similarity=similarity_6by2,
    ),
    CommunityTestCase(2.7777777777777777, abundance=abundance_3by2, viewpoint=2.0),
    CommunityTestCase(1.2, abundance=abundance_3by2, viewpoint=2.0, measure="rho"),
    CommunityTestCase(
        0.8319209039548022, abundance=abundance_3by2, viewpoint=2.0, measure="beta"
    ),
    CommunityTestCase(
        2.173913043478261, abundance=abundance_3by2, viewpoint=2.0, measure="gamma"
    ),
    CommunityTestCase(
        1.4634146341463414, abundance=abundance_3by2, viewpoint=2.0, normalize=True
    ),
    CommunityTestCase(
        0.6050420168067228,
        abundance=abundance_3by2,
        viewpoint=2.0,
        measure="rho",
        normalize=True,
    ),
    CommunityTestCase(
        1.612461673236969,
        abundance=abundance_3by2,
        viewpoint=2.0,
        measure="beta",
        normalize=True,
    ),
    CommunityTestCase(
        2.5, abundance=abundance_3by2, viewpoint=2.0, similarity=similarity_3by2
    ),
    CommunityTestCase(
        1.6502801833927663,
        abundance=abundance_3by2,
        viewpoint=2.0,
        measure="rho",
        similarity=similarity_3by2,
    ),
    CommunityTestCase(
        0.5942352817544037,
        abundance=abundance_3by2,
        viewpoint=2.0,
        measure="beta",
        similarity=similarity_3by2,
    ),
    CommunityTestCase(
        1.5060240963855422,
        abundance=abundance_3by2,
        viewpoint=2.0,
        measure="gamma",
        similarity=similarity_3by2,
    ),
    CommunityTestCase(
        1.2903225806451613,
        abundance=abundance_3by2,
        viewpoint=2.0,
        normalize=True,
        similarity=similarity_3by2,
    ),
    CommunityTestCase(
        0.8485572790897555,
        abundance=abundance_3by2,
        viewpoint=2.0,
        measure="rho",
        normalize=True,
        similarity=similarity_3by2,
    ),
    CommunityTestCase(
        1.1744247216675028,
        abundance=abundance_3by2,
        viewpoint=2.0,
        measure="beta",
        normalize=True,
        similarity=similarity_3by2,
    ),
]
