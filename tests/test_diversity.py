import torch
from pytest import mark
from hypothesis import strategies as st, given
from hypothesis.extra.numpy import arrays
import numpy as np

from diversityloss.diversity import Diversity

n_tests = 10

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
argvalues = [
    (0.0, "alpha", False, None, 6.0),
    (1.0, "rho", False, None, 1.0),
    (2.0, "beta", False, None, 1.0),
    (3.0, "gamma", False, None, 6.0),
    (4.0, "alpha", True, None, 3.0),
    (5.0, "rho", True, None, 0.5),
    (6.0, "beta", True, None, 2.0),
    (7.0, "gamma", True, None, 6.0),
    (8.0, "alpha", False, similarity, 3.0),
    (9.0, "rho", False, similarity, 2.05),
    (10.0, "beta", False, similarity, 0.487805),
    (11.0, "gamma", True, similarity, 1.463415),
    (12.0, "alpha", True, similarity, 1.5),
    (13.0, "rho", True, similarity, 1.025),
    (14.0, "beta", True, similarity, 0.97561),
    (15.0, "gamma", True, similarity, 1.463415),
]


@mark.parametrize(
    argnames="viewpoint, measure, normalize, similarity, expected", argvalues=argvalues
)
def test_diversity(viewpoint, measure, normalize, similarity, expected):
    diversity = Diversity(viewpoint=viewpoint, measure=measure, normalize=normalize)
    assert torch.isclose(diversity(abundance, similarity), torch.tensor(expected))


@st.composite
def abundance_similarity_strategy(draw):
    n_species = draw(st.integers(min_value=1, max_value=8))
    m_subcommunities = draw(st.integers(min_value=1, max_value=8))
    viewpoint = draw(st.floats(min_value=0.0, max_value=10.0))
    abundance = draw(
        arrays(
            np.float64,
            (n_species, m_subcommunities),
            elements=st.floats(0.001, 1, allow_infinity=False, allow_nan=False),
        ).map(
            lambda x: x / x.sum()
        )  # Ensure the array sums to 1
    )
    similarity = draw(
        arrays(
            np.float64,
            (n_species, n_species),
            elements=st.floats(0, 1, allow_infinity=False, allow_nan=False),
        )
    )
    np.fill_diagonal(similarity, 1.0)
    return viewpoint, abundance, similarity


@given(abundance_similarity_strategy())
def test_similarity_le_frequency(data):
    viewpoint, abundance, similarity = data
    abundance = torch.as_tensor(abundance)
    similarity = torch.as_tensor(similarity)
    diversity = Diversity(viewpoint=viewpoint, measure="alpha", normalize=True)
    frequency_sensitive = diversity(abundance)
    similarity_sensitive = diversity(abundance, similarity)
    assert (similarity_sensitive < frequency_sensitive) or torch.isclose(
        similarity_sensitive, frequency_sensitive
    )
