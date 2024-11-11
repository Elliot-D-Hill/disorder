import numpy as np
import torch
from diversityloss.diversity import Diversity
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pytest import mark

n_tests = 10

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
# args: abundance, viewpoint, measure, normalize, similarity, expected
argvalues = [
    (abundance_6by2, 0.0, "alpha", False, None, 6.0),
    (abundance_6by2, 1.0, "rho", False, None, 1.0),
    (abundance_6by2, 2.0, "beta", False, None, 1.0),
    (abundance_6by2, 3.0, "gamma", False, None, 6.0),
    (abundance_6by2, 4.0, "alpha", True, None, 3.0),
    (abundance_6by2, 5.0, "rho", True, None, 0.5),
    (abundance_6by2, 6.0, "beta", True, None, 2.0),
    (abundance_6by2, 7.0, "gamma", True, None, 6.0),
    (abundance_6by2, 8.0, "alpha", False, similarity_6by2, 3.0),
    (abundance_6by2, 9.0, "rho", False, similarity_6by2, 2.05),
    (abundance_6by2, 10.0, "beta", False, similarity_6by2, 0.487805),
    (abundance_6by2, 11.0, "gamma", True, similarity_6by2, 1.463415),
    (abundance_6by2, float("inf"), "alpha", True, similarity_6by2, 1.5),
    (abundance_6by2, float("-inf"), "rho", True, similarity_6by2, 1.025),
    (abundance_6by2, float("inf"), "beta", True, similarity_6by2, 0.97561),
    (abundance_6by2, float("-inf"), "gamma", True, similarity_6by2, 1.463415),
    (abundance_3by2, 2.0, "alpha", False, None, 2.7777777777777777),
    (abundance_3by2, 2.0, "rho", False, None, 1.2),
    (abundance_3by2, 2.0, "beta", False, None, 0.8319209039548022),
    (abundance_3by2, 2.0, "gamma", False, None, 2.173913043478261),
    (abundance_3by2, 2.0, "alpha", True, None, 1.4634146341463414),
    (abundance_3by2, 2.0, "rho", True, None, 0.6050420168067228),
    (abundance_3by2, 2.0, "beta", True, None, 1.612461673236969),
    (abundance_3by2, 2.0, "alpha", False, similarity_3by2, 2.5),
    (abundance_3by2, 2.0, "rho", False, similarity_3by2, 1.6502801833927663),
    (abundance_3by2, 2.0, "beta", False, similarity_3by2, 0.5942352817544037),
    (abundance_3by2, 2.0, "gamma", False, similarity_3by2, 1.5060240963855422),
    (abundance_3by2, 2.0, "alpha", True, similarity_3by2, 1.2903225806451613),
    (abundance_3by2, 2.0, "rho", True, similarity_3by2, 0.8485572790897555),
    (abundance_3by2, 2.0, "beta", True, similarity_3by2, 1.1744247216675028),
]


@mark.parametrize(
    argnames="abundance, viewpoint, measure, normalize, similarity, expected",
    argvalues=argvalues,
)
def test_diversity(abundance, viewpoint, measure, normalize, similarity, expected):
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
            elements=st.floats(0.0001, 1, allow_infinity=False, allow_nan=False),
        ).map(lambda x: x / x.sum())
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
def test_similarity_less_than_frequency(data):
    viewpoint, abundance, similarity = data
    abundance = torch.as_tensor(abundance)
    similarity = torch.as_tensor(similarity)
    diversity = Diversity(viewpoint=viewpoint, measure="alpha", normalize=True)
    frequency_sensitive = diversity(abundance)
    similarity_sensitive = diversity(abundance, similarity)
    assert (similarity_sensitive < frequency_sensitive) or torch.isclose(
        similarity_sensitive, frequency_sensitive
    )


@st.composite
def abundance_similarity_viewpoint_strategy(draw):
    data = draw(abundance_similarity_strategy())
    viewpoint_a = draw(st.floats(min_value=0.0, max_value=10.0))
    viewpoint_b = viewpoint_a + draw(st.floats(min_value=0.1, max_value=5.0))
    return viewpoint_a, viewpoint_b, data[1], data[2]


@given(abundance_similarity_viewpoint_strategy())
def test_diversity_monotonicity(data):
    viewpoint_a, viewpoint_b, abundance, similarity = data
    abundance = torch.as_tensor(abundance)
    similarity = torch.as_tensor(similarity)
    diversity_a = Diversity(viewpoint=viewpoint_a, measure="alpha", normalize=True)
    diversity_b = Diversity(viewpoint=viewpoint_b, measure="alpha", normalize=True)
    diversity_a_value = diversity_a(abundance, similarity)
    diversity_b_value = diversity_b(abundance, similarity)
    assert (diversity_a_value > diversity_b_value) or torch.isclose(
        diversity_a_value, diversity_b_value
    )
