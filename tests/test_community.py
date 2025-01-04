# from pytest import mark
import pytest
import torch
from conftest import COMMUNITY_CASES, CommunityTestCase
from disorder.community import Metacommunity
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=case.id) for case in COMMUNITY_CASES]
)
def test_diversity(case: CommunityTestCase):
    diversity = Metacommunity(
        viewpoint=case.viewpoint, measure=case.measure, normalize=case.normalize
    )
    torch.testing.assert_close(
        diversity(case.abundance, case.similarity), torch.tensor(case.expected)
    )


@st.composite
def abundance_similarity_strategy(draw):
    n_species = draw(st.integers(min_value=1, max_value=8))
    m_subcommunities = draw(st.integers(min_value=1, max_value=8))
    viewpoint = draw(st.floats(min_value=0.0, max_value=10.0))
    abundance = draw(
        arrays(
            float,
            (n_species, m_subcommunities),
            elements=st.floats(0.0001, 1, allow_infinity=False, allow_nan=False),
        ).map(lambda x: x / x.sum())
    )
    similarity = draw(
        arrays(
            float,
            (n_species, n_species),
            elements=st.floats(0, 1, allow_infinity=False, allow_nan=False),
        )
    )
    abundance = torch.tensor(abundance)
    similarity = torch.tensor(similarity)
    similarity.fill_diagonal_(1.0)
    return viewpoint, abundance, similarity


@given(abundance_similarity_strategy())
def test_similarity_lte_frequency(data):
    viewpoint, abundance, similarity = data
    abundance = torch.as_tensor(abundance)
    similarity = torch.as_tensor(similarity)
    diversity = Metacommunity(viewpoint=viewpoint, measure="alpha", normalize=True)
    frequency_sensitive = diversity(abundance)
    similarity_sensitive = diversity(abundance, similarity)
    assert similarity_sensitive <= frequency_sensitive


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
    diversity_a = Metacommunity(viewpoint=viewpoint_a, measure="alpha", normalize=True)
    diversity_b = Metacommunity(viewpoint=viewpoint_b, measure="alpha", normalize=True)
    diversity_a_value = diversity_a(abundance, similarity)
    diversity_b_value = diversity_b(abundance, similarity)
    assert (diversity_a_value > diversity_b_value) or torch.isclose(
        diversity_a_value, diversity_b_value
    )
