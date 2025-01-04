from dataclasses import dataclass

import torch
import torch.nn.functional as F
from disorder.entropy import cross_entropy, entropy, relative_entropy
from hypothesis import given
from hypothesis import strategies as st
from torch.distributions import Categorical


@dataclass
class EntropyCase:
    input: torch.Tensor
    target: torch.Tensor
    similarity: torch.Tensor | None = None


@st.composite
def make_renyi_case(draw):
    batch_size = draw(st.integers(min_value=1, max_value=10))
    num_classes = draw(st.integers(min_value=1, max_value=5))
    input = torch.randn(batch_size, num_classes)
    label = torch.randint(0, num_classes, (batch_size,))
    target = F.one_hot(label, num_classes=num_classes).float()
    similarity = torch.eye(num_classes)
    return EntropyCase(input, target, similarity=similarity)


@st.composite
def make_leinster_case(draw):
    test_case: EntropyCase = draw(make_renyi_case())
    n_features = draw(st.integers(min_value=1, max_value=5))
    num_classes = test_case.input.size(1)
    features = torch.randn(num_classes, n_features)
    similarity = torch.exp(-torch.cdist(features, features))
    test_case.similarity = similarity
    return test_case


@given(data=make_renyi_case())
def test_shannon_entropy(data: EntropyCase):
    result = entropy(data.input, order=1.0)
    expected = Categorical(logits=data.input).entropy().mean()
    torch.testing.assert_close(result, expected)


@given(data=make_renyi_case())
def test_renyi_cross_entropy(data: EntropyCase):
    result1 = cross_entropy(data.input, data.target, order=1.0, similarity=None)
    result2 = cross_entropy(
        data.input, data.target, order=1.0, similarity=data.similarity
    )
    expected = F.cross_entropy(data.input, data.target)
    torch.testing.assert_close(result1, expected)
    torch.testing.assert_close(result2, expected)


@given(data=make_leinster_case())
def test_leinster_lte_renyi(data: EntropyCase):
    leinster = cross_entropy(
        data.input, data.target, order=1.0, similarity=data.similarity
    )
    renyi = torch.nn.functional.cross_entropy(data.input, data.target)
    assert ((leinster < renyi) | torch.isclose(leinster, renyi)).all()


@given(data=make_renyi_case())
def test_renyi_relative_entropy(data: EntropyCase):
    result1 = relative_entropy(data.input, data.target, order=1.0, similarity=None)
    result2 = relative_entropy(
        data.input, data.target, order=1.0, similarity=data.similarity
    )
    expected = torch.nn.functional.kl_div(
        data.input, data.target, reduction="batchmean"
    )
    torch.testing.assert_close(result1, expected)
    torch.testing.assert_close(result2, expected)
