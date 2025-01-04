import torch
import torch.nn.functional as F
from disorder.entropy import cross_entropy, entropy, relative_entropy
from hypothesis import given
from hypothesis import strategies as st
from torch.distributions import Categorical


@st.composite
def make_dimensions(draw):
    batch_size = draw(st.integers(min_value=1, max_value=10))
    num_classes = draw(st.integers(min_value=1, max_value=5))
    return batch_size, num_classes


@st.composite
def make_dimensions_with_features(draw):
    batch_size, num_classes = draw(make_dimensions())
    n_features = draw(st.integers(min_value=1, max_value=5))
    return batch_size, num_classes, n_features


@given(data=make_dimensions())
def test_shannon_entropy(data):
    batch_size, num_classes = data
    input = torch.rand(batch_size, num_classes)
    result = entropy(input, order=1.0, reduction="mean")
    expected = Categorical(logits=input).entropy().mean()
    torch.testing.assert_close(result, expected)


@given(data=make_dimensions())
def test_renyi_cross_entropy(data):
    batch_size, num_classes = data
    input = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    target = F.one_hot(target, num_classes=num_classes).float()
    similarity = torch.eye(num_classes)
    result1 = cross_entropy(input, target, order=1.0, similarity=None)
    result2 = cross_entropy(input, target, order=2.0, similarity=similarity)
    result3 = cross_entropy(input, target, order=2.0, similarity=None)
    expected = F.cross_entropy(input, target)
    torch.testing.assert_close(result1, expected)
    torch.testing.assert_close(result2, expected)
    torch.testing.assert_close(result3, expected)


@given(data=make_dimensions_with_features())
def test_leinster_lte_renyi(data):
    num_classes, batch_size, n_features = data
    input = torch.randn(batch_size, num_classes)
    target = torch.rand(batch_size, num_classes)
    features = torch.randn(num_classes, n_features)
    similarity = torch.exp(-torch.cdist(features, features))
    similarity = torch.eye(num_classes)
    leinster = cross_entropy(input, target, order=1.0, similarity=similarity)
    renyi = torch.nn.functional.cross_entropy(input, target)
    assert ((leinster < renyi) | torch.isclose(leinster, renyi)).all()


@given(data=make_dimensions())
def test_renyi_relative_entropy(data):
    batch_size, num_classes = data
    input = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    target = F.one_hot(target, num_classes=num_classes).float()
    similarity = torch.eye(num_classes)
    result1 = relative_entropy(input, target, order=1.0, similarity=None)
    result2 = relative_entropy(input, target, order=1.0, similarity=similarity)
    result3 = relative_entropy(input, target, order=2.0, similarity=None)
    expected = torch.nn.functional.kl_div(input, target, reduction="batchmean")
    torch.testing.assert_close(result1, expected)
    torch.testing.assert_close(result2, expected)
    torch.testing.assert_close(result3, expected)
