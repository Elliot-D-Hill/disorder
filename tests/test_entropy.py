import torch
from disorder.entropy import cross_entropy, entropy, relative_entropy
from disorder.mean import weighted_power_mean
from hypothesis import given
from hypothesis import strategies as st


@st.composite
def setup_data(draw):
    n = draw(st.integers(min_value=2, max_value=10))
    m = draw(st.integers(min_value=1, max_value=5))
    alpha = draw(
        st.sampled_from([float("-inf"), 0.0, 0.5, 1.0, 1.5, 2.0, float("inf")])
    )
    p = torch.rand(n, m)
    p /= p.sum()
    q = torch.rand(n, m)
    q /= q.sum()
    if draw(st.booleans()):
        d = draw(st.integers(min_value=1, max_value=5))
        x = torch.rand(n, d)
        z = torch.exp(-torch.cdist(x, x))
    else:
        z = None
    return p, q, alpha, z


@given(data=setup_data())
def test_entropy(data):
    p, _, alpha, z = data
    zp = p if z is None else z @ p
    zp.masked_fill_(p < 1e-8, 1.0)
    result = entropy(p=p, alpha=alpha, z=z)
    expected = weighted_power_mean(p=p, x=1.0 / zp, t=1.0 - alpha).log()
    torch.testing.assert_close(result, expected)


@given(data=setup_data())
def test_cross_entropy(data):
    p, q, alpha, z = data
    result = cross_entropy(p=p, q=q, alpha=alpha, z=z)
    zq = q if z is None else z @ q
    is_zero = p < 1e-8
    zq = zq.masked_fill_(is_zero, 1.0)
    expected = weighted_power_mean(p=p, x=1.0 / zq, t=1.0 - alpha).log()
    torch.testing.assert_close(result, expected)


@given(data=setup_data())
def test_relative_entropy(data):
    p, q, alpha, z = data
    zp = p if z is None else z @ p
    zq = q if z is None else z @ q
    zq.masked_fill_(p < 1e-8, 1.0)
    result = relative_entropy(p=p, q=q, alpha=alpha, z=z)
    expected = weighted_power_mean(p=p, x=zp / zq, t=alpha - 1.0).log()
    torch.testing.assert_close(result, expected)
