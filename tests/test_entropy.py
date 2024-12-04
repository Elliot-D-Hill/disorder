import torch
from disorder.diversity import weighted_power_mean
from disorder.entropy import cross_entropy, entropy, relative_entropy
from hypothesis import given
from hypothesis import strategies as st


@st.composite
def setup_data(draw):
    n = draw(st.integers(min_value=2, max_value=10))
    m = draw(st.integers(min_value=1, max_value=5))
    q = draw(st.sampled_from([0.0, 0.5, 1.0, 1.5, 2.0, float("inf")]))
    p = torch.rand(n, m)
    p /= p.sum()
    r = torch.rand(n, m)
    r /= r.sum()
    if draw(st.booleans()):
        d = draw(st.integers(min_value=1, max_value=5))
        x = torch.rand(n, d)
        z = torch.exp(-torch.cdist(x, x))
    else:
        z = None
    return p, r, q, z


@given(data=setup_data())
def test_entropy(data):
    p, _, q, z = data
    zp = p if z is None else z @ p
    is_zero = p < 1e-8
    zp = zp.masked_fill(is_zero, 1.0)
    expected = weighted_power_mean(p=p, x=1.0 / zp, t=1.0 - q).log()
    result = entropy(p=p, q=q, z=z)
    torch.testing.assert_close(result, expected)


@given(data=setup_data())
def test_cross_entropy(data):
    p, r, q, z = data
    result = cross_entropy(p=p, r=r, q=q, z=z)
    zr = r if z is None else z @ r
    is_zero = p < 1e-8
    zr = zr.masked_fill(is_zero, 1.0)
    expected = weighted_power_mean(p=p, x=1.0 / zr, t=1.0 - q).log()
    print("q:", q, "z", z is None, "result:", result, "expected:", expected)
    torch.testing.assert_close(result, expected)


@given(data=setup_data())
def test_relative_entropy(data):
    p, r, q, z = data
    zp = p if z is None else z @ p
    zr = r if z is None else z @ r
    result = relative_entropy(p=p, r=r, q=q, z=z)
    expected = weighted_power_mean(p=p, x=zp / zr, t=q - 1.0).log()
    torch.testing.assert_close(result, expected)
