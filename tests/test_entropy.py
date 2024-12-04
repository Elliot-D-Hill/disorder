import pytest
import torch

from disorder.diversity import weighted_power_mean
from disorder.entropy import cross_entropy, entropy, relative_entropy

torch.manual_seed(0)


@pytest.fixture(params=[(6, 2, 3), (3, 3, 2)])
def setup_data(request):
    n, d, q = request.param
    p = torch.rand(n)
    p /= p.sum()
    r = torch.rand(n)
    r /= r.sum()
    x = torch.rand(n, d)
    z = torch.exp(-torch.cdist(x, x))
    return p, r, q, z


def test_entropy(setup_data):
    p, _, q, z = setup_data
    zp = z @ p
    result = entropy(p, q, z=z)
    expected = weighted_power_mean(items=1.0 / zp, weights=zp, order=1 - q).log()
    print("result:", result, "expected:", expected)
    torch.testing.assert_close(result, expected)


def test_cross_entropy(setup_data):
    p, r, q, z = setup_data
    result = cross_entropy(p, r, q, z=z)
    expected = weighted_power_mean(items=1.0 / (z @ r), weights=p, order=1.0 - q).log()
    torch.testing.assert_close(result, expected)


def test_relative_entropy(setup_data):
    p, r, q, z = setup_data
    zp = z @ p
    zr = z @ r
    result = relative_entropy(p, r, q, z=z)
    expected = weighted_power_mean(items=zp / zr, weights=p, order=q - 1.0).log()
    torch.testing.assert_close(result, expected)
