import pytest

from mytensor import Tensor


@pytest.mark.parametrize(
    ["shape", "size"], [
        ([], 0),
        ([0], 0),
        ([13], 13),
        ([4, 5], 20),
    ]
)
def test_tensor_size(shape, size):
    assert Tensor(shape).size() == size


def test_tensor_getitem():
    values = list(range(24))
    Tensor([4, 6], values)[2, 3].flatten() == values[2 * 6 + 3]


def test_tensor_setitem():
    t = Tensor([4, 6], list(range(24)))
    new_val = 42.
    t[2, 3] = new_val
    assert t[2, 3].flatten() == [new_val]


def test_views():
    t = Tensor([4, 6], list(range(24)))
    s = t[2, 3]
    new_val = 42.
    s[0,] = new_val
    assert t[2, 3].flatten() == [new_val]
