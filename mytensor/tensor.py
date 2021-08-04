from typing import Iterable

from .lib import Tensor as BaseTensor


class Tensor(BaseTensor):
    def __new__(cls, shape, data=None):
        if data and not isinstance(data, Iterable):
            data = (data,)
        if isinstance(data, Iterable):
            return BaseTensor(shape, [float(x) for x in data])
        return BaseTensor(shape)

    def __getitem__(self, idx):
        if isinstance(idx, Iterable):
            return super().__getitem__(idx)
        return super().__getitem__((idx,))

    def __setitem__(self, idx, val):
        if isinstance(idx, Iterable):
            super().__setitem__(idx, val)
        super().__setitem__((idx,), val)
