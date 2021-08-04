from typing import Iterable, Optional


class Tensor:
    def __new__(cls, shape: Iterable[int], data: Optional[Iterable[float]]):
        ...
