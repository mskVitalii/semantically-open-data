from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class AsyncBuffer(ABC, Generic[T]):
    @abstractmethod
    async def add(self, item: T) -> None:
        pass

    @abstractmethod
    async def flush(self) -> int:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass

    @property
    @abstractmethod
    def total_indexed(self) -> int:
        pass

    @property
    @abstractmethod
    async def size(self) -> int:
        pass
