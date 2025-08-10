from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List

T = TypeVar("T")


class AsyncBuffer(ABC, Generic[T]):
    @abstractmethod
    async def add(self, item: T) -> None:
        pass

    @abstractmethod
    async def add_batch(self, items: List[T]) -> None:
        pass

    @abstractmethod
    async def flush(self) -> int:
        pass

    @property
    @abstractmethod
    def total_indexed(self) -> int:
        pass
