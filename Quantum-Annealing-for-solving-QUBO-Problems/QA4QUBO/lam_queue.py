from collections import deque


class LamQueue:
    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n: int >= 0")

        self.n = n
        self._data = deque(maxlen=n)

    def enqueue(self, value: float) -> None:
        if self.is_full():
            pop = self.dequeue()
        
        self._data.append(value)

    def dequeue(self) -> float:
        if self.is_empty():
            raise ValueError("Queue is empty!")

        return self._data.popleft()

    def is_full(self) -> bool:
        return len(self._data) == self.n

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def size(self) -> int:
        return len(self._data)

    def avg(self) -> float:
        if self.is_empty():
            raise ValueError("Queue is empty!")

        return sum(self._data) / len(self._data)

    def to_list(self) -> list:
        return list(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"LamQueue(n={self.n}, data={list(self._data)})"