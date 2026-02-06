from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class RunSequence:
    # Ordered component names for stepping
    order: list[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[str]:
        return iter(self.order)
