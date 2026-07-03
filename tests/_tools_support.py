from __future__ import annotations

from dataclasses import dataclass

from vercor.grid import RectilinearGrid
from vercor.types import RuntimeArray


@dataclass
class DummyComponentA:
    name: str = "a"


@dataclass
class DummyComponentB:
    name: str = "b"


@dataclass
class DummyGridComponent:
    grid: RectilinearGrid
    fields: dict[str, RuntimeArray]

    @property
    def data(self) -> dict[str, RuntimeArray]:
        return self.fields

    def get(self, field_name: str) -> RuntimeArray:
        if field_name not in self.fields:
            raise KeyError(field_name)
        return self.fields[field_name]
