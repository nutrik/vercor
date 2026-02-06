from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator


@dataclass
class Clock:
    start: datetime
    dt_seconds: float
    steps: int

    def iter(self) -> Iterator[tuple[int, datetime, timedelta]]:
        """
        Iterator over simulation time steps.

        Yields:
            Tuple of (step index, current time, time delta)
        """
        time = self.start
        dt = timedelta(seconds=self.dt_seconds)
        for n in range(self.steps):
            yield n, time, dt
            time += dt
