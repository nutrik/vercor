class CouplerError(Exception):
    """Base class for exceptions inside Coupler."""

    pass


class ComponentError(CouplerError):
    """Base class for exceptions inside individual components."""

    pass


class RegridderError(CouplerError):
    """Base class for exceptions during regridding operations."""

    pass


class ExchangerError(CouplerError):
    """Base class for exceptions during data exchange between components."""

    pass
