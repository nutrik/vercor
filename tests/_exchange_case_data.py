"""Independent equal-area coastal expectations shared by exchange and swap tests."""

QUANTITY_CASES = (("intensive", 8.0), ("source_flux", 4.0), ("target_flux", 2.0))
# d(mean received)/d(each source fraction), target-fraction derivative.
COVERAGE_GRADIENT_CASES = (
    ("intensive", [-3.0, 1.0], 0.0),
    ("source_flux", [0.5, 2.5], 0.0),
    ("target_flux", [-0.75, 0.25], 2.0),
)
# d(mean received)/d(each source value).
FIELD_GRADIENT_CASES = (
    ("intensive", [0.125, 0.375]),
    ("source_flux", [0.0625, 0.1875]),
    ("target_flux", [0.03125, 0.09375]),
)
