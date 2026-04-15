import warnings

# Suppress cutlass DSL DeprecationWarning about struct pointer access.
# The warning originates inside the cutlass library (core.py) and is not
# actionable from user code; it will be resolved when we migrate to the
# new `struct.scalar.ptr` API in a future cutlass version.
warnings.filterwarnings(
    "ignore",
    message=r"Use explicit `struct\.scalar\.ptr` for pointer instead\.",
    category=DeprecationWarning,
)
