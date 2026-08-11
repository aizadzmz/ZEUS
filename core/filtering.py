def mask_inductive_points(dataset) -> None: #remove inductive tail
    """Mask points with a positive imaginary impedance (inductive artifacts),
    in place."""
    Z = dataset.data.get_impedances(masked=None)  # all points, incl. masked
    dataset.data.set_mask({i: bool(z.imag > 0) for i, z in enumerate(Z)})


def clear_mask(dataset) -> None: #re-add the inductive points
    """Unmask all points of an EISDataset (in place)."""
    dataset.data.set_mask({})


def detached_copy(dataset):
    """A copy of an EISDataset that no longer shares its point data, so masking
    one leaves the other alone.

    The copy keeps the original's index and file, and so its key: results
    computed from it still file under the sweep it came from."""
    from copy import deepcopy

    from core.io_utils import EISDataset

    return EISDataset(
        deepcopy(dataset.data), dataset.index, dataset.source_file, dataset.file_id
    )


def inductive_tail_removed(dataset):
    """A detached copy of an EISDataset with the inductive tail (Im(Z) > 0)
    masked on top of whatever is already masked.

    A copy rather than an in-place mask, unlike mask_inductive_points: this is
    a per-analysis filter, and the shared mask is what validation results are
    checked against, so moving it would mark them stale."""
    filtered = detached_copy(dataset)
    Z = filtered.data.get_impedances(masked=None)  # all points, incl. masked
    inductive = {i: True for i, z in enumerate(Z) if z.imag > 0}
    # Guarded: set_mask({}) means "unmask everything", so an empty dict here
    # would hand back a copy with the original's masking undone.
    if inductive:
        filtered.data.set_mask(inductive)
    return filtered


def mask_points(dataset, indices) -> None:
    """Force the given point indices masked, in place, leaving every other
    point's state alone. Used to replay an iterative prune's removals, which
    _refresh cannot re-derive without re-running the validation."""
    if indices:
        dataset.data.set_mask({int(i): True for i in indices})


def apply_manual_overrides(dataset, masked, kept) -> None:
    """Force the given point indices masked / unmasked, in place, on top of
    whatever the automatic filters have already decided."""
    overrides = {int(i): True for i in masked}
    overrides.update({int(i): False for i in kept})
    if overrides:
        dataset.data.set_mask(overrides)

#def mask_diffusion_points(dataset):
# this is a placeholder for mathematically removing the diffusion points