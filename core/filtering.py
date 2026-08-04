def mask_inductive_points(dataset) -> None: #remove inductive tail
    """Mask points with a positive imaginary impedance (inductive artifacts),
    in place."""
    Z = dataset.data.get_impedances(masked=None)  # all points, incl. masked
    dataset.data.set_mask({i: bool(z.imag > 0) for i, z in enumerate(Z)})


def clear_mask(dataset) -> None: #re-add the inductive points
    """Unmask all points of an EISDataset (in place)."""
    dataset.data.set_mask({})


def apply_manual_overrides(dataset, masked, kept) -> None:
    """Force the given point indices masked / unmasked, in place, on top of
    whatever the automatic filters have already decided."""
    overrides = {int(i): True for i in masked}
    overrides.update({int(i): False for i in kept})
    if overrides:
        dataset.data.set_mask(overrides)

#def mask_diffusion_points(dataset):
# this is a placeholder for mathematically removing the diffusion points