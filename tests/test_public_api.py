import types

import eegfeat

EXPECTED = {
    "__version__",
    "Band",
    "BANDS_STANDARD",
    "Window",
    "Spectra",
    "FeatureMeta",
    "FeatureTable",
    "concat",
    "band_power",
    "peak_frequency",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_edge",
    "spectral_entropy",
    "aperiodic",
    "aperiodic_ratio",
    "band_ratio",
    "asymmetry",
}


def test_public_namespace_is_exactly_the_documented_surface() -> None:
    assert set(eegfeat.__all__) == EXPECTED


def test_every_exported_name_resolves() -> None:
    for name in eegfeat.__all__:
        assert getattr(eegfeat, name) is not None


def test_no_private_name_is_exported() -> None:
    assert not [n for n in eegfeat.__all__ if n.startswith("_") and n != "__version__"]


def test_the_only_private_attributes_are_submodules() -> None:
    # Importing eegfeat._expand binds the name `_expand` on the package. That is
    # how Python packages work, not an export. What must not appear is a private
    # *value*: a helper function or constant that escaped into the namespace.
    private = [n for n in dir(eegfeat) if n.startswith("_") and not n.startswith("__")]
    assert all(isinstance(getattr(eegfeat, n), types.ModuleType) for n in private)
