import eegfeat


def test_package_exposes_a_version() -> None:
    assert isinstance(eegfeat.__version__, str)
    assert eegfeat.__version__
