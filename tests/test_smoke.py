import eegtable


def test_package_exposes_a_version() -> None:
    assert isinstance(eegtable.__version__, str)
    assert eegtable.__version__
