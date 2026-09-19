from importlib.util import find_spec

# Modules here that import eegfeat.model, which refuses to import without scikit-learn.
collect_ignore = [] if find_spec("sklearn") is not None else ["test_equivalence_model.py"]
