eegfeat
=======

Labelled spectral feature extraction for MNE objects.

``eegfeat`` turns an MNE ``Spectrum`` or ``EpochsTFR`` into a ``FeatureTable``: values
plus one metadata record per column, so band, channel, window and normalization
are structured fields rather than fragments of a column name.

It computes no time-frequency transform of its own. You bring a ``Spectrum`` or an
``EpochsTFR``; ``eegfeat`` turns it into features.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   methods

API Reference
-------------

.. automodule:: eegfeat
   :members:
   :undoc-members:
   :show-inheritance:
