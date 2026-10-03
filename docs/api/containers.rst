Containers and I/O
==================

:class:`~eegtable.FeatureTable` is the return type of every extractor. The other
types on this page are inputs. Field definitions are in :doc:`/concepts`.

Feature tables
--------------

.. autoclass:: eegtable.FeatureTable
   :members:
   :show-inheritance:

.. autoclass:: eegtable.FeatureMeta
   :members:
   :show-inheritance:

.. autoclass:: eegtable.ComputationSpec
   :members:
   :show-inheritance:

.. autofunction:: eegtable.concat

.. autofunction:: eegtable.stack_rows

Reading and writing
-------------------

.. autoclass:: eegtable.io.FeatureDataset
   :members:

.. autofunction:: eegtable.io.read_dataset

.. autofunction:: eegtable.io.read_table

.. autofunction:: eegtable.io.write_table

Group samples
-------------

.. autoclass:: eegtable.group.GroupDataset
   :members:

.. autoclass:: eegtable.group.GroupDesign
   :members:

Native BIDS input
-----------------

.. autoclass:: eegtable.bids.BIDSQuery
   :members:

.. autoclass:: eegtable.bids.BIDSRecording
   :members:

.. autofunction:: eegtable.bids.discover_bids

.. autofunction:: eegtable.bids.read_bids

.. autofunction:: eegtable.bids.preprocess_bids

Bands and windows
-----------------

.. autoclass:: eegtable.Band
   :members:
   :show-inheritance:

.. autoclass:: eegtable.Window
   :members:
   :show-inheritance:

.. autodata:: eegtable.BANDS_STANDARD
   :no-value:

.. autofunction:: eegtable.passband_fraction

.. autofunction:: eegtable.check_passband

Signal containers
-----------------

.. autoclass:: eegtable.Spectra
   :members:
   :show-inheritance:

.. autoclass:: eegtable.Signal
   :members:
   :show-inheritance:

.. autoclass:: eegtable.BandSignal
   :members:
   :show-inheritance:
