Containers and I/O
==================

:class:`~eegfeat.FeatureTable` is the return type of every extractor. The other
types on this page are inputs. Field definitions are in :doc:`/concepts`.

Feature tables
--------------

.. autoclass:: eegfeat.FeatureTable
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.FeatureMeta
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.ComputationSpec
   :members:
   :show-inheritance:

.. autofunction:: eegfeat.concat

.. autofunction:: eegfeat.stack_rows

Reading and writing
-------------------

.. autoclass:: eegfeat.io.FeatureDataset
   :members:

.. autofunction:: eegfeat.io.read_dataset

.. autofunction:: eegfeat.io.read_table

.. autofunction:: eegfeat.io.write_table

Bands and windows
-----------------

.. autoclass:: eegfeat.Band
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.Window
   :members:
   :show-inheritance:

.. autofunction:: eegfeat.passband_fraction

.. autofunction:: eegfeat.check_passband

Signal containers
-----------------

.. autoclass:: eegfeat.Spectra
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.Signal
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.BandSignal
   :members:
   :show-inheritance:
