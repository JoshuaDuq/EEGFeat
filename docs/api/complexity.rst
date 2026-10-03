Complexity and Microstates
==========================

Their definitions are in :doc:`/methods/complexity`. Microstate segmentation
requires the ``microstates`` extra.

Entropy and Fractal Dimension
-----------------------------

.. autofunction:: eegtable.higuchi_fractal_dimension

.. autofunction:: eegtable.sample_entropy

.. autofunction:: eegtable.multiscale_entropy

.. autofunction:: eegtable.permutation_entropy

.. autofunction:: eegtable.lempel_ziv_complexity

.. autofunction:: eegtable.detrended_fluctuation

Microstates
-----------

.. autoclass:: eegtable.MicrostateModel
   :members:

.. autofunction:: eegtable.microstates.segment

.. autofunction:: eegtable.microstates.microstate_coverage

.. autofunction:: eegtable.microstates.microstate_duration

.. autofunction:: eegtable.microstates.microstate_occurrence

.. autofunction:: eegtable.microstates.microstate_transitions

.. autoclass:: eegtable.microstates.MicrostateSegmentation
   :members:
   :show-inheritance:
