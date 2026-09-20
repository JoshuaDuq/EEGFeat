Complexity and Microstates
==========================

Their definitions are in :doc:`/methods/complexity`. Microstate segmentation
requires the ``microstates`` extra.

Entropy and Fractal Dimension
-----------------------------

.. autofunction:: eegfeat.higuchi_fractal_dimension

.. autofunction:: eegfeat.sample_entropy

.. autofunction:: eegfeat.multiscale_entropy

Microstates
-----------

.. autofunction:: eegfeat.microstates.segment

.. autofunction:: eegfeat.microstates.microstate_coverage

.. autofunction:: eegfeat.microstates.microstate_duration

.. autofunction:: eegfeat.microstates.microstate_occurrence

.. autofunction:: eegfeat.microstates.microstate_transitions

.. autoclass:: eegfeat.microstates.MicrostateSegmentation
   :members:
   :show-inheritance:
