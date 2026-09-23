Installation
============

.. raw:: html

   <p class="hero-lede">
     Python 3.11 or newer. Install from source. Core dependencies are
     <code>numpy</code>, <code>scipy</code>, <code>pandas</code>, and <code>mne</code>.
   </p>

Prerequisites
-------------

.. grid:: 2
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Python ≥ 3.11
      :link: https://www.python.org/downloads/
      :link-type: url

      Python 3.11 or newer.

   .. grid-item-card:: Core Scientific Stack
      :link: https://mne.tools/stable/
      :link-type: url

      ``numpy>=1.26``, ``scipy>=1.11``, ``pandas>=2.0``, ``mne>=1.8``.

Setup
-----

macOS / Linux
~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/JoshuaDuq/EEGFeat.git
   cd EEGFeat
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .

Windows PowerShell
~~~~~~~~~~~~~~~~~~

.. code-block:: powershell

   git clone https://github.com/JoshuaDuq/EEGFeat.git
   cd EEGFeat
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -e .

Optional Dependencies
---------------------

Optional extras.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Extra
     - Packages
     - Used For
   * - ``[connectivity]``
     - ``mne-connectivity>=0.7``
     - Weighted Phase Lag Index (:func:`~eegfeat.connectivity.wpli`).
   * - ``[microstates]``
     - ``scikit-learn>=1.3``
     - GFP peak clustering and microstate segmentation (:mod:`eegfeat.microstates`).
   * - ``[model]``
     - ``scikit-learn>=1.3``
     - Design matrices, grouped cross-fitting, metrics, nulls, uncertainty, and
       model selection (:mod:`eegfeat.model`).
   * - ``[importance]``
     - ``scikit-learn>=1.3``, ``shap>=0.45``
     - SHAP explanations (:mod:`eegfeat.model.importance`). Permutation importance
       is in the ``model`` extra.
   * - ``[preprocessing]``
     - ``mne>=1.13.2``, ``PyYAML``, ``scikit-learn``, ``h5io``, ``h5py``, ``filelock``
     - Raw-to-epochs preprocessing workflow (:doc:`/guides/preprocessing`).
   * - ``[preprocessing-auto]``
     - ``pyprep``, ``autoreject``, ``mne-icalabel``, ``python-picard``, ``onnxruntime``
     - PyPREP candidates, ICLabel, the Picard ICA solver, and autoreject.
   * - ``[bids]``
     - ``mne-bids``, ``pybv``
     - The study conversion in ``paradigm_specific/thermal_pain``. Not used by ``eegfeat``.
   * - ``[preprocessing-gui]``
     - ``mne-qt-browser``, ``PyQt6``
     - MNE viewers for ``preprocess review`` and ``preprocess inspect``.
   * - ``[dev]``
     - ``pytest``, ``ruff``, ``black``, ``mypy``, type stubs
     - Tests, type checking, and linting.
   * - ``[docs]``
     - ``furo``, ``myst-parser``, ``sphinx-copybutton``, ``sphinx-design``, ``scikit-learn``
     - This documentation, including the modeling API pages.

To install with all extras:

.. code-block:: bash

   pip install -e ".[connectivity,microstates,model,importance,preprocessing,preprocessing-auto,dev,docs]"

Modeling needs the ``model`` extra. SHAP needs the ``importance`` extra.
Permutation importance is included in ``model``.

The preprocessing terminal front end is a Go program, not a Python extra.
Build it with ``cd tui && go build -o eegfeat-tui .`` (Go 1.24 or newer).
Keys are in :doc:`/guides/preprocessing`.

Verification
------------

Run the tests with

.. code-block:: bash

   pytest
