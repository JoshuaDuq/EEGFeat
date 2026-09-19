Installation
============

.. raw:: html

   <p class="hero-lede">
     Install <code>eegfeat</code> from source or package manager into a Python 3.11+
     virtual environment. Core dependencies are kept strictly minimal, with optional
     extras for specialized estimators.
   </p>

Prerequisites
-------------

.. grid:: 2
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Python ≥ 3.11
      :link: https://www.python.org/downloads/
      :link-type: url

      Required runtime for modern typing and performance.

   .. grid-item-card:: Core Scientific Stack
      :link: https://mne.tools/stable/
      :link-type: url

      ``numpy>=1.26``, ``scipy>=1.11``, ``pandas>=2.0``, ``mne>=1.8``.

Setup
-----

macOS / Linux
~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/JoshuaDuq/eegfeat.git
   cd eegfeat
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .

Windows PowerShell
~~~~~~~~~~~~~~~~~~

.. code-block:: powershell

   git clone https://github.com/JoshuaDuq/eegfeat.git
   cd eegfeat
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -e .

Optional Dependencies
---------------------

Install optional components as needed depending on your analysis scope:

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
     - Design matrices, leakage-safe cross-fitting, metrics, nulls, uncertainty, and model
       selection (:mod:`eegfeat.model`).
   * - ``[importance]``
     - ``scikit-learn>=1.3``, ``shap>=0.44``
     - Held-out permutation importance and SHAP explanations
       (:mod:`eegfeat.model.importance`).
   * - ``[knee]``
     - ``specparam>=2.0.0rc7,<3``
     - Spectral knee parameterization.
   * - ``[dev]``
     - ``pytest``, ``ruff``, ``black``, ``mypy``, type stubs
     - Test suite execution, type checking, and linting.
   * - ``[docs]``
     - ``furo``, ``myst-parser``, ``sphinx-copybutton``, ``sphinx-design``, ``scikit-learn``
     - Building the Sphinx documentation locally, including the modeling API reference.

To install with all extras:

.. code-block:: bash

   pip install -e ".[connectivity,microstates,knee,dev,docs]"

For predictive modeling, install ``.[model]``. Add ``.[importance]`` when SHAP explanations
are needed; permutation importance itself is available with the model extra.

Verification
------------

Verify your installation by running the test suite:

.. code-block:: bash

   pytest
