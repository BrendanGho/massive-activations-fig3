"""Shared building blocks for the Figure 3 reproduction pipeline.

The modules here are split by dependency weight on purpose:

* ``config``, ``clustering``, ``io`` are pure ``numpy`` / ``scikit-learn`` and are
  safe to import anywhere (including the test suite) without a GPU or the heavy
  ``fig3`` extra.
* ``model_utils`` imports ``torch`` / ``diffusers`` / ``transformers`` **lazily**
  inside its functions, so importing the module itself is still cheap.
"""
