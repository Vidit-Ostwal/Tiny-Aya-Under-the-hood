"""
Tiny Aya Under The Hood (uth) — Multilingual Representation Analysis.

This package provides tools and utilities for analyzing how Tiny Aya
(CohereLabs/tiny-aya-global) and its regional variants process
information across languages. By examining layer-wise representations,
we identify where language-agnostic (universal) processing emerges
and where language-specific specialization occurs.

Sub-packages:
    - ``uth.utils``: Language registry, metadata, and shared utilities.
    - ``uth.data``: Data loading (FLORES-200, translation pipelines).
    - ``uth.analysis``: Cross-lingual alignment analysis (CKA, hooks,
      retrieval metrics, clustering, visualization).

Quick start::

    from uth.utils.languages import Language, LANGUAGE_FAMILIES
    from uth.data.flores_loader import load_flores_parallel_corpus
    from uth.analysis.hooks import load_model
    from uth.analysis.cross_lingual_alignment import (
        CrossLingualAlignmentAnalyzer,
    )
"""

__version__ = "0.1.0"
