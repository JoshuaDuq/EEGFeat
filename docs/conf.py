from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "eegfeat"
author = "Joshua Duquette"
release = "0.1.0.dev0"
copyright = "2026, Joshua Duquette"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

# ---------------------------------------------------------------------------
# Source / build
# ---------------------------------------------------------------------------
templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = [
    "_build",
    "superpowers",
    "Thumbs.db",
    ".DS_Store",
]

# ---------------------------------------------------------------------------
# Napoleon (docstring style)
# ---------------------------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# ---------------------------------------------------------------------------
# Autodoc
# ---------------------------------------------------------------------------
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_mock_imports = ["mne_connectivity", "shap"]
# Keep the name written in the source (`BANDS_STANDARD`) instead of its repr.
autodoc_preserve_defaults = True
# One parameter per line once a signature no longer fits the content column.
maximum_signature_line_length = 68
python_trailing_comma_in_multi_line_signatures = True

# ---------------------------------------------------------------------------
# MyST (Markdown support)
# ---------------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 4

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "mne": ("https://mne.tools/stable", None),
}

# ---------------------------------------------------------------------------
# HTML output — furo theme
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "eegfeat"
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap",
    "custom.css",
]
pygments_style = "friendly"
pygments_dark_style = "monokai"
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
html_favicon = "_static/favicon.svg"
html_logo = "_static/favicon.svg"

html_theme_options = {
    "dark_css_variables": {
        "color-background-primary": "#0A0B0D",
        "color-background-secondary": "#131518",
        "color-background-hover": "#1A1C20",
        "color-background-border": "#24262A",
        "color-foreground-primary": "#E6E8EB",
        "color-foreground-secondary": "#A4A8AD",
        "color-foreground-muted": "#6A6D72",
        "color-foreground-border": "#3D4045",
        "color-brand-primary": "#F2F3F5",
        "color-brand-content": "#BFC3C7",
        "color-highlight-on-target": "rgba(242, 243, 245, 0.07)",
        "color-admonition-background": "#131518",
        "color-api-name": "#F2F3F5",
        "color-api-pre-name": "#6A6D72",
        "color-api-keyword": "#A4A8AD",
        "color-api-paren": "#6A6D72",
        "color-api-overall": "#E6E8EB",
        "color-api-background": "#131518",
        "color-api-background-hover": "#1A1C20",
        "font-stack": "'Inter', 'Segoe UI', system-ui, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', ui-monospace, monospace",
    },
    "light_css_variables": {
        "color-background-primary": "#FAFAFB",
        "color-background-secondary": "#F1F2F4",
        "color-background-hover": "#E8EAEC",
        "color-background-border": "#D8DADD",
        "color-foreground-primary": "#1A1C20",
        "color-foreground-secondary": "#4A4D52",
        "color-foreground-muted": "#74777C",
        "color-foreground-border": "#C4C6C9",
        "color-brand-primary": "#1A1C20",
        "color-brand-content": "#3D4045",
        "color-highlight-on-target": "rgba(26, 28, 32, 0.06)",
        "color-admonition-background": "#F1F2F4",
        "color-api-name": "#0E0F12",
        "color-api-pre-name": "#74777C",
        "color-api-keyword": "#4A4D52",
        "color-api-paren": "#74777C",
        "color-api-overall": "#1A1C20",
        "color-api-background": "#F1F2F4",
        "color-api-background-hover": "#E8EAEC",
        "font-stack": "'Inter', 'Segoe UI', system-ui, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', ui-monospace, monospace",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/JoshuaDuq/EEGFeatML",
    "source_branch": "main",
    "source_directory": "docs/",
}
