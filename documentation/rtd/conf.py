"""Configuration for the Sphinx documentaiton generator."""

project = "Yggdrasil Decision Forests"
copyright = "2022, Google"
author = "Mathieu Guillame-Bert"

release = "1.0.1"
version = "0.2.5"

master_doc = "index"

# Extensions for sphinx.
extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "myst_parser",
]

# Extensions for markdown files.
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

myst_heading_anchors = 3

html_theme = "sphinx_book_theme"
html_logo = "image/both_logo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}

epub_show_urls = "footnote"
