# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))

import builtins
builtins.__sphinx_build__ = True  # type: ignore

import glob
import shutil


def copy_examples_from_manual(dest_folder: str, create_rst: bool = True):
    # Copy Examples From Manual
    nb_manual_files = glob.glob("../../../Spy4CastManual/*.ipynb")
    py_manual_files = glob.glob("../../../Spy4CastManual/*.py")
    added = set()
    for file in glob.glob(dest_folder+"/*"):
        filename = os.path.basename(file)
        if filename != "manual.rst":
            print(f"[INFO] REMOVING: {file}")
            os.remove(file)
    for file in nb_manual_files:
        filename = os.path.basename(file)
        dest_file = os.path.join(dest_folder, filename)
        print(f"[INFO] COPY: {file} -> {dest_file}")
        shutil.copy(file, dest_file)
        name, _ext = os.path.splitext(filename)
        added.add(name)

    for file in py_manual_files:
        filename = os.path.basename(file)
        name, _ext = os.path.splitext(filename)
        if name in added:
            # Already included the ipynb ersion
            continue
        dest_file = os.path.join(dest_folder, filename)
        print(f"[INFO] COPY: {file} -> {dest_file}")
        shutil.copy(file, dest_file)
        if create_rst:
            with open(os.path.join(dest_folder, name+".rst"), "w") as f:
                title = name.replace("_", " ")
                f.write(title+"\n")
                f.write("="*len(title)+"\n")
                f.write("\n")
                f.write(f".. literalinclude:: {filename}")

copy_examples_from_manual("manual/")
copy_examples_from_manual("../../examples/Spy4CastManual/", create_rst=False)

# -- Project information -----------------------------------------------------

project = 'Spy4Cast'
copyright = '2022, Pablo Duran'
author = 'Pablo Duran'
with open('../../pyproject.toml', 'r') as f:
    content = f.read()
    start = content.find('version = ')
    end = content[start:].find('\n') + start
    version = content[start:end].split(' = ')[-1]
    print(f'[INFO] Running on version {version}')
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx_panels',
    'enum_tools.autoenum',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
]

source_suffix = [".rst", ".md"]

numpydoc_show_class_members = False

automodapi_writereprocessed = True

# autodoc_member_order = 'bysource'

html_favicon = "_static/images/favicon.png"

# Napoleon settings: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = [".py"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'furo'
# html_theme = 'sphinx_book_theme'
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'css/style.css',
]

todo_include_todos = True
