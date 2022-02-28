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
import enum
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
import inspect


def list_submodules(module):
	modules = inspect.getmembers(module, inspect.ismodule)
	moduleblacklist = ["abc", "atexit", "spy4cast", "builtins", "ctypes",
					   "core", "struct", "sys", "traceback", "code", "enum",
					   "json", "numbers", "threading", "re", "requests", "os", "startup",
					   "associateddatastore", "range", "pyNativeStr", "cstr", "fnsignature",
					   "get_class_members", "datetime", "inspect", "subprocess", "site",
					   "string", "random", "uuid", "queue", "collections", "np"]
	return sorted(set(x for x in modules if x[0] not in moduleblacklist))


def list_items(module):
	items = inspect.getmembers(module, lambda e: inspect.isfunction(e) or inspect.isclass(e))
	return sorted(set(x for x in items))


def use_sth_odd():
	os.environ["BN_DISABLE_USER_SETTINGS"] = "True"
	os.environ["BN_DISABLE_USER_PLUGINS"] = "True"
	os.environ["BN_DISABLE_REPOSITORY_PLUGINS"] = "True"
	import spy4cast

	def classlist(module):
		members = inspect.getmembers(module, inspect.isclass)
		classblacklist = ['builtins']
		if module.__name__ != "spy4cast.enums":
			members = sorted(x for x in members if type(x[1]) != enum.EnumMeta and x[1].__module__ not in classblacklist)
			members.extend(fnlist(module))
		return (x for x in members if not x[0].startswith("_"))

	def fnlist(module):
		return [x for x in inspect.getmembers(module, inspect.isfunction) if x[1].__module__ == module]

	def setup(app):
		app.add_css_file('css/other.css')
		app.is_parallel_allowed('write')

	def generaterst():
		pythonrst = open("index.rst", "w")
		pythonrst.write('''Binary Ninja Python API Documentation
	=====================================
	Welcome to the Binary Ninja API documentation. The below methods are available
	from the root of the `binaryninja` package, but most of the API is organized
	into the modules shown in the left side-bar.
	You can also scroll to the end to view a class list of all available classes.
	The search bar on the side works both online and offline.
	.. automodule:: spy4cast
	   :members:
	   :undoc-members:
	   :show-inheritance:
	Full Class List
	---------------
	.. toctree::
	   :maxdepth: 2
	   
	   usage
	   methodologies
	''')

		for modulename, module in modulelist(spy4cast):
			filename = f"spy4cast.{modulename}-module.rst"
			pythonrst.write(f"   {modulename} <{filename}>\n")
			modulefile = open(filename, "w")
			underline = "="*len(f"{modulename} module")
			modulefile.write(f'''{modulename} module
	{underline}
	.. autosummary::
	   :toctree:
	''')

			for (classname, classref) in classlist(module):
				modulefile.write(f"   spy4cast.{modulename}.{classname}\n")

			modulefile.write('''\n.. toctree::
	   :maxdepth: 2\n''')

			modulefile.write(f'''\n\n.. automodule:: spy4cast.{modulename}
	   :members:
	   :undoc-members:
	   :show-inheritance:''')
			modulefile.close()

		pythonrst.close()


	generaterst()


def generate_functions_tree(module, tocpath):
	# print(modulelist(module))

	for sub_name, submod in list_submodules(module):
		with open(os.path.join(tocpath, f'spy4cast.{sub_name}.rst'), 'a') as f:
			f.write("\n.. toctree::\n")
			for f_name, _func in list_items(submod):
				f.write(f"    {module.__name__}.{sub_name}.{f_name} <:py:meth:`{module.__name__}.{sub_name}.{f_name}`>\n")

# import spy4cast
# generate_functions_tree(spy4cast, '.toctrees')

# -- Project information -----------------------------------------------------

project = 'Spy4Cast'
copyright = '2022, Pablo Duran'
author = 'Pablo Duran'
version = '0.0.1'
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'enum_tools.autoenum',
]
autodoc_member_order = 'bysource'
html_favicon = "_static/favicon.ico"

# Napoleon settings
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
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

todo_include_todos = True