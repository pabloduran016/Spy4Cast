import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spy4cast",
    version="0.0.9",
    author="Pablo Duran",
    author_email="pdrm56@gmail.com",
    description="Python API for applying methodologies to .nc Datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pabloduran016/Spy4Cast",
    project_urls={
        "Repo": "https://github.com/pabloduran016/Spy4Cast",
        "Documentation": "https://spy4cast-docs.netlify.app",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where=""),
    install_requires=[
        'numpy', 'matplotlib', 'xarray', 'dask', 'pandas', 'netcdf4'
    ],
    python_requires=">=3.6",
)
