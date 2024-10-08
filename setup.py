import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('setup.cfg', 'r') as f:
    content = f.read()
    start = content.find('version = ')
    end = content[start:].find('\n') + start
    version = content[start:end].split(' = ')[-1]
    print(f'[INFO] Running on version {version}')

setuptools.setup(
    name="spy4cast",
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
        'numpy==1.22.4', 'matplotlib', 'xarray==2022.3.0', 'dask', 'pandas', 'netcdf4', 'scipy==1.7.3', 
    ],
    python_requires=">=3.6",
)
