import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SWE_VT",
    version="0.0.1",
    author="Victor Trappler",
    author_email="victor.trappler@univ-grenoble-alpes.fr",
    description="Shallow Water 1D for toy problem of robust estimation of bottom friction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
