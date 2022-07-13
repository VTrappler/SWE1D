import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SWE1D",
    version="0.0.2",
    author="Victor Trappler",
    author_email="victor.trappler@gmail.com",
    description="Shallow Water 1D for toy problem of robust estimation of bottom friction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vtrappler/SWE1D",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy"],
)
