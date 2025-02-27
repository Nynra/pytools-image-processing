# Pytools-Image-Processing

This python package contains some basic image processing functions that I use in several of my hobby projects. The code is not optimized for speed but for readability and ease of use.

While some of the functions might seem trivial, I have found that I use them often enough to warrant a package. The package is not meant to be a replacement for OpenCV or other image processing libraries but rather a wrapper to provide an easy to update and maintain set of functions that I use often in other projects.

## Installation

NOTE: All instruction assume the user is running Linux and has Python 3.12 or higher installed. The code should work on Windows and Mac but has not been tested (If you have tested the code on Windows or Mac please let me know).

The easiers way to install the package is GitHub. You can install the package by running the following command in your terminal.

```bash
# This will pull the latest version from GitHub
pip install git+https://github.com/Nynra/pytools-image_processing.git
```

To install a specific version you can use the following command.

```bash
# This will pull the version 0.1.0 from GitHub
pip install git+https://github.com/Nynra/pytools-image_processing.git@0.1.0
```

You can also use the package by downloading the repo and installing the package as editable. This way you can make changes to the code and see the results without having to reinstall the package.

```bash
# This code is for NON ANACONDA users on Linux
# Clone the repo
git clone https://github.com/Nynra/pytools-image_processing.git
cd pytools-image_processing

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate # Activate venv for linux
pip install --editable .  # Install the package
```

As I use VsCode and Linux myself I cannot provide instructions on how to use this code on Windows, Mac or in other IDEs. All the code should work on Mac and Windows but the installation steps might be different and has not been tested. If you have installed and/or used the package in other configurations please let me know by creating an issue or a pull request.
