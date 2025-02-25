# Pytools-Image-Processing

This python package contains some basic image processing functions that I use in several of my hobby projects. The code is not optimized for speed but for readability and ease of use. The code is not tested on Windows or Mac but should work on those platforms.

## Installation

The easiers way to install the package is using the Python Package Index (PyPi). You can install the package by running the following command in your terminal.

```bash
# This will pull the latest version from PyPi
pip install pytools-image-processing
```

You can also use the package by downloading the repo and installing the package as editable. This way you can make changes to the code and see the changes in your code without having to reinstall the package.

```bash
# This code is for NON ANACONDA users on Linux
# Clone the repo
git clone https://github.com/Nynra/pytools-image_processing.git
cd pytools-lithography

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate # Activate venv for linux
pip install --editable .  # Install the package
```

As I use VsCode and Linux myself I cannot provide instructions on how to use this code on Windows or Mac. All the code should work on Mac and Windows but the installation steps might be different and has not been tested. If you have any issues please let me know by creating an issue on the github page.