import setuptools
import os


# Read the README.md file
with open("README.md", "r") as f:
    long_description = f.read()


install_requires = [
    "opencv-python",
    "scikit-image",
    "matplotlib",
    "scipy",
    "numpy",
]

setuptools.setup(
    include_package_data=True,
    name="pytools_image_processing",
    version="0.0.5",
    author="Nynra",
    author_email="nynradev@pm.me",
    description="Some usefull functions for image processing and analysis.",
    py_modules=["pytools_image_processing"],
    package_dir={"": "src"},
    install_requires=install_requires,
)
