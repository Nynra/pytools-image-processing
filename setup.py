import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyimageprocessing",
    version="0.0.2",
    author="Nynra",
    author_email="nynradev@pm.me",
    description="Some usefull functions for image processing and analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["hhslablib"],
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "opencv-python",
        "matplotlib",
        "scikit-image",
        "scipy",
    ],
)