import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.read().splitlines()

setuptools.setup(
    name="pytools_image_processing",
    version="0.0.2",
    author="Nynra",
    author_email="nynradev@pm.me",
    description="Some usefull functions for image processing and analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["pytools_image_processing"],
    package_dir={"": "src"},
    install_requires=install_requires,
)