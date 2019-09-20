from setuptools import setup, find_packages

setup(
    name="toplayer",
    version="0.0.1",
    keywords=("pip", "tensorflow", "keras", "machinelearning", "deeplearning"),
    description="",
    long_description="The decorator base class for TensorFlow's Keras advanced API top-level external to other "
                     "frameworks.",
    license="MIT Licence",

    author="chenquan",
    author_email="chenquan@osai.club",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=["tensorflow", "numpy"]
)
