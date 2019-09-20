from setuptools import setup, find_packages

setup(
    name="toplayer",
    version="0.0.1",
    keywords=("pip", "tensorflow", "keras", "machinelearning", "deeplearning"),
    description="eds sdk",
    long_description="eds sdk for python",
    license="MIT Licence",

    url="http://xiaoh.me",
    author="chenquan",
    author_email="chenquan@osai.club",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=["tensorflow", "numpy"]
)
