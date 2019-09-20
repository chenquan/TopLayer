#    Copyright 2019 Quan Chen
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from setuptools import setup, find_packages

setup(
    name="toplayer",
    version="0.0.1",
    keywords=("pip", "tensorflow", "keras", "machinelearning", "deeplearning"),
    description="",
    long_description="The decorator base class for TensorFlow's Keras advanced API top-level external to other "
                     "frameworks.",

    author="chenquan",
    author_email="chenquan@osai.club",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=["tensorflow", "numpy"]
)
