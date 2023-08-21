from setuptools import setup, find_packages
import os


def get_requirements(filepath: str):
    """This function returns a list of requirements"""
    requirements = []
    with open(filepath, "r") as f:
        requirements = f.readlines()
        requirements = [requirement.replace(
            "\n", "") for requirement in requirements]

    if '-e .' in requirements:
        requirements.remove('-e .')

    return requirements


setup(
    name="breast_cancer_prediction",
    version="0.0.1",
    author="amulyaprasanth",
    author_email="amulyaprasanth301@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
