import importlib
from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    install_requires = f.read()

setup(
    name="polliwog",
    version=importlib.import_module("polliwog").__version__,
    description="Computation library for 2D and 3D geometry",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Body Labs, Metabolize, and other contributors",
    author_email="github@paulmelnikow.com",
    url="https://github.com/lace/polliwog",
    project_urls={
        "Issue Tracker": "https://github.com/lace/polliwog/issues",
        "Documentation": "https://polliwog.readthedocs.io/en/stable/",
    },
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "Topic :: Artistic Software",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        # "Programming Language :: Python :: 3",
    ],
)
