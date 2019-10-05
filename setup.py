from setuptools import setup, find_packages

# Set version_info[__version__], while avoiding importing numpy, in case numpy
# and vg are being installed concurrently.
# https://packaging.python.org/guides/single-sourcing-package-version/
version_info = {}
exec(open("polliwog/package_version.py").read(), version_info)

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    install_requires = f.read()

setup(
    name="polliwog",
    version=version_info["__version__"],
    description="2D and 3D computational geometry library which scales from prototyping to production",
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
        "Programming Language :: Python :: 3",
    ],
)
