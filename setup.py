import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="waterstructureCreator",
    version="0.0.1",
    author="Nicolas G. Hoermann",
    author_email="hoermann@fhi.mpg.de",
    description=
    "Creation of water structures on substrates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'scipy==1.7.1', 'numpy==1.17.5', 'matplotlib==3.1.3', 'ipython==7.26.0', 'scikit-learn==0.24.1', 'ase==3.20.1', 'pymatgen==2020.11.11'
    ],
    extras_require={'testing': ['pytest>=5.0']},
    python_requires='==3.8.3',
)
