from setuptools import setup, find_packages

VERSION = '0.0.2' 
DESCRIPTION = 'PyCESim - classical simulation of Coulomb explosion'

setup(
        name="PyCESim", 
        version=VERSION,
        author="Felix Allum",
        author_email="fallum@stanford.edu",
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=['cclib', 'numpy', 'matplotlib', 'scipy', 'pandas'],
        url='https://github.com/f-allum/PyCESim/',
        download_url='https://github.com/f-allum/PyCESim/archive/refs/tags/v0.0.2.tar.gz',
        keywords=['Coulomb explosion']
)