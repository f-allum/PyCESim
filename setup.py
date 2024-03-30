from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'PyCESim - classical simulation of Coulomb explosion'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="PyCESim", 
        version=VERSION,
        author="Felix Allum",
        author_email="<fallum@stanford.edu",
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=['cclib'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['Coulomb explosion']
)