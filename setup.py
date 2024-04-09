from setuptools import find_packages, setup

setup(
    name='src',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    version='1.0.0',
    description='mensa',
    author='Christian Marius Lillelund',
    author_email='cl@ece.au.dk',
    license='MIT',
)
