from setuptools import setup, find_packages

setup(
    name='shopee-product-matching',
    version='2.0.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    data_files=[],
    entry_points={}
)
