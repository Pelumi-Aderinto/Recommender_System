from setuptools import setup, find_packages

setup(
    name='movielens_analysis',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'tqdm',
        'wget'
    ],
    entry_points={
        'console_scripts': [
            'movielens_analysis=main:main',
        ],
    },
)
