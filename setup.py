from setuptools import setup, find_packages

readme = open('README.md').read()

VERSION = '0.0.1'

requirements = [
    'numpy',
    'pandas'
]

setup(
    # Metadata
    name='sharpen',
    version=VERSION,
    author='cestwc',
    author_email='cestwc@gmail.com',
    url='https://github.com/cestwc/sharpen',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='MIT',

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
