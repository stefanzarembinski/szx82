from setuptools import setup, find_packages

setup(
    name='szx82',
    version='0.1',
    author='sz',
    author_email='stefan.zarenmbinski@gmail.com',
    packages=find_packages(),
    package_data={
        'hist': ['EURUSD\data_store\hist_data.pkl',]
    },
    include_package_data=True,
    # install_requires=[
    #     'numpy-2.2.3',
    #     'matplotlib-3.10.0',
    #     'pickle',
    #     'torch',
    #     'ipython-8.32.0',
    #     'scipy-1.15',
    # ],
    url='https://github.com/stefanzarembinski/szx82',

    description='Encoder-Decoder Transformer czy może się przydać?',
    long_description=open('README.txt').read(),
    long_description_content_type='text',

    python_requires='>=3.10'
)

# Install locally for development:
# C:\Users\stefa\Documents\workspaces\szx82\szx82> pip install -e .

# To ship extra directories, explicitly state them in the manifest to be packaged. We will do this inside the 'MANIFEST.in.'

# To create distribution file:
    # cd /szx82 # where `setup.py`
    # python setup.py sdist
# Result - wheel and egg - goes to `dist` directory.

# To install from wheel:
# pip install C:\Users\stefa\Documents\workspaces\szx82\dist\szx82-0.1-py3-none-any.whl
 





 

