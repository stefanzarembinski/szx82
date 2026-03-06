## Installation

### Add the package to the Python path

To build wheel:

    PS .... szx82> python -m build

To install from wheel:

    PS .... szx82> pip install dist\szx82-0.1-py3-none-any.whl

To uninstall:

    PS ... > pip uninstall szx82

To install in editable mode:

    PS ... > szx82> pip install -e .

### Set the storage directory of the project according to the forex instrument used

    PS .... szx81> python -m szx81.config set C:\Users\stefa\Documents\workspaces\szx81\EURUSD
    PS .... szx81> python -m szx81.config set C:\Users\stefa\Documents\workspaces\EURUSD-26.10.2024

-- for example