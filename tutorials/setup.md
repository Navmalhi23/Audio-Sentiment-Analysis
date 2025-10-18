# Project Setup Tutorial
## Python3 Installation
If not already installed, download Python3 (version >=3.12) based on your operating system from https://www.python.org/downloads/

Verify the installation in your terminal using the command ```python3 --version```

## Clone Repository

Clone this repository using your terminal:

```cd desired_parent_dir```

```git clone https://github.com/svarshh/Audio-Sentiment-Analysis```


## Setup Virtual Environment

Once cloned the repository, create a Python virtual environment for this project:

```cd Audio-Sentiment-Analysis```

```python3 -m venv venv```

This command creates a virtual environment called ```venv``` for this project.

To activate the virtual environment, make sure you're in the ```Audio-Sentiment-Analysis``` directory and enter the following command:

```source venv/bin/activate```

Once the virtual environment has been activated, we can install the dependencies into it.  To do this, run the command

```pip install -r requirements.txt```

Now all the dependencies needed for the project specified so far have been installed into the virtual environment

To install any further packages, ensure the virtual environment is activated before you use pip to install the dependencies.

To add newly installed dependencies to ```requirements.txt```, use the following command while in the ```Audio-Sentiment-Analysis``` directory:

```pip freeze > requirements.txt```

This command will add all the new dependencies along the specified versions to ```requirements.txt```

To deactivate the virtual environment, simply run the command ```deactivate```
