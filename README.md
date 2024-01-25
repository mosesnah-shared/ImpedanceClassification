# Project Title

This code is for running Machine Learning codes for Binary Classification, Autoencoder methods and others. 
The main project is Impedance Classification from Lachner, Tessari, West and Nah.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install the software and how to install them:

- Python 3.x
- pip (Python package installer)

### Installing

A step by step series of examples that tell you how to get a development environment running.

#### Setting up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies for your project. This keeps your global site-packages directory clean and manages your project's dependencies separately.

1. **Install `venv`** (if you don't have it installed)

    ```bash
    python3 -m pip install --user virtualenv
    ```

2. **Create a Virtual Environment**

    Navigate to your project directory in the terminal and run:

    ```bash
    python3 -m venv venv
    ```

    This will create a directory named `venv` in your project directory. Inside, it will install a local version of Python and a local version of `pip`. You can use this local version to install and manage packages specific to your project.

3. **Activate the Virtual Environment**

    Before you can start installing or using packages in your virtual environment you'll need to activate it:

    - On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    - On Windows:
        ```cmd
        .\venv\Scripts\activate
        ```

    You can tell that you have the virtual environment activated if your terminal prompt is prefixed with `(venv)`. 

#### Installing Dependencies

With your virtual environment active, you can install the required packages for the project.

1. **Install Dependencies**

    Ensure you're in the project root directory and run:

    ```bash
    pip install -r requirements.txt
    ```

    This will install all the packages listed in the `requirements.txt` file.

## Running the Project

After setting up the virtual environment and installing the dependencies, you're ready to run the project. Include specific instructions on how to run your project here.

For example, to run binary classification:
```bash
python binaryclassification.py
