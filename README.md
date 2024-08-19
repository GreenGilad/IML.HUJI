# Introduction to Machine Learning

"Introduction to Machine Learning" serves as an entry point into the dynamic field of machine and statistical learning. Aimed at undergraduate computer science students, the book builds on a solid foundation in mathematics (calculus, linear algebra, and probability theory) and programming, covering essential topics for aspiring machine learning experts. It addresses key concepts such as estimation theory, regression and classification models, boosting and regularization techniques, kernel methods, unsupervised learning, gradient-based methods, and neural networks, alongside the theoretical aspects of statistical learning.

<div style="text-align: center;"><img src="https://github.com/user-attachments/assets/35c93c33-ff85-43c6-b860-3ed1c786284c" height=400/></div>

Our approach integrates formal mathematical proofs with high-quality code examples, helping students grasp not only the theory behind the models but also how they work, why they perform as they do, when they might fail, and how to fine-tune parameters for optimal results. This learning experience is further enhanced by hands-on practice in programming, fitting, and evaluating models.

This comprehensive resource includes:
1) An online version of the book with links to code examples and animations illustrating key topics (for animations to work, use a PDF reader that supports animated figures, such as Adobe).
2) Hands-on Guided Labs - Enabling readers to explore the concepts discussed in the book.
3) Code Examples and Graph Generation Scripts - Featured throughout the book.
4) Skeleton Files for the IMLearn Software Package - Providing functions and classes to be implemented while reading. Upon completion, this skeleton transforms into a fully functioning machine learning software package.





## Setting Up Code and Labs Environment
To experience the code examples and guided labs follow these instructions to set up the conda environment. Set a local copy of this GitHub repository. Do so by [forking and cloning the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo) or [cloning the repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) using GitBash and 
```
cd LOCAL_REPOSITORY_PATH
git clone https://github.com/GreenGilad/IML.HUJI.git
```

or by downloading and unzipping it in `LOCAL_REPOSITORY_PATH`. Then:

### Anaconda + Jupyter Notebooks Setup
- Download and install Anaconda from [official website](https://www.anaconda.com/products/individual#Downloads). 
- Verify instllation by starting the Anaconda Prompt. A terminal should start with the text `(base)` written at the beginning of the line.
- Set the IML conda environment. Start the Anaconda Prompt and write:
  ```
  conda env create -f "LOCAL_REPOSITORY_PATH\environment.yml"
  ```
  This will create a conda envoronment named `iml.env` with the specifications defined in `environment.yml`. If creating failed due to `ResolvePackageNotFound: plotly-orca` remove this line from environment file, create environment without, and then after activating environment run:
  ```
  conda install -c plotly plotly-orca
  ```
- Activate the environment by `conda activate iml.env`.
- To open one of the Jupyter notebooks:
  ```
  jupyter notebook "LOCAL_REPOSITORY_PATH\lab\Lab 00 - A - Data Analysis In Python - First Steps.ipynb"
  ```

### Using PyCharm
Another option is to run the Jupyter notebooks through the PyCharm IDE plug-in. 
- Install the PyCharm IDE (professional edition) as described on the [Install PyCharm](https://www.jetbrains.com/help/pycharm/installation-guide.html) page. Be sure to install the Jupyter plug-in.
- Follow the [Configure a Conda virtual environment](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html#conda-requirements) page.
- Open a PyCharm project from `LOCAL_REPOSITORY` and set the project's interpreter to be the `iml.env` environment.

### Using Google Colab
One can also view and run the labs and code examples via Google Colab. It supports loading and running Jupyter notebooks and running using a specified Conda environemnt.
