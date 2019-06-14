# UC Berkeley CS182 HW03: Natural Language Processing

Welcome to the third homework of CS182, in this assignment, you will learn about processing and generating text. Specifically, you will build a neural network to generate news headlines, through the training of an LSTM-based language model. Then you will train a Transformer to summarize news articles.

# Deliverables

To complete each assignment, you must fill in, the two provided Jupyter notebooks with your solution, this might lead you to edit Python files in this folder.

Through the Notebook, you will produce 2 model files in the .ckpt format. These are part of your deliverables and should be uploaded with your project, as they will be tested against an **unreleased test set** for each notebook.

To prepared your ZIP deliverable, read and follow instructions in the "Preparing a Submission" section.


# Installation

This assignment requires Python 3. If you plan to complete this assignment with a local machine, and not in the Google Colab, please install Python 3.

## Linux / MAC Os

Installing on Linux is very similar to the previous assignments. You can run the following:
```bash
virtualenv .env --python=python3
````
To create a virtual environment. Running:
```bash
source .env/bin/activate
```
will install the python requirements for the assignment. **Make sure that you are running a version of python >= 3.5**. If you are not (which can be checked with `python3 --version`) install an updated version of python using the instructions here: https://www.tecmint.com/install-python-in-ubuntu/

Open one of the .ipynb assignment files through Jupyter in a browser and make sure you are using the right Python environment by going into the "Kernel" tab "change kernel" and selecting the correct kernel to run the code in (Python 3).

You now must install the required packages for the assignment. If you are using a GPU-enabled machine:
```bash
pip3 install -r requirements_gpu.txt
```
Otherwise:
```bash
pip3 install -r requirements.txt
```

## Windows

Windows is not officially supported. There is a chance that you will be able to get this working with Anaconda on windows, however we strongly suggest using a virtual machine (virtual box) with an installation of linux, as it will likely be more stable, and the GSIs will be able to provide additional assistance. 

## Colab

You can use our modified Notebooks for the Google Colab. Google Colab notebooks can be used with a CPU, or with a K-80 GPU (on this assignment, it seems to give a 5x training speed boost).
The Notebooks we have shared on Google Colab have been modified so that it is easy to access the datasets in your Google Drive, and save the model files. If you opt for this option, you will have to duplicate the folder on Google Drive and work on the assignment in your copy.
However, you must recuperate the models, and verify that they work locally before submitting your files.
he dataset files are available for download in this folder: https://drive.google.com/open?id=1TNhUy9ldZ5mv_GLNNmCBFnLfT3DXwntF
You must be logged in on your UC Berkeley Google account to see the folder, and you must duplicate it (copy it to your own drive) to have write access over the files. The .ipynb can be opened into the Google Colab by double clicked and selecting "Open with Colaboratory".

# Downloading the Data

To download the data, run the following command from the assignment root directory:
```bash
bash download_data.sh
```
If you get the error "bash: ./download_data.sh: Permission denied" run `chmod +x download_data.sh` and try again. 

# Preparing a Submission

To prepare a submission, run the following command from the assignment root directory:
```bash
bash prepare_submission.sh
```
This will create a "submission.zip" file which can be submitted for the homework. You may get a warning if the files required do not exist - please pay attention to this, as we will not be responsible if you forget to include these files in the zipped folder.
Check that your submission contains:
- "1 Language Modeling.ipynb" and "2 Summarization.ipynb"
- "capita.py", "transformer.py", "transformer_attention.py", "transformer_layers.py"
- Your model files (.index, .cptk, .data) for each notebook. The names of the models you upload should match the names you put in the Jupyter notebooks. This should be a total of 6 files.

If you get the error "bash: ./prepare_submission.sh: Permission denied" run `chmod +x download_data.sh` and try again. 

# Questions

Ask your questions on Piazza.
If you have questions about the creation of the dataset or the assignment, contact Philippe Laban (phillab@berkeley.edu) or John Canny (canny@berkeley.edu).
If you are an instructor looking for the unreleased test-set and Solution notebooks, you can contact us as well.
