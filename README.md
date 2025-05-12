# Machine Learning Debugging Workshop 🐞
Welcome! This repository contains the code for the two cases you will work on during the workshop.

Make sure you follow all the installation instructions in good time **before** the workshop (especially if you're on Windows, I only tested this on a Mac 😅). 

If you run into any problems open an issue in this repo and I'll do my best to answer asap.

Try your best not to look at the cases 🙈, leave that for the workshop. It's best that we all start from the same point then!


# Installation instructions
---
Note: If installing the project on a Windows machine, I recommend using [Git Bash](https://gitforwindows.org/#:~:text=and%20novices%20alike.-,Git%20BASH,-Git%20for%20Windows)

Clone this repository and move inside it:
```
git clone https://github.com/JulianoLagana/debugging_workshop.git
cd debugging_workshop
```

Install Python 3.10. This step is OS specific. For example, using brew on mac:
```
brew install python@3.10
```

Install Poetry using the [official installer](https://python-poetry.org/docs/#installing-with-the-official-installer):
```
curl -sSL https://install.python-poetry.org | python3 -
```

Install the project
```
poetry config virtualenvs.in-project true
poetry env use 3.10
poetry install
```

Activate the virtual environment you just created.

On Mac/Linux:
```
source .venv/bin/activate
```
On Windows, from a Git bash terminal:
```
source .venv/Scripts/activate
```

---

Test the hard failure case:
```
cd src/ml_debugging_workshop/hard_failure/
python preprocess.py
```
This should give you the following error message (at the end of the traceback)
```
ValueError: Input X contains infinity or a value too large for dtype('float64').
```

---

Test the soft failure case:
```
cd ../soft_failure/
dvc repro
```
This should preprocess the dataset, train a model, and evaluate it. You should see training performance being really good, while validation and test being close to random chance.
