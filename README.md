Cell Painting
==============================

Deep learning approach for compound similarity analysis. \
For deep insight into our work please read pdf file contained in _reports_ folder



Getting Started
------------

```sh
# Bootstrap the Python Virtual Environment
python -m venv .venv
# NOTE: PyTorch currently supports only Python 3.7, 3.8 and 3.9
# You may need to use one of those versions to create the venv

# Remember to activete it. E.g. in bash it is:
source .venv/bin/activate

# Now you can install all the necessary Python packages
pip install -r requirements.txt

# To make Jupyter see the packages from venv, install a local kernel
pip install ipykernel
# refresh the PATH cache
hash -r
# and create the appropriate kernelspec
ipython kernel install \
	--user
	--name cell-painting \
	--display-name "Python (Cell Painting)"

# If you don't want to leak the venv into the user configuration, replace
# `--user` flag with `--prefix $VIRTUAL_ENV` and make the venv (de)activate
# script also (un)set the `JUPYTER_PATH` environmental variable so that
# it contains `$VIRTUAL_ENV/share/jupyter` only when the venv is activated.
# If you prefer, you can also have it set up that way permanently.
```

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        └── models         <- Scripts to train models and then use trained models to make
            │                 predictions
            ├── predict_model.py
            └── train_model.py

--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
