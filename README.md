# ium-long-stay-patterns

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project implements a service that predicts whether a given property is more likely to be booked for short-term or long-term stays. The system includes a complete data analysis pipeline, dataset preparation, and training of machine learning models using a previously validated best set of hyperparameters. The project contains two baseline models: a naive reference model and a target binary classification model that performs the actual prediction. The service allows automatic inference based on input data and evaluation of model performance.

# Setup

## Necessary
1. python 3.13
2. poetry
3. make

## Environment activation
    eval $(poetry env activate)

## Install dependencies
    make requirements

## start prediction service
*dockerized:*

    make run
*local:*

    make dev

*other commands are self-documented in **Makefile***

## Project Organization

```
├── LICENSE                    <- Open-source license if one is chosen
├── Makefile                   <- Makefile with convenience commands
├── README.md                  <- The top-level README for developers
├── pyproject.toml             <- Project configuration file with package metadata and dependencies(poetry)
├── setup.cfg                  <- Configuration file for flake8 and other tools
│
├── data                       <- Data directory
│   ├── external               <- Data from third party sources (compressed datasets)
│   ├── interim                <- Intermediate transformed data
│   ├── processed              <- The final, canonical data sets for modeling
│   └── raw                    <- The original, immutable data dump
│
├── models                     <- Model code for different architectures
│   ├── binary.py              <- Binary classification model implementation
│   ├── naive.py               <- Naive baseline model
│   └── xgBoost.py             <- XGBoost model placeholder
│
├── saved_models               <- Trained and serialized models and scalers
│   ├── binary_classifier_model.pth
│   ├── binary_scaler.joblib
│   └── naive_classifier_model.pth
│
├── notebooks                  <- Jupyter notebooks for exploration and analysis
│   ├── amenities_processing.ipynb
│   ├── amenities_standarisation.ipynb
│   ├── analyze_numeric_data.ipynb          <- set up dataset for training on mostly numerical values
│   ├── classification_numeric_data.ipynb   <- Training models and summary
│   ├── listing_stats.ipynb                 <- creation of dataset with listing statistics
│   └── parameters_tuning.ipynb             <- hiperparameters tuning for binary classifier
│
├── prediction_service         <- FastAPI prediction service
│   ├── app.py                 <- Main FastAPI application
│   ├── ab_test.py             <- A/B testing utilities
│   ├── analyze_logs.py        <- Log analysis utilities
│   ├── test.py                <- Service tests
│   ├── Dockerfile             <- Docker configuration for containerization
│   ├── dto                    <- Data transfer objects
│   ├── logs                   <- Service logs
│   └── test_data              <- Test datasets and fixtures
│
├── reports                    <- Generated analysis and reports
│   ├── analyze_logs.md         <- Analyze and summary logs of A/B test
│   ├── naive_results_bookings.md
│   ├── naive_results_listings.md
│   ├── stage_1_outline.md
│   ├── figures                <- Generated graphics and figures
│   │   ├── metrics_summary_comparison.csv
│   │   ├── metrics_summary.csv
│   │   └── *.json files
│   └── tasks                  <- Project tasks and milestones
│
├── tests                      <- Unit and integration tests
│   └── test_data.py           <- Data loading tests
│
└── ium_long_stay_patterns     <- Source code package
    ├── __init__.py            <- Makes ium_long_stay_patterns a Python module
    ├── config.py              <- Store useful variables and configuration
    ├── dataset.py             <- Scripts to download or generate data
    ├── features.py            <- Code to create features for modeling
    ├── plots.py               <- Code to create visualizations
    │
    ├── modeling               <- Modeling pipeline
    │   ├── __init__.py
    │   ├── predict.py         <- Code to run model inference with trained models
    │   └── train.py           <- Code to train models
    │
    └── src                    <- Core utilities and helpers
        ├── main.py            <- Main execution script
        ├── correlation_matrix.py
        ├── naive_solver.py
        ├── run_naive.py
        ├── helper_methods.py
        ├── helpers           <- Helper modules
        │   ├── create_listing_stats_dataset.py
        │   ├── create_numerical_dataset.py
        │   ├── data_loaders.py
        │   └── plotting.py
        └── split_file.sh      <- Data splitting utility script
```

--------

