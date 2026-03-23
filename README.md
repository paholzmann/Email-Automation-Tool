# Email Classification API

A machine learning project that classifies incoming email text into
predefined categories through a FastAPI endpoint.

## Overview

Companies often receive large volumes of emails that need to be sorted
manually.\
This project builds a simple end-to-end ML system that takes raw email
text as input and returns a predicted category.

The system includes: - data loading - text preprocessing - model
training - model evaluation - model persistence - API-based inference

## Tech Stack

-   Python
-   FastAPI
-   scikit-learn
-   pandas
-   joblib
-   matplotlib
-   seaborn

## Problem

Manual email routing is slow and inefficient.

This project solves that by automatically classifying email text into
categories using a machine learning model exposed via an API.

## API Usage

Start the API:

uvicorn main:app --reload

POST /predict

Request: { "text": "I WANT A REFUND NOW!" }

Response: { "prediction": "refund" }

## What I Learned

-   End-to-end ML workflow
-   Text preprocessing
-   Model training & evaluation
-   Model persistence
-   FastAPI deployment

## Limitations

-   baseline model only
-   feature engineering not integrated yet
-   training & inference coupled
-   no tests

## Next Steps

-   better features
-   model tuning
-   separate training/inference
-   docker + deployment

## Author

Patrick-André Holzmann