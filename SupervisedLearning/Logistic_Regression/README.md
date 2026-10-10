# Bank Customer Churn Predictor

A Streamlit app that estimates bank customer churn risk with a logistic regression model.

## Setup

Run these commands from this project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train the Model

```bash
python src/cleaned_data.py
python src/train.py
```

The preprocessing script saves the transformed dataset and scaler. The training script fits the model with balanced class weights and saves it under `models/`.

## Run the App

```bash
streamlit run app.py
```

Enter a customer's account details and select **Calculate churn risk** to see the model's predicted churn probability. The app also displays holdout metrics and a confusion matrix.

## Model Notes

On the current stratified 80/20 holdout split, the model achieved 71.4% accuracy, 70.0% churn recall, 38.7% churn precision, and a 0.777 ROC-AUC. Balanced class weights help identify more churners, but they also increase false positives. Treat predictions as a review signal, not a final decision.
