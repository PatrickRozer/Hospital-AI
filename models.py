import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models

def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_xgb(X_train, y_train):
    model = XGBClassifier(n_estimators=200, learning_rate=0.05, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model

def build_nn(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model

def evaluate(model, X_test, y_test, keras=False):
    if keras:
        y_pred_proba = model.predict(X_test).ravel()
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        y_pred_proba = model.predict_proba(X_test)[:,1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba)
    }
