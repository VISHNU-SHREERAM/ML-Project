from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from src.plotting import plot_residual

import numpy as np


def get_training_scores(df):
    tgt1 = df["Stress Level"]
    feat1 = df.drop(columns=["Stress Level", "Sleep Disorder"])

    scoreList = []

    for i in range(1, 5):
        poly = PolynomialFeatures(i)
        x = poly.fit_transform(feat1)

        X_train, X_test, y_train, y_test = train_test_split(
            x, np.array(tgt1), random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        scoreList.append(r2_score(y_test, pred))

    return scoreList


def regress_linear(feat, tgt):
    poly = PolynomialFeatures(3)
    x = poly.fit_transform(feat)

    X_train, X_test, y_train, y_test = train_test_split(
        x, np.array(tgt), random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("r2_score =", r2_score(y_test, pred))
    print("mean_squared_error =", mean_squared_error(y_test, pred))


def regress_bagging(X_train, y_train, X_test, y_test):
    regr = BaggingRegressor(estimator=SVR(), n_estimators=80, random_state=0).fit(
        X_train, y_train
    )

    y_pred = regr.predict(X_test)
    print("mean squared error", mean_squared_error(y_test, y_pred))


def regress_rf(X_train, y_train, X_test, y_test):
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train, y_train)

    y_pred_reg = rf_regressor.predict(X_test)
    mse_reg = mean_squared_error(y_test, y_pred_reg)
    r2_reg = r2_score(y_test, y_pred_reg)

    print("mean squared error: ", mse_reg)
    print("r2-score: ", r2_reg)

    # Calculate the differences (residuals)
    residuals = y_test - y_pred_reg
    return residuals, y_pred_reg


def compare_regs(X_train, y_train, X_test, y_test, regressors):
    # Evaluate each regressor
    for name, model in regressors.items():
        model.fit(X_train, y_train)
        y_pred_reg = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_reg)
        r2 = r2_score(y_test, y_pred_reg)
        print(f"{name} Regressor - MSE: {mse:.4f}, R2 Score: {r2:.4f}")
        residuals = y_test - y_pred_reg

        plot_residual(y_test, y_pred_reg, residuals)
