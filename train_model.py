import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from joblib import dump
import matplotlib.pyplot as plt
import joblib
from lazypredict.Supervised import LazyClassifier
from lightgbm import LGBMClassifier

data_path = r"C:\Users\naula\OneDrive\Desktop\Nasa Return 0\dataset\cleaned\cleaned_data.csv"
df = pd.read_csv(data_path)

y = df["koi_disposition"]  
X = df.drop(columns=["koi_disposition"]) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print("Model Performance Table:\n")
print(models)

best_model_name = models.index[0]
print(f"\nBest performing classifier: {best_model_name}")

best_model=None

if best_model_name=='LGBMClassifier':
    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.07,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    lgbm.fit(X_train, y_train)
    lgbm_preds = lgbm.predict(X_test)
    lgbm_acc = accuracy_score(y_test, lgbm_preds)

    print("LGBMClassifier Accuracy:", lgbm_acc)
    print(classification_report(y_test, lgbm_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, lgbm_preds))

    os.makedirs("../models", exist_ok=True)
    dump(lgbm, f"../models/best_model.pkl")

    importances = lgbm.feature_importances_

    best_model=lgbm

elif best_model_name=='RandomForestClassifier':
    rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)

    print("\nRandom Forest Accuracy:", rf_acc)
    print(classification_report(y_test, rf_preds))

    os.makedirs("../models", exist_ok=True)
    dump(rf_model, f"../models/best_model.pkl")

    importances = rf_model.feature_importances_

    best_model=rf_model

else:
    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)

    os.makedirs("../models", exist_ok=True)
    dump(xgb_model, f"../models/best_model.pkl")

    importances = xgb_model.feature_importances_

    best_model=xgb_model
    

feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
feat_imp.head(15).plot(kind="barh")
plt.title(f"Top 15 Features - {best_model}")
plt.show()
joblib.dump(best_model, r"C:\Users\naula\OneDrive\Desktop\Nasa Return 0\models\exoplanet_model.pkl")
print("Model saved at models/exoplanet_model.pkl")