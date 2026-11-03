import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.base import clone
# 1. Load Dataset
df = pd.read_csv(
    "C:/Users/nandu/Downloads/india_road_accidents_2025.csv"
)
print("First 5 Rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())
# 2. Data Preprocessing
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

df["Hour"] = df["Timestamp"].dt.hour
df["Month"] = df["Timestamp"].dt.month
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

df["Accident_Severity"] = np.where(
    df["Fatalities"] > 0, 1, 0
)

print("\nAccident Severity Distribution:")
print(df["Accident_Severity"].value_counts())


# 3. Define Features and Target

X = df[
    [
        "State",
        "Road_Type",
        "Primary_Vehicle",
        "Primary_Cause",
        "Weather_Condition",
        "Speed_Limit_KMH",
        "Hour",
        "Month",
        "DayOfWeek"
    ]
]
y = df["Accident_Severity"]
# 4. Identify Numerical and Categorical Columns
numerical_features = [
    "Speed_Limit_KMH",
    "Hour",
    "Month",
    "DayOfWeek"
]
categorical_features = [
    "State",
    "Road_Type",
    "Primary_Vehicle",
    "Primary_Cause",
    "Weather_Condition"
]
# 5. Preprocessing Pipelines
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
# 6. Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
# 7. Define Five Machine Learning Models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}
# 8. Train Models and Evaluate Performance
results = []
trained_models = {}
for name, model in models.items():
    print("\nTraining:", name)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", clone(model))
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })
    trained_models[name] = pipeline
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))
    # 9. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title(name + " - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
# 10. Create Results DataFrame
results_df = pd.DataFrame(results)
print("\n========== MODEL PERFORMANCE ==========")
print(results_df.round(4))
# 11. Accuracy Comparison Graph
plt.figure(figsize=(10, 5))
sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)
plt.title("Accuracy Comparison of Five Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
# 12. F1 Score Comparison Graph
plt.figure(figsize=(10, 5))
sns.barplot(
    data=results_df,
    x="Model",
    y="F1 Score"
)
plt.title("F1 Score Comparison of Five Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("F1 Score")
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
# 13. Performance Metrics Comparison
metrics_df = results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
]
metrics_df.plot(
    kind="bar",
    figsize=(12, 6)
)
plt.title("Performance Metrics Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
# 14. Find Best and Worst Models
best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]
worst_model = results_df.loc[
    results_df["F1 Score"].idxmin()
]
print("\n========== BEST MODEL ==========")
print("Best Model:", best_model["Model"])
print("Accuracy:", round(best_model["Accuracy"], 4))
print("Precision:", round(best_model["Precision"], 4))
print("Recall:", round(best_model["Recall"], 4))
print("F1 Score:", round(best_model["F1 Score"], 4))
print("ROC-AUC:", round(best_model["ROC-AUC"], 4))
print("\n========== WORST MODEL ==========")
print("Worst Model:", worst_model["Model"])
print("Accuracy:", round(worst_model["Accuracy"], 4))
print("Precision:", round(worst_model["Precision"], 4))
print("Recall:", round(worst_model["Recall"], 4))
print("F1 Score:", round(worst_model["F1 Score"], 4))
print("ROC-AUC:", round(worst_model["ROC-AUC"], 4))
# 15. Best and Worst Model Comparison Graph
comparison_df = pd.DataFrame({
    "Model": [
        best_model["Model"],
        worst_model["Model"]
    ],
    "F1 Score": [
        best_model["F1 Score"],
        worst_model["F1 Score"]
    ]
})
plt.figure(figsize=(8, 5))
sns.barplot(
    data=comparison_df,
    x="Model",
    y="F1 Score"
)
plt.title("Best and Worst Model Comparison")
plt.xlabel("Models")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
# 16. Final Conclusion
print("\n========== FINAL CONCLUSION ==========")
print("Best Performing Model:", best_model["Model"])
print("Worst Performing Model:", worst_model["Model"])
print(
    "Best Model F1 Score:",
    round(best_model["F1 Score"], 4)
)
print(
    "Worst Model F1 Score:",
    round(worst_model["F1 Score"], 4)
)
print("\nAll Five Models Trained and Evaluated Successfully!")