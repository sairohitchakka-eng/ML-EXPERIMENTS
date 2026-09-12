import pandas as pd
import numpy as np
 
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from google.colab import files
 
uploaded = files.upload()
df = pd.read_csv("placement_predict_50k Dataset.csv")
 
print("Dataset shape:", df.shape)
df.head()

print(df.columns.tolist())
df.info()
print(df.isnull().sum())
print("PlacementStatus:")
print(df["PlacementStatus"].value_counts())
 
print("\nCGPA_Tier:")
print(df["CGPA_Tier"].value_counts())
exclude_cols = [
    "PlacementStatus",
    "CGPA_Tier",
    "Salary Package",
    "StudentID",
    "IsAnomaly"
]
 
X = df.drop(columns=exclude_cols, errors="ignore")
 
print("Number of features:", X.shape[1])
print(X.columns.tolist())
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()
 
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()
 
print("Numerical features:")
print(numeric_features)
 
print("\nCategorical features:")
print(categorical_features)
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)
 
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)
X_binary = df.drop(
    columns=[
        "PlacementStatus",
        "CGPA_Tier",
        "Salary Package",
        "StudentID",
        "IsAnomaly"
    ],
    errors="ignore"
)
 
y_binary = df["PlacementStatus"]
X_train_bin, X_val_bin, y_train_bin, y_val_bin = train_test_split(
    X_binary,
    y_binary,
    test_size=0.20,
    random_state=42,
    stratify=y_binary
)
 
print("Training samples:", X_train_bin.shape[0])
print("Validation samples:", X_val_bin.shape[0])
X_train_bin, X_val_bin, y_train_bin, y_val_bin = train_test_split(
    X_binary,
    y_binary,
    test_size=0.20,
    random_state=42,
    stratify=y_binary
)
 
print("Training samples:", X_train_bin.shape[0])
print("Validation samples:", X_val_bin.shape[0])
def evaluate_classifier(model, X_train, y_train, X_val, y_val):
    # Fit the model
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # Accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)

    print("-" * 70)
    print("MODEL EVALUATION")
    print("-" * 70)

    print(f"\nTrain Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    print("\nClassification Report - Validation Data")
    print("-" * 70)
    print(classification_report(y_val, val_pred))

    print("\nConfusion Matrix - Validation Data")
    print("-" * 70)
    print(confusion_matrix(y_val, val_pred))

    return {
        "model": model,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "train_predictions": train_pred,
        "val_predictions": val_pred
    }
binary_result = evaluate_classifier(
    binary_lr,
    X_train_bin,
    y_train_bin,
    X_val_bin,
    y_val_bin
)
X_multi = df.drop(
    columns=[
        "PlacementStatus",
        "CGPA_Tier",
        "Salary Package",
        "StudentID",
        "IsAnomaly"
    ],
    errors="ignore"
)
 
y_multi = df["CGPA_Tier"]
X_train_multi, X_val_multi, y_train_multi, y_val_multi = train_test_split(
    X_multi,
    y_multi,
    test_size=0.20,
    random_state=42,
    stratify=y_multi
)
 
print("Training samples:", X_train_multi.shape[0])
print("Validation samples:", X_val_multi.shape[0])
multi_lr = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42
        ))
    ]
)
multi_lr = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42
        ))
    ]
)
multi_result = evaluate_classifier(
    multi_lr,
    X_train_multi,
    y_train_multi,
    X_val_multi,
    y_val_multi
)
academic_features = [
    "SGPA_Sem1",
    "SGPA_Sem2",
    "SGPA_Sem3",
    "SGPA_Sem4",
    "SGPA_Sem5",
    "SGPA_Sem6",
    "SGPA_Sem7",
    "SGPA_Sem8",
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore"
]
 
academic_features = [
    col for col in academic_features
    if col in df.columns
]
 
print("Academic features:")
print(academic_features)
X_academic = df[academic_features]
y_academic = df["PlacementStatus"]
X_train_acad, X_val_acad, y_train_acad, y_val_acad = train_test_split(
    X_academic,
    y_academic,
    test_size=0.20,
    random_state=42,
    stratify=y_academic
)
academic_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)
academic_lr = Pipeline(
    steps=[
        ("preprocessor", academic_preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)
academic_result = evaluate_classifier(
    academic_lr,
    X_train_acad,
    y_train_acad,
    X_val_acad,
    y_val_acad
)
results_df = pd.DataFrame({
    "Model": [
        "Binary PlacementStatus LR",
        "Multinomial CGPA_Tier LR",
        "Academic PlacementStatus LR"
    ],
    "Train Accuracy": [
        binary_result["train_accuracy"],
        multi_result["train_accuracy"],
        academic_result["train_accuracy"]
    ],
    "Validation Accuracy": [
        binary_result["val_accuracy"],
        multi_result["val_accuracy"],
        academic_result["val_accuracy"]
    ]
})
 
results_df
print(results_df.to_string(index=False))
print(results_df.to_string(index=False))
top20_binary = coef_binary_df.head(20).sort_values(
    "Coefficient"
)
 
plt.figure(figsize=(10, 8))
 
plt.barh(
    top20_binary["Feature"],
    top20_binary["Coefficient"]
)
 
plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Feature")
plt.title("Top 20 Coefficients — PlacementStatus Logistic Regression")
 
plt.tight_layout()
plt.show()
feature_names_multi = (
    multi_lr
    .named_steps["preprocessor"]
    .get_feature_names_out()
)
 
multi_coefficients = (
    multi_lr
    .named_steps["classifier"]
    .coef_
)
 
print("Coefficient matrix shape:")
print(multi_coefficients.shape)
classes = multi_lr.named_steps["classifier"].classes_
 
for i, class_label in enumerate(classes):
 
    coef_df = pd.DataFrame({
        "Feature": feature_names_multi,
        "Coefficient": multi_coefficients[i]
    })
 
    coef_df["AbsoluteCoefficient"] = (
        coef_df["Coefficient"].abs()
    )
 
    coef_df = coef_df.sort_values(
        "AbsoluteCoefficient",
        ascending=False
    )
 
    top20 = coef_df.head(20).sort_values(
        "Coefficient"
    )
 
    plt.figure(figsize=(10, 8))
 
    plt.barh(
        top20["Feature"],
        top20["Coefficient"]
    )
 
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.title(
        f"Top 20 Coefficients — CGPA_Tier Class {class_label}"
    )
 
    plt.tight_layout()
    plt.show()
coef_plot = coef_binary_df.head(15).sort_values(
    "Coefficient"
)
 
plt.figure(figsize=(10, 7))
 
plt.barh(
    coef_plot["Feature"],
    coef_plot["Coefficient"]
)
 
plt.xlabel("Coefficient")
plt.title(
    "Top Logistic Regression Features for Placement Prediction"
)
 
plt.tight_layout()
plt.show()
print("\n\nMODEL 1: BINARY PLACEMENT STATUS")
binary_result = evaluate_classifier(
    binary_lr,
    X_train_bin,
    y_train_bin,
    X_val_bin,
    y_val_bin
)
 
 
print("\n\nMODEL 2: MULTINOMIAL CGPA TIER")
multi_result = evaluate_classifier(
    multi_lr,
    X_train_multi,
    y_train_multi,
    X_val_multi,
    y_val_multi
)
 
 
print("\n\nMODEL 3: ACADEMIC PLACEMENT STATUS")
academic_result = evaluate_classifier(
    academic_lr,
    X_train_acad,
    y_train_acad,
    X_val_acad,
    y_val_acad
)
