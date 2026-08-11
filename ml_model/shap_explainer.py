import joblib
import shap
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"ml_model\models\improved_random_forest.pkl"
DATA_PATH = r"dataset\UNSW_NB15_testing_encoded.csv"
OUTPUT_PATH = r"ml_model\shap_explanation.csv"


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("Loading Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully.")
print("Model expects", model.n_features_in_, "features.")


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading testing dataset...")

df = pd.read_csv(DATA_PATH)

print("Testing dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 3. GET EXACT MODEL FEATURES
# ============================================================

feature_names = list(model.feature_names_in_)

print("\nSelecting model features...")

missing_features = [
    feature for feature in feature_names
    if feature not in df.columns
]

if missing_features:
    print("ERROR: Missing features:")
    print(missing_features)
    raise ValueError("Required model features are missing.")

X = df[feature_names].copy()

print("Features selected successfully.")
print("Feature shape:", X.shape)


# ============================================================
# 4. SELECT SAMPLE
# ============================================================

sample = X.iloc[[0]].copy()

print("\nSample selected for SHAP explanation.")


# ============================================================
# 5. PREDICTION
# ============================================================

prediction = model.predict(sample)[0]

probabilities = model.predict_proba(sample)[0]

confidence = float(np.max(probabilities) * 100)


print("\n================================")
print("       AI PREDICTION")
print("================================")

print("Prediction :", prediction)
print(f"Confidence : {confidence:.2f}%")

if prediction == 1:
    print("Threat     : ATTACK")
else:
    print("Threat     : NORMAL")


# ============================================================
# 6. CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(model)

print("SHAP explainer created successfully.")


# ============================================================
# 7. CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")

shap_values = explainer.shap_values(sample)

shap_array = np.asarray(shap_values)

print("SHAP values calculated successfully.")
print("SHAP output shape:", shap_array.shape)


# ============================================================
# 8. EXTRACT ATTACK CLASS SHAP VALUES
# ============================================================

number_of_features = len(feature_names)


# Your SHAP output is:
#
# (1, 42, 2)
#
# 1  = sample
# 42 = features
# 2  = classes
#
# Therefore:
#
# shap_array[0, :, 0] = Normal contribution
# shap_array[0, :, 1] = Attack contribution
#
# We want class 1 = Attack.

if shap_array.ndim == 3:

    if (
        shap_array.shape[0] == 1
        and shap_array.shape[1] == number_of_features
        and shap_array.shape[2] == 2
    ):

        shap_array = shap_array[0, :, 1]

    else:

        raise ValueError(
            f"Unexpected 3D SHAP shape: {shap_array.shape}"
        )


# Some SHAP versions may return:
#
# (42, 2)

elif shap_array.ndim == 2:

    if (
        shap_array.shape[0] == number_of_features
        and shap_array.shape[1] == 2
    ):

        shap_array = shap_array[:, 1]

    elif (
        shap_array.shape[0] == 1
        and shap_array.shape[1] == number_of_features
    ):

        shap_array = shap_array[0]

    else:

        raise ValueError(
            f"Unexpected 2D SHAP shape: {shap_array.shape}"
        )


# Some versions may already return:
#
# (42,)

elif shap_array.ndim == 1:

    if len(shap_array) != number_of_features:

        raise ValueError(
            f"Unexpected SHAP value count: {len(shap_array)}"
        )

else:

    raise ValueError(
        f"Unexpected SHAP dimensions: {shap_array.ndim}"
    )


# ============================================================
# 9. FINAL SHAP ARRAY
# ============================================================

shap_array = np.asarray(shap_array).reshape(-1)

print("\nFeature count :", number_of_features)
print("SHAP values   :", len(shap_array))


if len(shap_array) != number_of_features:

    raise ValueError(
        "SHAP values do not match model feature count."
    )


# ============================================================
# 10. FEATURE VALUES
# ============================================================

feature_values = sample.iloc[0].to_numpy()

feature_values = np.asarray(
    feature_values
).reshape(-1)


# ============================================================
# 11. CREATE EXPLANATION TABLE
# ============================================================

explanation = pd.DataFrame(
    {
        "feature": feature_names,
        "feature_value": feature_values,
        "shap_value": shap_array
    }
)


# ============================================================
# 12. ABSOLUTE IMPACT
# ============================================================

explanation["absolute_impact"] = (
    explanation["shap_value"].abs()
)


# ============================================================
# 13. SORT BY IMPORTANCE
# ============================================================

explanation = explanation.sort_values(
    by="absolute_impact",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 14. DISPLAY TOP FEATURES
# ============================================================

print("\n================================")
print("TOP CONTRIBUTING FEATURES")
print("================================")

for index, row in explanation.head(10).iterrows():

    print(
        f"{index + 1:2}. "
        f"{row['feature']:25} "
        f"Value: {str(row['feature_value']):12} "
        f"SHAP: {row['shap_value']:+.6f}"
    )


# ============================================================
# 15. INTERPRETATION
# ============================================================

print("\n================================")
print("SHAP INTERPRETATION")
print("================================")

for _, row in explanation.head(10).iterrows():

    feature = row["feature"]
    shap_value = row["shap_value"]

    if shap_value > 0:

        print(
            f"{feature}: "
            f"contributes toward ATTACK "
            f"(SHAP {shap_value:+.6f})"
        )

    elif shap_value < 0:

        print(
            f"{feature}: "
            f"contributes toward NORMAL "
            f"(SHAP {shap_value:+.6f})"
        )

    else:

        print(
            f"{feature}: "
            f"little/no contribution "
            f"(SHAP {shap_value:+.6f})"
        )


# ============================================================
# 16. SAVE EXPLANATION
# ============================================================

explanation.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSHAP explanation saved successfully:")
print(OUTPUT_PATH)


# ============================================================
# 17. COMPLETED
# ============================================================

print("\n================================")
print("SHAP EXPLANATION COMPLETED")
print("================================")