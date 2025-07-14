

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
df = pd.read_csv("student_dropout.csv")
print("✅ Loaded dataset:", df.shape)

# Step 2: Impute missing values
for col in ["Subscription_Type", "Course_Difficulty", "Device_Used"]:
    df[col].fillna(df[col].mode()[0], inplace=True)
df["Satisfaction_Rating"].fillna(df["Satisfaction_Rating"].median(), inplace=True)

# Step 3: Feature Engineering - Add temporal metrics
df["Enrollment_Date"] = pd.to_datetime(df["Enrollment_Date"], errors='coerce')
df["Activity_Start_Date"] = pd.to_datetime(df["Activity_Start_Date"], errors='coerce')
df["Last_Login_Date"] = pd.to_datetime(df["Last_Login_Date"], errors='coerce')
analysis_reference_date = pd.to_datetime("2025-07-10")

df["Days_Between_Enrollment_and_Activity"] = (
    df["Activity_Start_Date"] - df["Enrollment_Date"]
).dt.days.fillna(0)

df["Days_Since_Last_Login"] = (
    analysis_reference_date - df["Last_Login_Date"]
).dt.days.fillna(0)

df["Progress_Bucket"] = pd.cut(
    df["Course_Progress_%"], bins=[0, 20, 50, 100], labels=["Low", "Moderate", "High"]
)

# Step 4: Risk flag feature
df["High_Risk_Flag"] = (
    (df["Course_Progress_%"] < 20) &
    (df["Time_Spent_Total_Min"] < 60) &
    (df["Satisfaction_Rating"] <= 2) &
    (df["Days_Since_Last_Login"] > 10)
)

# Step 5: Drop non-numeric & raw ID columns before training
drop_cols = [
    "Student_ID", "Course_ID", "Enrollment_Date",
    "Activity_Start_Date", "Last_Login_Date", "Progress_Bucket"
]
df_model = df.drop(columns=drop_cols)

# Step 6: Encode categorical columns
label_cols = ["Device_Used", "Subscription_Type", "Course_Difficulty", "Country", "Dropout"]
encoder = LabelEncoder()
for col in label_cols:
    df_model[col] = encoder.fit_transform(df_model[col].astype(str))

# Step 7: Train-test split & model
X = df_model.drop("Dropout", axis=1)
y = df_model["Dropout"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Summary Insights
print("\n📊 Dropout Rate (%):")
print(df["Dropout"].value_counts(normalize=True).round(3) * 100)

print("\n📉 Dropout Rate by Category:")
for col in ["Subscription_Type", "Course_Difficulty", "Device_Used"]:
    dropout_by_cat = df.groupby(col)["Dropout"].value_counts(normalize=True).unstack().fillna(0)
    print(f"\n{col}:\n", (dropout_by_cat * 100).round(2))

print("\n🚨 High-risk student count flagged:", df["High_Risk_Flag"].sum())

# Step 9: Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = feature_importance.head(5)
print("\n🌟 Top 5 Features Influencing Dropout:\n", top_features)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top Features Affecting Dropout")
plt.tight_layout()
plt.show()

# Step 10: Strategic Recommendations Table
recommendations = pd.DataFrame({
    "Insight Area": [
        "Low engagement cohorts",
        "Hard courses with churn",
        "Device dropout patterns",
        "Satisfaction dip zones",
        "At-risk student profile"
    ],
    "Actionable Strategy": [
        "Send re-engagement nudges after 7 days inactivity",
        "Add guided walkthroughs or simplified exercises",
        "Enhance mobile UX to improve retention",
        "Embed feedback forms mid-course",
        "Create alert system for early intervention"
    ]
})
print("\n📋 Strategic Recommendations:\n")
print(recommendations.to_string(index=False))

