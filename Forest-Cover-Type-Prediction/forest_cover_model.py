import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib
import joblib

# Load the dataset
data = pd.read_csv('forest_cover_type.csv')

# Display basic info
print("Dataset shape:", data.shape)
print("Columns:", list(data.columns))
print("Target distribution:")
print(data['Cover_Type'].value_counts())

# Features and target
X = data.drop('Cover_Type', axis=1)
y = data['Cover_Type']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with aggressive size reduction
model = RandomForestClassifier(n_estimators=5, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save the trained model with maximum compression
joblib.dump(model, 'forest_cover_model.pkl', compress=9)
print("Model saved as 'forest_cover_model.pkl'")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Feature Importances:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance.head(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
