# 🫀 Heart Disease Prediction using Decision Trees & Random Forests

This machine learning project demonstrates how to build a predictive model for heart disease using **Decision Trees** and **Random Forests**, along with techniques for analyzing overfitting, interpreting feature importance, and evaluating models via cross-validation.

---

## 📁 Dataset

The dataset includes the following columns:

- `age`: Age of the patient  
- `sex`: Sex (1 = male, 0 = female)  
- `cp`: Chest pain type (0–3)  
- `trestbps`: Resting blood pressure  
- `chol`: Serum cholesterol (mg/dl)  
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)  
- `restecg`: Resting electrocardiographic results (0–2)  
- `thalach`: Maximum heart rate achieved  
- `exang`: Exercise-induced angina (1 = yes; 0 = no)  
- `oldpeak`: ST depression induced by exercise  
- `slope`: Slope of peak exercise ST segment  
- `ca`: Number of major vessels colored by fluoroscopy (0–3)  
- `thal`: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)  
- `target`: Heart disease (1 = present; 0 = not present)

---

## 🚀 Workflow Steps

### 1️⃣ Train a Decision Tree Classifier

- Use `DecisionTreeClassifier` to train on the dataset.
- Visualize the tree using `plot_tree`.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
plt.show()

#Analyze Overfitting & Prune Tree Depth
- A fully grown tree may overfit (perfect on training, poor on testing).
- Prune with max_depth to reduce overfitting.

from sklearn.metrics import accuracy_score

# Full tree
clf_full = DecisionTreeClassifier(random_state=42)
clf_full.fit(X_train, y_train)

# Pruned tree
clf_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_pruned.fit(X_train, y_train)

# Compare accuracies
train_acc_full = accuracy_score(y_train, clf_full.predict(X_train))
test_acc_full = accuracy_score(y_test, clf_full.predict(X_test))

train_acc_pruned = accuracy_score(y_train, clf_pruned.predict(X_train))
test_acc_pruned = accuracy_score(y_test, clf_pruned.predict(X_test))

# Train a Random Forest Classifier
- Use RandomForestClassifier for improved generalization.

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

train_acc_rf = accuracy_score(y_train, rf_clf.predict(X_train))
test_acc_rf = accuracy_score(y_test, rf_clf.predict(X_test))

#Interpret Feature Importances
- Random Forest models can identify the most important features.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

importances = rf_clf.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
colors = plt.cm.Reds(np.linspace(0.4, 1, len(feat_importances)))
feat_importances.plot(kind='bar', color=colors)
plt.title("Feature Importances (Random Forest)")
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Evaluate with Cross-Validation
- Use cross_val_score to measure generalization accuracy across folds.

from sklearn.model_selection import cross_val_score
import numpy as np

cv_scores_rf = cross_val_score(rf_clf, X, y, cv=5)
cv_scores_dt = cross_val_score(clf_pruned, X, y, cv=5)

print(f"Random Forest CV Mean: {cv_scores_rf.mean():.3f}")
print(f"Decision Tree CV Mean: {cv_scores_dt.mean():.3f}")

#Visualize Cross-Validation Results
- Show average CV score and standard deviation for each model.

means = [cv_scores_dt.mean(), cv_scores_rf.mean()]
stds = [cv_scores_dt.std(), cv_scores_rf.std()]
models = ['Decision Tree', 'Random Forest']

colors = plt.cm.Blues(np.linspace(0.5, 1, len(models)))

plt.figure(figsize=(8, 6))
plt.bar(models, means, yerr=stds, capsize=10, color=colors)
plt.title('Mean CV Accuracy with Std Dev')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

#Results Summary
- Decision Tree (full depth): High training accuracy, low test accuracy → overfitting

- Pruned Tree (max_depth=3): Better generalization

- Random Forest: Best performance due to ensemble averaging

- Feature Importances: Highlights which attributes most influence predictions

- Cross-validation: Confirms consistent model performance across folds

