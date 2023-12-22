import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.svm import SVC

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
pd.set_option('display.max_columns', None)

# look at dataframe
#print(df.head(5))
#print(df.shape)
print(df.columns)

# check for missing values
#print(df.isna().sum().sort_values())

# viewing value proportions in binary variables
df['anaemia'].value_counts()
df['diabetes'].value_counts()
df['high_blood_pressure'].value_counts()
df['sex'].value_counts()
df['smoking'].value_counts()
df['DEATH_EVENT'].value_counts()

#print(df.describe())

# titanic[["Sex", "Age"]].groupby("Sex").mean()
# titanic.groupby("Sex").mean(numeric_only=True)
# titanic.groupby("Sex")["Age"].mean()
# titanic.groupby(["Sex", "Pclass"])["Fare"].mean()
# titanic["Pclass"].value_counts()
# titanic.groupby("Pclass")["Pclass"].count()

# check for multicollinearity
corr_mat = sns.heatmap(df.corr(), annot=True)
#plt.show()

y = df['DEATH_EVENT']
X = df.drop(['time', 'DEATH_EVENT'], axis=1)

# look at relationship between predictors and outcome
sns.pairplot(df,  kind='scatter', hue='DEATH_EVENT') #vars=['Feature1', 'Feature2', 'Feature3', 'Feature4'],
plt.suptitle('Pairplot of Predictors vs Target', y=1.02)
sns.pairplot(df, vars=['creatinine_phosphokinase', 'ejection_fraction',
                       'platelets', 'serum_creatinine', 'serum_sodium'], kind='scatter', hue='DEATH_EVENT') #vars=['Feature1', 'Feature2', 'Feature3', 'Feature4'],
plt.suptitle('Pairplot of Predictors vs Target', y=1.02)
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 200, 350, 500],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

svm = SVC(kernel='rbf')
clf = GridSearchCV(svm, param_grid, scoring='f1') #['accuracy', 'precision', 'recall', 'f1']
clf.fit(X_train_scaled, y_train)
print(pd.DataFrame(clf.cv_results_))
print("clf.best_params_\n", clf.best_params_)
print("clf.best_score_\n", clf.best_score_)
best_svm = clf.best_estimator_

# CV models plot
results = clf.cv_results_
C_values = results['param_C'].data
gamma_values = results['param_gamma'].data
metric_scores = results['mean_test_score']

plt.figure(figsize=(8, 6))
for gamma_value in sorted(set(gamma_values)):
    gamma_indices = [i for i, val in enumerate(gamma_values) if val == gamma_value]
    plt.plot(C_values[gamma_indices], metric_scores[gamma_indices], label=f"gamma={gamma_value}", linewidth=2)

plt.xlabel("C parameter")
plt.ylabel("F1 Score")  # Adjust label based on metric
plt.title("Hyperparameter Tuning Results")
plt.legend()
plt.grid(True)
plt.show()

###



#enet_mod = ElasticNet(random_state=42)
#rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
#grid_search = GridSearchCV(enet_mod, param_grid, cv=rkf, scoring='f1')

enet_log_mod = LogisticRegression(max_iter=1000) #penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5
rkf = RepeatedKFold(n_splits=4, n_repeats=10, random_state=42)
scores = cross_val_score(enet_log_mod, X, y, scoring='f1', cv=rkf) #, n_jobs=-1
print('scores\n', scores, '\n')
print('Mean f1: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
scores = cross_val_score(enet_log_mod, X, y, scoring='recall', cv=rkf)
print('scores\n', scores, '\n')
print('Mean recall: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
scores = cross_val_score(enet_log_mod, X, y, scoring='precision', cv=rkf)
print('scores\n', scores, '\n')
print('Mean precision: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


