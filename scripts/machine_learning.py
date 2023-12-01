import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
import matplotlib


"""
author: @dohunkang
A python code to analyze the relationship between host-guest atomic property and THH preference
using random forest classifier, shap analysis, and gaussian process classifier
"""


df = pd.read_csv("../data/magpie_feature_w_preference.csv")
df.head()
arr = df.to_numpy()
X = df.drop(['preference'], axis=1)
X = X[['dev_MendeleevNumber', 'min_Column', 'mean_Row', 'mean_CovalentRadius',
       'dev_CovalentRadius', 'mean_Electronegativity', 'dev_NfValence', 'mean_NsUnfilled',
       'dev_NUnfilled', 'dev_GSvolume_pa', 'frac_dValence', 'CanFormIonic']]
Y = df['preference'] > 0.6
corr = X.corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr)
plt.show()

x = X.to_numpy()
arr = np.zeros((12, 12))
for ii in range(12):
    for jj in range(12):
        aa = x[:,ii]
        bb = x[:,jj]
        arr[ii, jj] = np.corrcoef(aa, bb)[0][1]       

x = X.to_numpy()
y = Y.to_numpy()
rf = RandomForestClassifier()
scores = cross_val_score(rf, x, y, cv=5)
print(f"random forest classfication CV: {np.average(scores)}")

feature_names = X.columns
rf.fit(x,y)
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)


sorted_indices = np.argsort(importances)[::-1]
feature_names = feature_names[sorted_indices]
importances = importances[sorted_indices]
std = std[sorted_indices]

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots(figsize=(8, 6))
forest_importances.plot.bar(ax=ax, fontsize=8)
ax.set_title("Feature importances using MDI", fontsize=13)
ax.set_ylabel("Mean decrease in impurity", fontsize=13)
plt.xticks(rotation=45)
plt.yticks(fontsize=13)
fig.tight_layout()
print(forest_importances)


import shap

rf.fit(X,Y)
print(np.average(Y))
explainer = shap.TreeExplainer(rf, X, feature_names=X.columns)
shap_values = explainer.shap_values(X, Y, check_additivity=False)
shap.summary_plot(shap_values, X, feature_names=X.columns, class_names=["favor", "unfavor"], show=False)
fig, ax = plt.gcf(), plt.gca()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(12)
plt.show()
shap.summary_plot(shap_values[1], X, feature_names=X.columns, class_names=["favor", "unfavor"])

X = df[['dev_MendeleevNumber','dev_GSvolume_pa']].to_numpy()
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits)
kernel = 1.0 * RBF([1.0] * X.shape[-1], length_scale_bounds=(1e-5, 1e5))
scores = np.zeros(n_splits)
for ii, (train_index, test_index) in enumerate(skf.split(X, Y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X_train, y_train)
    scores[ii] = gpc.score(X_test, y_test)
    
print(f"5-fold CV score for gpc: {np.average(scores):.5f}")

h = 0.1  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

gpc.fit(X,Y)
Z = gpc.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
Z = Z.reshape((xx.shape[0], xx.shape[1]))

fig, ax = plt.subplots(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=matplotlib.cm.bwr.reversed(), alpha=0.3, levels=np.linspace(0, 1, 11))
cbar = plt.colorbar(format='%.2f')
cbar.ax.tick_params(labelsize=17)

decision_boundary = 0.5
levels = [decision_boundary]
cs = plt.contour(xx, yy, Z, levels=levels, colors='black', linestyles='dashed', linewidths=2)
plt.clabel(cs, fontsize=20)


scatter = ax.scatter(X[:,0], X[:, 1], c=Y, cmap=matplotlib.cm.bwr.reversed(), edgecolor='k')
handles, labels = scatter.legend_elements()
legend1 = ax.legend(handles=handles, labels=["unfavored", "favored"], fontsize=20, borderpad=0.1, handletextpad=0.1)

plt.title("Predicted preference probability", fontsize=23)
plt.tight_layout()
plt.xlabel("dev (Mendeleev Number)", fontsize=23)
plt.ylabel("dev (DFT calculated size)", fontsize=23)
plt.yticks(fontsize=23)
plt.xticks(fontsize=23)
plt.show()
