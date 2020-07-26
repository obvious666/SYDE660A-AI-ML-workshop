import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import Normalizer
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from collections import OrderedDict
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

dataset1 = np.genfromtxt('countyhealth.csv',delimiter=',',skip_header=1,usecols = np.arange(2,37),missing_values='NA')
dataset1 = np.where(np.isnan(dataset1), np.ma.array(dataset1, mask=np.isnan(dataset1)).mean(axis=0), dataset1)  #replace nan value with mean


# split data into train and test sets
seed = 7
test_size = 0.33

X = dataset1[:,1:]
y = dataset1[:,0]

#normalization
# transformer = Normalizer().fit(X)  # fit does nothing.
# X=transformer.transform(X)

# lle = LocallyLinearEmbedding(n_components=4, n_neighbors=5)
# X = lle.fit_transform(X)

# isomap = Isomap(n_components=4, n_neighbors=5)
# X = isomap.fit_transform(X)

# lda = LDA(n_components=4)
# X = lda.fit_transform(X, y)

# pca = PCA(n_components=4)
# X = pca.fit_transform(X)

# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# X = sel.fit_transform(X)

# X = SelectKBest(chi2, k=2).fit_transform(X, y)

# find the best parameter
# def isbest_parameter(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
#
#     r_score = {}
#     colsample_bytree=[0.4,0.6,0.8]
#     gamma = [0,0.03,0.1,0.3]
#     min_child_weight = [1.5,6,10]
#     learning_rate = [0.1,0.07]
#     max_depth = [3,5]
#     n_estimators = [50]
#     reg_alpha = [1e-5, 1e-2,  0.75]
#     reg_lambda = [1e-5, 1e-2, 0.45]
#     subsample = [0.6,0.95]
#     for a in colsample_bytree:
#         for b in gamma:
#             for c in min_child_weight:
#                 for d in learning_rate:
#                     for e in max_depth:
#                         for f in n_estimators:
#                             for g in reg_alpha:
#                                 for h in reg_lambda:
#                                     for i in subsample:
#                                         bst = xgb.XGBRegressor(colsample_bytree=a,gamma=b,min_child_weight=c,learning_rate=d,max_depth=e,n_estimators=f,reg_alpha=g,reg_lambda=h,subsample=i)
#                                         bst.fit(X_train, y_train)
#                                         preds = bst.predict(X_test)
#                                         score = r2_score(y_test, preds)
#                                         # mse = mean_squared_error(y_test, preds)
#                                         r_score[a,b,c,d,e,f,g,h,i]=score
#     return max(r_score, key=r_score.get), max(r_score.values())  # get the key with largest value, and the largest value
#
#
# result = isbest_parameter(X,y)
# print (result)

# process of xgboost
# def boost(X,y):
#     X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=seed)
#
#     bst = xgb.XGBRegressor()
#     bst.fit(X_train,y_train)
#     preds = bst.predict(X_test)
#     score = r2_score(y_test,preds)
#     mse = mean_squared_error(y_test,preds)
#     return [score,mse],bst

# result = boost(X,y)
# print (result[0])

# importance = result[1].feature_importances_   # bst.feature_importance
# feature_name = np.genfromtxt('countyhealth.csv',delimiter=',',dtype=None,encoding=None)[0][3:37]
#
#
# feature_importance = np.array([feature_name,importance])
# feature_importance = np.stack(feature_importance,axis=1)
# print (feature_importance)  # features with importance

# index = []
# for i in range(len(importance)):
#     if importance[i] >0.01:   #only use the features with importance greater than 0.01
#         index.append(i)
#
# new_X = X[:,index]
#
# print (boost(new_X,y)[0])


#plot important features
# plt.scatter(dataset1[:,12],dataset1[:,0],c="blue",marker = "s")
# plt.title("COVID19_Death VS Chlamydia")
# plt.xlabel("Chlamydia")
# plt.ylabel("COVID19_Death")
# plt.show()

# plt.scatter(dataset1[:,15],dataset1[:,0],c="blue",marker = "s")
# plt.title("COVID19_Death VS Numbers of Primary Care phsicians")
# plt.xlabel("Numbers of Primary Care phsicians")
# plt.ylabel("COVID19_Death")
# plt.show()

# plt.scatter(dataset1[:,28],dataset1[:,0],c="blue",marker = "s")
# plt.title("COVID19_Death VS Single Parent Household")
# plt.xlabel("Single Parent Household")
# plt.ylabel("COVID19_Death")
# plt.show()

# plt.scatter(dataset1[:,30],dataset1[:,0],c="blue",marker = "s")
# plt.title("COVID19_Death VS Number of Associations")
# plt.xlabel("Number of Associations")
# plt.ylabel("COVID19_Death")
# plt.show()
