import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.stats import norm
from scipy import stats

dataset1 = np.genfromtxt('final.txt',delimiter=',',skip_header=1,dtype=None,encoding=None)
dataset1 = np.array(dataset1.tolist())
# dataset1 = dataset1[:,2:]
# dataset1 = np.where(np.isnan(dataset1), ma.array(dataset1, mask=np.isnan(dataset1)).mean(axis=0), dataset1)
# train = dataset1[0:2304:]
# train_label = train[:,2]
# train_feature = train[:,3:]
# split data into train and test sets
X = dataset1[:,3:].astype(np.float)
y = dataset1[:,2].astype(np.float)
#dataset = dataset1.astype(np.float)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)
#X = np.array(X)
#y = np.array(y)
label = ['STATE','COUNTY','COVID_19_CASES','PHYSICAL_UNHEALTHY_DAYS','MENTALLY_UNHEALTHY_DAYS','LOWBIRTHWEIGHT','SMOKERS','ADULTSWITHOBESITY','PHYSICALLYINACTIVE','EXERCISEOPPORTUNITIES','EXCESSIVEDRINKING','CHLAMYDIA','UNINSURED','PM_PHYSICIANS','PH_RATE','VACCINATED','HIGHSCHOOLGR','POPULATION','CHILDRENINPOVERTY','ASSOCIATIONS','PM25','OVERCROWDING','DRIVEALONE']
label = label[3:]
def re(x):
    return label[int(x)]
X = X.rename(columns = re)
print(X.sample(3))
def featureselection(X, y,label):
    Xreturn, yreturn = X,y
    Xreturn = MinMaxScaler().fit_transform(Xreturn)
    clf = RandomForestClassifier(n_estimators = 500)
    clf.fit(Xreturn, yreturn)
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1][:15]
    Xreturn = Xreturn[:,indices]
    for i in indices:
        print(label[i])
    return Xreturn, yreturn
def plotfrequency(df,feature):
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    # creating a grid of 3 cols and 3 rows.
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # Customizing the histogram grid.
    ax1 = fig.add_subplot(grid[0, :2])
    # Set the title.
    ax1.set_title('Histogram')
    # plot the histogram.
    sns.distplot(df.loc[:, feature],
                 hist=True,
                 kde=True,
                 fit=norm,
                 ax=ax1,
                 color='#e74c3c')
    ax1.legend(labels=['Normal', 'Actual'])

    # customizing the QQ_plot.
    ax2 = fig.add_subplot(grid[1, :2])
    # Set the title.
    ax2.set_title('Probability Plot')
    # Plotting the QQ_Plot.
    stats.probplot(df.loc[:, feature].fillna(np.mean(df.loc[:, feature])),
                   plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor('#e74c3c')
    ax2.get_lines()[0].set_markersize(12.0)

    # Customizing the Box Plot.
    ax3 = fig.add_subplot(grid[:, 2])
    # Set title.
    ax3.set_title('Box Plot')
    # Plotting the box plot.
    sns.boxplot(df.loc[:, feature], orient='v', ax=ax3, color='#e74c3c')
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=24))

    plt.suptitle(f'{feature}', fontsize=15)
    plt.savefig('./feature/'+'%s.jpg'%label[feature])
    plt.show()
def discrete(label):
    tem = label[0]
    newlabel = pd.qcut(tem,9,labels=False)
    return newlabel
newlabel = discrete(y)
def ChiMerge_MaxInterval_Original(df, col, target,max_interval):
    colLevels=set(df[col])
    colLevels=sorted(list(colLevels))
    N_distinct=len(colLevels)
    if N_distinct <= max_interval:
        print("the row is cann't be less than interval numbers")
        return colLevels[:-1]
    else:
        total=df.groupby([col])[target].count()
        total=pd.DataFrame({'total':total})
        bad=df.groupby([col])[target].sum()
        bad=pd.DataFrame({'bad':bad})
        regroup=total.merge(bad, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)
        N=sum(regroup['total'])
        B=sum(regroup['bad'])
        overallRate=B*1.0/N
        groupIntervals=[[i] for i in colLevels]
        groupNum=len(groupIntervals)
        while(len(groupIntervals)>max_interval):
            chisqList=[]
            for interval in groupIntervals:
                df2=regroup.loc[regroup[col].isin(interval)]
                chisq=Chi2(df2,'total','bad',overallRate)
                chisqList.append(chisq)
            min_position=chisqList.index(min(chisqList))
            if min_position==0:
                combinedPosition=1
            elif min_position==groupNum-1:
                combinedPosition=min_position-1
            else:
                if chisqList[min_position-1]<=chisqList[min_position + 1]:
                    combinedPosition=min_position-1
                else:
                    combinedPosition=min_position+1
            #合并箱体
            groupIntervals[min_position]=groupIntervals[min_position]+groupIntervals[combinedPosition]
            groupIntervals.remove(groupIntervals[combinedPosition])
            groupNum=len(groupIntervals)
        groupIntervals=[sorted(i) for i in groupIntervals]
        print(groupIntervals)
        cutOffPoints=[i[-1] for i in groupIntervals[:-1]]
        return cutOffPoints
interval = ChiMerge_MaxInterval_Original(X,'PHYSICAL_UNHEALTHY_DAYS', newlabel, 300)
print(interval)

def detectmissing(data):
    fig, ax = plt.subplots(ncols=1, figsize=(20, 6))
    sns.heatmap(data.isnull(),
                yticklabels=False,
                cbar=False,
                cmap='magma')
    ax.set_title('Train Data Missing Values')
    plt.xticks(rotation=90)
    plt.show()
#detectmissing(X)
#for i in label:
#    plotfrequency(X,i)
#print(featureselection(X,y,label))
def vaild(X,y):
    final = []
    kf = KFold(n_splits = 10, shuffle= False)
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        clf = RandomForestClassifier(n_estimators = 500)
        clf.fit(train_X, train_y)
        res = clf.predict(test_X)
        final.append(r2_score(test_y, res))
    return sum(final)/10
#print (vaild(X,y),vaild(Xselect,yselect))
