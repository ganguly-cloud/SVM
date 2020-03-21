
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
df = pd.read_csv('Social_Network_Ads.csv')
print df.head()
'''
    User ID  Gender  Age  EstimatedSalary  Purchased
0  15624510    Male   19            19000          0
1  15810944    Male   35            20000          0
2  15668575  Female   26            43000          0
3  15603246  Female   27            57000          0
4  15804002    Male   19            76000          0'''

# Devide x and y i.e dependent and independent variables

x = df.iloc[:,[2,3]].values
print x[:4]
'''
[[   19 19000]
 [   35 20000]
 [   26 43000]
 [   27 57000]]'''

y = df.iloc[:,4].values
print y[:5]   # [0 0 0 0 0]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =.25,random_state=0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print x_train[:6]
'''
[[ 0.58164944 -0.88670699]
 [-0.60673761  1.46173768]
 [-0.01254409 -0.5677824 ]
 [-0.60673761  1.89663484]
 [ 1.37390747 -1.40858358]
 [ 1.47293972  0.99784738]]'''
print x_test[:5]
'''
[[-0.80480212  0.50496393]
 [-0.01254409 -0.5677824 ]
 [-0.30964085  0.1570462 ]
 [-0.80480212  0.27301877]
 [-0.30964085 -0.5677824 ]]'''

classifier = SVC(kernel ='linear',random_state=0)
# we can try kernel in with 'rbf' and 'poly'

print classifier.fit(x_train,y_train)
'''
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=0, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)'''

# Predict the test set results
y_pred = classifier.predict(x_test)
print y_pred
'''
[0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0
 0 0 1 0 0 0 0 1 0 0 1 0 1 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0
 0 0 1 0 1 1 1 1 0 0 1 1 0 1 0 0 0 1 0 0 0 0 0 0 1 1]
[0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0
 0 0 1 0 0 0 0 1 0 0 1 0 1 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0
 0 0 1 0 1 1 1 1 0 0 1 1 0 1 0 0 0 1 0 0 0 0 0 0 1 1]'''
# Confusion Matrix

from sklearn.metrics import confusion_matrix,accuracy_score

Accu = accuracy_score(y_test,y_pred)
print Accu   # 0.89

cm = confusion_matrix(y_test,y_pred)
print cm

'''
[[65  3]
 [ 8 24]]'''

from matplotlib.colors import ListedColormap

X_set,y_set =x_test,y_test
x1,x2= np.meshgrid(np.arange(start =X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                   np.arange(start =X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated slary')
plt.savefig('Clear_map_shows')
plt.show()
                

