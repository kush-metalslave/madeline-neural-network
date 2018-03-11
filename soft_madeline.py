import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
csvfl=pd.read_csv('imports-85.data.csv')
#print(csvfl)
y=csvfl[['price']]
y=y.astype(np.float)
csvfl=csvfl[['symboling','wheel-base','length','width','height','curb-weight',
'engine-size','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg']]
for j in csvfl:
    for k in range(len(csvfl)):
        if csvfl[j].iloc[k]=='?':
            csvfl[j].iloc[k]=int(9999)
#print(csvfl)
csvfl[['bore','stroke','horsepower','peak-rpm']]=csvfl[['bore','stroke','horsepower','peak-rpm']].apply(pd.to_numeric)
z=np.min(csvfl,axis=0)
x=np.max(csvfl,axis=0)-np.min(csvfl,axis=0)
csvfl=(csvfl-z)/x
#print(csvfl)
train=csvfl[:-9]
y_train=y[:-9]
y_test=y[-9:]
test=csvfl[-9:]
#print(test)
#print(y_test)
#this is the code for perceptron
start_time1 = time.time()
clf1 = perceptron.Perceptron(n_iter=200, verbose=0, random_state=None, fit_intercept=True, eta0=0.02)
clf1.fit(train,y_train)
s1=time.time()-start_time1
#this is the code for multi layered perceptron
start_time2 = time.time()
clf2 = MLPClassifier(solver='lbfgs', alpha=0.001, max_iter=200,hidden_layer_sizes=(7,), random_state=1)
clf2.fit(train,y_train)
s2=time.time()-start_time2
#this is the code for adaline
start_time3 = time.time()
clf3 = SGDClassifier(alpha=0.001,learning_rate='optimal',epsilon=0.1,n_iter=500)
clf3.fit(train,y_train)
s3=time.time()-start_time3
#this is the code for back prpogation network
start_time4 = time.time()
clf4 = MLPClassifier(solver='lbfgs', alpha=0.001, max_iter=200,hidden_layer_sizes=(12,2), random_state=1)
clf4.fit(train,y_train)
s4=time.time()-start_time4
accuracy1=clf1.score(train,y_train)*100
accuracy2=clf2.score(train,y_train)*100
accuracy3=clf3.score(train,y_train)*100
accuracy4=clf4.score(train,y_train)*100
plt.rcdefaults()
fig, ax = plt.subplots()
fig1,bx=plt.subplots()
# Example data
networks = ('Perceptron', 'MPN','Adaline', 'BPN')
y_pos = np.arange(len(networks))
performance1 = [s1,s2,s3,s4]
performance = [accuracy1,accuracy2,accuracy3,accuracy4]
ax.barh(y_pos, performance, align='center', color='green')
ax.set_yticks(y_pos)
ax.set_yticklabels(networks)
ax.invert_yaxis()  
ax.set_xlabel('Accuracy %')
ax.set_title('Accuracy  percentage of different Classifiers')
bx.barh(y_pos, performance1, align='center', color='green')
bx.set_yticks(y_pos)
bx.set_yticklabels(networks)
bx.invert_yaxis()
bx.set_xlabel('Time taken')
bx.set_title('Time taken by different Classifiers')
#for i in range(len(performance1)):
 #   print("Time taken to run "+str(networks[i])+" is "+str(performance1[i]))
#for i in range(len(performance1)):
 #   print("Accuracy of the network "+str(networks[i])+" is "+str(performance[i]))
plt.show()
print("\n")
print("\n")
print(" weights for Perceptron after 200 epoch: ")
print(clf1.coef_[1])
print("\n")
print("\n")
print(" weights for MultiLayred Perceptron after 200 epoch: ")
print(clf2.coefs_[1])
print("\n")
print("\n")
print(" weights for Adaline after 200 epoch: ")
print(clf3.coef_[1])
print("\n")
print("\n")
print(" weights for Back Propogation Network after 200 epoch: ")
print(clf4.coefs_[1])

