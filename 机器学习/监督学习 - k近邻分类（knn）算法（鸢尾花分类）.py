import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

#return_X_y=True为了返回数据是元组而不是字典,
# 作用是让函数load_iris直接返回 (特征矩阵X, 标签数组y) 这个元组
X,y=load_iris(return_X_y=True)

#将数据集进行划分，训练集占7成，测试集占3成
'''stratify=y	按标签分层划分	关键作用：保证训练集和测试集中，
各类别的样本比例和原数据一致（比如原数据 3 类鸢尾花各 50 个，
划分后训练集每类 35 个，测试集每类 15 个），避免因随机划分导致某类样本在测试集占比失衡'''

'''random_state=42	固定随机种子	作用：
让每次运行代码的划分结果完全相同（方便复现结果），如果不设置，每次运行划分结果都不一样'''
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.3,train_size=0.7,stratify=y,random_state=42)
scores=[]

#使用不同的邻居数进行训练测试
for n in range(1,6):
    knn=KNeighborsClassifier(n_neighbors=n)
    #训练
    knn.fit(train_X,train_y)
    #预测
    pred=knn.predict(test_X)
    #准确率并保留3位小数
    '''knn.score(test_X, test_y) —— 计算模型的评估得分
    round() 是 Python 内置函数，用于对数字进行四舍五入，
    语法是：round(数字, 保留小数位数)'''
    score=round(knn.score(test_X,test_y),3)
    scores.append(score)

print('分类数由1到5的模型依次得分：',scores)
plt.figure()
plt.plot(range(1,6),scores,'o--',color='blue')
plt.xlabel('$n\_neighbors$',fontsize=14)
plt.ylabel('$precision$',fontsize=14)
for x,y in zip(range(1,6),scores):
    plt.text(x-0.18,y-0.1,f'${y}$',fontsize=14)
plt.title(f'$precision\ of\ different\ neighbors$',fontsize=14)
plt.xticks(np.arange(1,6))
plt.yticks(np.linspace(0,1,5))
plt.show()

