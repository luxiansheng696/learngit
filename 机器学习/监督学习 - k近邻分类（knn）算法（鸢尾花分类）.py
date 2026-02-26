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
#创建一个新的绘图窗口,可以指定画布大小（如 plt.figure(figsize=(8,6))）
plt.figure()
#绘制折线图
'''
range(1,6)：x 轴数据，对应 n_neighbors 的取值（1、2、3、4、5）；
scores：y 轴数据，对应每个 K 值的精准率（precision）；
'o--'：线条样式 ——o 表示在每个数据点画圆形标记，-- 表示线条是虚线；
color='blue'：设置线条和标记点的颜色为蓝色'''
plt.plot(range(1,6),scores,'o--',color='blue')
#设置 x 轴的标签（名称）,fontsize=14：设置标签字体大小为 14 号,$...$美化文本
plt.xlabel('$n\_neighbors$',fontsize=14)
#设置 y 轴的标签（名称）
plt.ylabel('$precision$',fontsize=14)
#在每个数据点旁标注具体的精准率数值（避免看图猜值）
'''
zip(range(1,6),scores)：把 x 轴值（1-5）和对应的 y 轴值（scores）一一配对；
x-0.18, y-0.1：标注文本的坐标（相对于数据点 (x,y) 向左偏移 0.18、向下偏移 0.1，避免遮挡数据点）；
f'${y}$'：要显示的文本内容（当前精准率值），$...$ 让数值显示更美观；'''
for x,y in zip(range(1,6),scores):
    plt.text(x-0.18,y-0.1,f'${y}$',fontsize=14)
#设置图表的标题
plt.title(f'$precision\ of\ different\ neighbors$',fontsize=14)
#自定义 x 轴的刻度值。
plt.xticks(np.arange(1,6))
#自定义 y 轴的刻度值
plt.yticks(np.linspace(0,1,5))
#显示绘制好的图表
plt.show()

