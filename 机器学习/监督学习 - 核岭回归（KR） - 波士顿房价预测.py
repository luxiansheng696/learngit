# 1. 导入所需库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体（避免图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 2. 加载波士顿房价数据集
def load_boston_housing():
    # 核心：读取housing.data.txt并解析成特征(X)和目标值(y)
    # 路径说明：r'.\波士顿房价数据.txt' 表示当前目录下的housing.data.txt文件
    with open(r'波士顿房价数据.txt', 'r') as f:
        # 存储所有数据的列表
        data_list = []
        # 逐行读取（跳过空行/注释行）
        for line in f:
            # 去除首尾空格，按任意空格分割（适配数据中的多空格分隔）
            line_data = line.strip().split()
            # 跳过空行
            if not line_data:
                continue
            # 转换为浮点数并加入列表
            data_list.append([float(x) for x in line_data])

    # 将列表转换为numpy数组（方便后续机器学习使用）
    data = np.array(data_list)
    # 拆分特征(X)和目标值(y)（最后一列是房价，前面的是特征）
    X = data[:, :-1]  # 所有行，除最后一列外的所有列（特征）
    y = data[:, -1]  # 所有行，最后一列（房价）
    feature_names = [
        "人均犯罪率", "住宅用地占比", "非零售商业用地比例", "是否邻近查尔斯河",
"一氧化氮浓度", "住宅平均房间数", "老旧住宅比例", "就业中心加权距离",
"高速公路便利指数", "财产税税率", "师生比例", "黑人比例与低收入组合变量",
"低收入人群比例"
    ]
    return X, y, feature_names

# 3. 数据加载与预处理
X, y, feature_names = load_boston_housing()
print("波士顿房价数据加载成功")
print(f"特征矩阵形状：{X.shape}，房价目标值形状：{y.shape}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 数据标准化
'''把数据特征转换到 “均值为 0、标准差为 1” 的统一尺度上，
核心目的是消除不同特征之间的量纲差异，让模型能公平对待每个特征，提升训练效果。'''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 模型训练
'''RBF（径向基函数）是衡量样本间相似度的函数，核心是 “距离越近，相似度越高”'''
model = KernelRidge(kernel='rbf', gamma=0.1, alpha=0.5)
model.fit(X_train_scaled, y_train)

# 5. 模型预测与评估
y_pred = model.predict(X_test_scaled)             #房价预测
mse = mean_squared_error(y_test, y_pred)          #方差
rmse = np.sqrt(mse)                               #标准差
r2 = r2_score(y_test, y_pred)                     #决定系数，越接近1越好

# 打印评估结果（无特殊符号）
print("\n模型评估结果")
print("-" * 30)
print(f"测试集均方误差（MSE）：{mse:.2f}")
print(f"测试集均方根误差（RMSE）：{rmse:.2f} 千美元")
print(f"测试集决定系数（R²）：{r2:.2f}")

# 6. 可视化图表（核心部分）
# 创建画布，设置子图布局（1行3列）
'''fig是整张画布，axes包含三个画框'''
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 子图1：波士顿房价分布直方图
axes[0].hist(y, bins=30, edgecolor='black', alpha=0.7)  #直方图，分成30区间,透明度0.7
axes[0].set_title('波士顿房价分布', fontsize=12)
axes[0].set_xlabel('房价（千美元）')
axes[0].set_ylabel('样本数量')
axes[0].grid(alpha=0.3)                                 #网格线

# 子图2：真实房价 vs 预测房价散点图
axes[1].scatter(y_test, y_pred, alpha=0.7, edgecolor='black', s=50)   #s=50，散点大小
# 添加y=x参考线（完美预测线）
'''真实房价线y=x，线之上高于真实值，线之下低于真实值'''
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  #r--红色虚线
axes[1].set_title('真实房价 vs 预测房价', fontsize=12)
axes[1].set_xlabel('真实房价（千美元）')
axes[1].set_ylabel('预测房价（千美元）')
axes[1].grid(alpha=0.3)

# 子图3：特征与房价的相关性（Top5）
# 计算特征与房价的相关系数
df = pd.DataFrame(X, columns=feature_names)
df['房价'] = y
corr = df.corr()['房价'].sort_values(ascending=False)      #线性相关系数计算
top5_corr = corr[1:6]  # 排除自身，取Top5正相关/负相关特征
print('特征与房价的相关性（Top5）:',top5_corr.index)

axes[2].barh(top5_corr.index, top5_corr.values, color='skyblue', edgecolor='black')   #水平柱状图
axes[2].set_title('特征与房价的相关性（Top5）', fontsize=12)
axes[2].set_xlabel('相关系数')
axes[2].grid(alpha=0.3)

# 调整子图间距，显示图表
plt.tight_layout()
plt.show()

# 打印预测示例
print("\n预测示例")
print("-" * 30)
print(f"前5个真实房价：{y_test[:5].round(2)}")
print(f"前5个预测房价：{y_pred[:5].round(2)}")