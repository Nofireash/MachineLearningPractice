# CART决策树使用基尼指数来划分特征，基尼值：指从样本集合D中随机抽取两个样本，其类别不一致的概论
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = load_iris().data
y = load_iris().target

# test_size:float or int, default=None 测试集的大小,测试集的大小，如果是小数的话，值在（0,1）之间，表示测试集所占有的比例 如果是整数，表示的是测试集的具体样本数
X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)

'''
# 选择gini准则即为CART算法  参数max_depth（树的最大深度）和max_leaf_nodes（最大叶节点数） 可以进行剪枝
# criterion参数用来决定使用哪种计算方式评估节点的“重要性” 不填默认为基尼不纯度,填写"gini"使用基尼系数,填写“entropy”使用信息增益
如何选取参数：
（1）通常就使用基尼系数
（2）数据维度很大，噪音很大时使用基尼系数
（3）维度低，数据比较清晰的时候，信息熵和基尼系数没区别
（4）当决策树的拟合程度不够的时候，使用信息熵

# splitter 参数
用来控制决策树生成时，“节点”选择方式的参数，有两种取值：
（1）splitter=“best”：分枝时虽然随机，但是还是会优先选择更“重要”的特征进行分枝
（2）splitter=“random”：一种“放弃治疗”感觉的随机（可能产生“过拟合”，因为模型会因为含有更多的不必要信息而更深更大）
'''
destree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, max_leaf_nodes=4)

cart_tree = destree.fit(X_train, Y_train)

y_train_pre = cart_tree.predict(X_train)
y_test_pre = cart_tree.predict(x_test)

print(f'训练集精确度：{accuracy_score(Y_train, y_train_pre)}')
print(f'测试集精确度：{accuracy_score(y_test, y_test_pre)}')

plt.figure(figsize=(8, 6))
plot_tree(cart_tree)

plt.show()
