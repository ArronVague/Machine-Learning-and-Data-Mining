from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. 准备数据
# 假设你已经准备好了特征矩阵 X 和标签向量 y
df = pd.read_csv("illness.csv")
df["class"] = df["class"].map({"Abnormal": 0, "Normal": 1}).fillna(-1)

# print(df.isnull().sum())

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 创建模型实例
model = LogisticRegression()

# 5. 拟合模型
model.fit(X_train, y_train)

weights = model.coef_
print("Omega (weights):")
print(weights)

# 6. 进行预测
y_pred = model.predict(X_test)

# 7. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
