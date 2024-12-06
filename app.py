import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from flask import Flask, request, jsonify
import joblib

# Load the uploaded file
file_path = 'D:/pythonProject3/corrected_fatigue_simulation_data.csv'
data = pd.read_csv(file_path)

# 1. 特征和标签
X = data.drop(columns=["Fatigue_Label"])
y = data["Fatigue_Label"]

# 2. 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 模型训练
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. 预测
y_pred = model.predict(X_test)

# 5. 评估
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 保存模型
joblib.dump(model, 'fatigue_model.pkl')
app = Flask(__name__)

# 加载模型
model = joblib.load('fatigue_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # 获取前端传来的数据
    input_data = request.json  # 假设为 JSON 格式
    print(123)
    print(input_data)
    df = pd.DataFrame([input_data])  # 转换为 DataFrame
    prediction = model.predict(df)
    print(prediction)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
 
