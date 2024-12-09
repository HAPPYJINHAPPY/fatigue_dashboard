import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

# Load the uploaded file
file_path = 'C:/Users\X2006936/fatigue_dashboard/corrected_fatigue_simulation_data.csv'
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

# 训练模型后保存路径
MODEL_PATH = 'fatigue_model.pkl'

# 检查模型文件是否存在并加载
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Ensure the model is trained and saved.")

app = Flask(__name__)
CORS(app)  # 允许跨域访问

@app.route('/')
def home():
    return "Welcome to the Fatigue Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # 检查请求内容是否为 JSON
    if not request.is_json:
        return jsonify({'error': 'Request content must be JSON'}), 400

    input_data = request.json  # 假设为 JSON 格式
    if not input_data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # 将输入数据转化为 DataFrame
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)
        return jsonify({'prediction': int(prediction[0])}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
