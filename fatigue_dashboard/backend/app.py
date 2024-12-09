import pandas as pd
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import os

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
    port = int(os.environ.get("PORT", 5000))  # Vercel 会提供端口
    app.run(debug=False, host='0.0.0.0', port=port)
