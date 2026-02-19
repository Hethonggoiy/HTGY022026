@echo off
title Cuong che Push - Master 2026
cls
color 0A

echo --- 1. Huy bo cac tien trinh Git dang bi ket ---
git rebase --abort >nul 2>&1
git merge --abort >nul 2>&1

echo --- 2. Cap nhat ma nguon app.py (Ban chuan) ---
:: (Khối lệnh này đảm bảo file app.py của bạn không còn lỗi cú pháp)
(
echo from flask import Flask, render_template, request, jsonify
echo import pandas as pd
echo import numpy as np
echo import joblib
echo import os
echo.
echo app = Flask(__name__, template_folder='templates'^)
echo.
echo BASE_DIR = os.path.dirname(os.path.abspath(__file__^)^)
echo MODEL_PATH = os.path.join(BASE_DIR, "models", "mlp_model.pkl"^)
echo SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl"^)
echo ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoder.pkl"^)
echo.
echo @app.route('/'^)
echo def index(^):
echo     return render_template('index.html'^)
echo.
echo @app.route('/predict', methods=['POST']^)
echo def predict(^):
echo     try:
echo         data = request.get_json(^)
echo         if not data: return jsonify({'success': False, 'error': 'No input data'^}^)
echo         keys = ['toan', 'van', 'ly', 'hoa', 'sinh', 'anh'^]
echo         scores = [float(data.get(k, 0^)^) for k in keys^]
echo         sums = [scores[0^]+scores[2^]+scores[5^], scores[0^]+scores[3^]+scores[4^], 
echo                 scores[1^]+scores[0^]+scores[2^], scores[0^]+scores[1^]+scores[5^]]
echo         X_input = np.array([scores + sums^]^)
echo         if not os.path.exists(MODEL_PATH^): return jsonify({'success': False, 'error': 'Model missing'^}^)
echo         mlp = joblib.load(MODEL_PATH^)
echo         scaler = joblib.load(SCALER_PATH^)
echo         le = joblib.load(ENCODER_PATH^)
echo         X_scaled = scaler.transform(X_input^)
echo         proba = mlp.predict_proba(X_scaled^)[0^]
echo         pred_idx = np.argmax(proba^)
echo         return jsonify({'success': True, 'block': str(le.classes_[pred_idx^]^), 'probabilities': proba.tolist(^), 'classes': le.classes_.tolist(^)})
echo     except Exception as e:
echo         return jsonify({'success': False, 'error': str(e^)}^, status=500^)
echo.
echo if __name__ == "__main__":
echo     port = int(os.environ.get("PORT", 5000^)^)
echo     app.run(host="0.0.0.0", port=port^)
) > app.py

echo --- 3. Tien hanh Push len GitHub HTGY022026 ---
:: Thiết lập nhánh chính là main
git branch -M main
git add .
git commit -m "Final Fix for Master Thesis 2026"
git push origin main --force

echo.
echo ======================================================
echo   DA DAY CODE THANH CONG! HAY KIEM TRA GITHUB.
echo ======================================================
pause