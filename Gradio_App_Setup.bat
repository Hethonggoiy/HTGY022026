@echo off
title Gradio Builder - Master 2026
cls
color 0D

echo --- 1. Tao file app_gradio.py ---
(
echo import gradio as gr
echo import joblib
echo import numpy as np
echo import os
echo.
echo # Duong dan mo hinh
echo BASE_DIR = os.path.dirname(os.path.abspath(__file__^)^)
echo MODEL_PATH = os.path.join(BASE_DIR, "models", "mlp_model.pkl"^)
echo SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl"^)
echo ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoder.pkl"^)
echo.
echo def predict_block(toan, van, ly, hoa, sinh, anh^):
echo     try:
echo         # 1. Chuan bi du lieu
echo         scores = [float(toan^), float(van^), float(ly^), float(hoa^), float(sinh^), float(anh^)]
echo         sums = [scores[0^]+scores[2^]+scores[5^], scores[0^]+scores[3^]+scores[4^], 
echo                 scores[1^]+scores[0^]+scores[2^], scores[0^]+scores[1^]+scores[5^]]
echo         X_input = np.array([scores + sums^]^)
echo.
echo         # 2. Load model
echo         mlp = joblib.load(MODEL_PATH^)
echo         scaler = joblib.load(SCALER_PATH^)
echo         le = joblib.load(ENCODER_PATH^)
echo.
echo         # 3. Du bao
echo         X_scaled = scaler.transform(X_input^)
echo         proba = mlp.predict_proba(X_scaled^)[0^]
echo         res = le.classes_[np.argmax(proba^)]
echo.
echo         # Tao dict ket qua cho bieu do
echo         confidences = {le.classes_[i^]: float(proba[i^]^) for i in range(len(le.classes_^)^)}
echo         return res, confidences
echo     except Exception as e:
echo         return str(e^), None
echo.
echo # --- GIAO DIEN GRADIO ---
echo demo = gr.Interface(
echo     fn=predict_block,
echo     inputs=[
echo         gr.Number(label="Toan"^), gr.Number(label="Van"^), gr.Number(label="Ly"^),
echo         gr.Number(label="Hoa"^), gr.Number(label="Sinh"^), gr.Number(label="Anh"^)
echo     ],
echo     outputs=[
echo         gr.Textbox(label="Khoi thi goi y"^),
echo         gr.Label(label="Xac suat chi tiet"^)
echo     ],
echo     title="HE THONG GOI Y CHON KHOI THI - MLP 8 LAYERS",
echo     description="Nhap diem cua ban de mo hinh MLP du bao khoi thi phu hop nhat."
echo ^)
echo.
echo if __name__ == "__main__":
echo     demo.launch(server_name="0.0.0.0", server_port=10000^)
) > app_gradio.py

echo --- 2. Cap nhat requirements.txt ---
echo gradio >> requirements.txt
echo joblib >> requirements.txt
echo numpy >> requirements.txt
echo scikit-learn >> requirements.txt

echo --- 3. Day len GitHub ---
git add .
git commit -m "Switch to Gradio interface"
git push origin main --force

echo [OK] Da xong! Hay sang Render doi Deploy.
pause