import gradio as gr
import joblib
import numpy as np
import os

# Duong dan mo hinh
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mlp_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoder.pkl")

def predict_block(toan, van, ly, hoa, sinh, anh):
    try:
        # 1. Chuan bi du lieu
        scores = [float(toan), float(van), float(ly), float(hoa), float(sinh), float(anh)]
        sums = [scores[0]+scores[2]+scores[5], scores[0]+scores[3]+scores[4], 
                scores[1]+scores[0]+scores[2], scores[0]+scores[1]+scores[5]]
        X_input = np.array([scores + sums])

        # 2. Load model
        mlp = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(ENCODER_PATH)

        # 3. Du bao
        X_scaled = scaler.transform(X_input)
        proba = mlp.predict_proba(X_scaled)[0]
        res = le.classes_[np.argmax(proba)]

        # Tao dict ket qua cho bieu do
        confidences = {le.classes_[i]: float(proba[i]) for i in range(len(le.classes_))}
        return res, confidences
    except Exception as e:
        return str(e), None

# --- GIAO DIEN GRADIO ---
demo = gr.Interface(
    fn=predict_block,
    inputs=[
        gr.Number(label="Toan"), gr.Number(label="Van"), gr.Number(label="Ly"),
        gr.Number(label="Hoa"), gr.Number(label="Sinh"), gr.Number(label="Anh")
    ],
    outputs=[
        gr.Textbox(label="Khoi thi goi y"),
        gr.Label(label="Xac suat chi tiet")
    ],
    title="HE THONG GOI Y CHON KHOI THI - MLP 8 LAYERS",
    description="Nhap diem cua ban de mo hinh MLP du bao khoi thi phu hop nhat."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
