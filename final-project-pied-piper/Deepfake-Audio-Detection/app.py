import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import plotly.graph_objects as go
from transformers import Wav2Vec2Model

# ==========================================
# 1. GLOBAL CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Audio Forensic Lab", layout="wide", page_icon="🎙️")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4253;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: #888;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #3e4253;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. OPTIMIZED MODEL ARCHITECTURE
# ==========================================
class Wav2Vec2Classifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        outputs = self.wav2vec(x)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.classifier(pooled).squeeze(1)

@st.cache_resource
def load_model():
    model = Wav2Vec2Classifier("facebook/wav2vec2-base")
    # map_location='cpu' is critical for Mac compatibility
    model.load_state_dict(torch.load('best_model_2604.pth', map_location='cpu'))
    model.eval()
    return model

# Pre-load model to avoid delays during button click
model = load_model()

# ==========================================
# 3. MULTI-TAB INTERFACE
# ==========================================
st.title("🛡️ Deepfake Audio Detection System")
tab_lab, tab_metrics, tab_theory = st.tabs(["🔍 Detection Lab", "📊 Performance Metrics", "📖 Methodology"])

# --- TAB 1: DETECTION LAB ---
with tab_lab:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Upload & Preview")
        uploaded_file = st.file_uploader("Upload WAV File (16kHz recommended)", type=["wav"])
        
        if uploaded_file:
            st.audio(uploaded_file)
            y, sr = librosa.load(uploaded_file, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Duration", f"{duration:.2f}s")
            m_col2.metric("Sample Rate", f"{sr}Hz")

    with col2:
        st.subheader("📊 Analysis Results")
        if uploaded_file and st.button("🚀 Run Deepfake Scan", use_container_width=True):
            with st.spinner("Scanning for synthetic artifacts..."):
                # 1. Preprocessing
                y_norm = (y - np.mean(y)) / (np.std(y) + 1e-6)
                target_len = 160000 # 10s windows
                
                # 2. Faster Windowing (Limit windows if file is very long)
                if len(y_norm) <= target_len:
                    windows = [np.pad(y_norm, (0, target_len - len(y_norm)))]
                else:
                    # To prevent hanging, we take a max of 5 overlapping segments
                    step = max(target_len // 2, len(y_norm) // 5)
                    windows = [y_norm[i : i + target_len] for i in range(0, len(y_norm) - target_len + 1, step)]
                
                # 3. Batch Inference (Process all windows at once if possible)
                input_tensor = torch.tensor(np.array(windows), dtype=torch.float32)
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.sigmoid(logits).numpy()

                final_prob = np.max(probs)
                is_fake = final_prob > 0.3

                # 4. Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = final_prob * 100,
                    title = {'text': "Spoof Confidence (%)", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ff4b4b" if is_fake else "#00cc96"},
                        'steps': [{'range': [0, 30], 'color': "#1a2a1a"}, {'range': [30, 100], 'color': "#2a1a1a"}],
                        'threshold': {'line': {'color': "white", 'width': 4}, 'value': 30}
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)

                if is_fake:
                    st.error("### 🚨 VERDICT: DEEPFAKE DETECTED")
                else:
                    st.success("### ✅ VERDICT: AUTHENTIC AUDIO")
        else:
            st.info("Awaiting audio input for forensic scanning.")

# --- TAB 2: PERFORMANCE METRICS ---
with tab_metrics:
    st.header("📈 Model Evaluation Results")
    st.write("Results based on test evaluation against EnvSDD and LENS-DF datasets.")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", "87.00%")
    m2.metric("Precision", "93.70%")
    m3.metric("Recall", "79.33%")
    m4.metric("F1 Score", "0.8592")
    m5.metric("Loss", "0.4130")
    
    st.divider()
    st.subheader("Metric Interpretation")
    st.markdown("""
    - **High Precision (93.7%):** The model is highly reliable when flagging an audio as 'Fake', minimizing false accusations.
    - **Balanced F1:** Demonstrates robust performance despite the complex nature of synthetic environmental sounds.
    """)

# --- TAB 3: METHODOLOGY ---
with tab_theory:
    st.header("📖 Theoretical Background")
    st.markdown("""
    ### Wav2Vec2: Large-Scale Self-Supervised Learning
    This system utilizes the **Wav2Vec2** architecture, a transformer-based model that learns powerful representations from raw audio waveforms.
    
    #### How it detects Deepfakes:
    1. **Feature Extraction:** A multi-layer convolutional network extracts latent features from the raw signal.
    2. **Contextual Representations:** Transformer layers analyze the temporal dependencies, identifying stutters or phase shifts common in AI-generated speech (vocoder artifacts).
    3. **Background Analysis:** Unlike standard voice-only detectors, this model is fine-tuned to identify synthetic manipulations in **environmental and background audio**.
    """)
    st.info("System optimized for 16,000Hz mono-channel WAV input.")

# ==========================================
# 4. FOOTER
# ==========================================
st.markdown(f"""
    <div class="footer">
        <p>Developed by <b>Abhishek • Ritik • Jaden</b> | University at Buffalo</p>
    </div>
    """, unsafe_allow_html=True)