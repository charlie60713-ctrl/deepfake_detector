import streamlit as st
import librosa
import numpy as np
import tempfile

st.set_page_config(page_title="AI 語音深偽檢測系統", page_icon="🛡️", layout="centered")

st.title("🛡️ AI 語音深偽檢測系統")
st.write("透過聲學物理特徵（高頻偽影、短時過零率變異數、頻譜平坦度）分析音訊真偽。")

uploaded_file = st.file_uploader("請上傳要檢測的音訊檔案", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes)
    
    if st.button("開始進行聲學分析", type="primary"):
        with st.spinner("正在提取頻譜與聲學特徵中..."):
            # 將上傳的檔案存成暫存檔供 librosa 讀取
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            # 載入音訊
            y, sr = librosa.load(tmp_path, sr=None)
            
            # 1. 頻譜平坦度 (Spectral Flatness)
            flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            
            # 2. 過零率變異數 (Zero-Crossing Rate Variance)
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_var = np.var(zcr)
            
            # 3. 高頻偽影能量比 (High-Frequency Artifacts)
            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            high_freq_mask = freqs > 6000
            if np.any(high_freq_mask):
                high_energy = np.sum(S[high_freq_mask, :])
                total_energy = np.sum(S) + 1e-8
                high_freq_ratio = high_energy / total_energy
            else:
                high_freq_ratio = 0.0
            
            # 綜合特徵指標評估
            anomaly_score = (flatness * 40) + (zcr_var * 80) + (high_freq_ratio * 30)
            
            st.subheader("📊 聲學分析特徵指標")
            col1, col2, col3 = st.columns(3)
            col1.metric("頻譜平坦度", f"{flatness:.4f}")
            col2.metric("過零率變異數", f"{zcr_var:.4f}")
            col3.metric("高頻能量偽影比", f"{high_freq_ratio:.4f}")
            
            st.divider()
            
            # 判定結果
            if anomaly_score > 0.35:
                st.error("⚠️ **檢測警告**：此音訊含有高度疑似 AI 合成或深偽（Deepfake）的異常聲學物理特徵！")
            else:
                st.success("✅ **檢測結果**：此音訊特徵符合正常真人語音範圍，未發現明顯深偽偽影。")