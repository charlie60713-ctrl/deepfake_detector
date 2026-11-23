import os
import tempfile
import librosa
import numpy as np
import streamlit as st

st.set_page_config(page_title="AI 語音深偽檢測系統", page_icon="🛡️", layout="centered")

st.title("🛡️ AI 語音深偽檢測系統")
st.write("透過聲學物理特徵 (高頻偽影 , 短時過零率變異數 , 頻譜平坦度) 分析音訊真偽 , 結果僅供聲學參考。")

uploaded_file = st.file_uploader("請上傳要檢測的音訊檔案", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    audio_bytes = uploaded_file.getvalue()
    st.audio(audio_bytes)

    if st.button("開始進行聲學分析", type="primary"):
        with st.spinner("正在提取頻譜與聲學特徵中..."):
            suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                y, sr = librosa.load(tmp_path, sr=None, mono=True)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            if y.size == 0:
                st.warning("無法讀取有效的音訊訊號 , 請更換檔案後再試。")
                st.stop()

            # 1. 頻譜平坦度 (Spectral Flatness)
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

            # 2. 過零率變異數 (Zero-Crossing Rate Variance)
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_var = float(np.var(zcr))

            # 3. 高頻偽影能量比 (High-Frequency Artifacts)
            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            high_freq_mask = freqs > 6000
            if np.any(high_freq_mask):
                high_energy = float(np.sum(S[high_freq_mask, :]))
                total_energy = float(np.sum(S)) + 1e-8
                high_freq_ratio = high_energy / total_energy
            else:
                high_freq_ratio = 0.0

            # 依實測真假音差異調整權重與門檻 (可再依更多樣本微調)
            # 真音參考: flatness ~ 0.2344 , zcr_var ~ 0.0046 , high_freq_ratio ~ 0.0354
            # 假音參考: flatness ~ 0.2539 , zcr_var ~ 0.0165 , high_freq_ratio ~ 0.1263
            anomaly_score = (
                (flatness * 0.4)
                + (zcr_var * 12.0)
                + (high_freq_ratio * 2.5)
            )
            threshold = 0.45

            st.subheader("📊 聲學分析特徵指標")
            col1, col2, col3 = st.columns(3)
            col1.metric("頻譜平坦度", f"{flatness:.4f}")
            col2.metric("過零率變異數", f"{zcr_var:.4f}")
            col3.metric("高頻能量偽影比", f"{high_freq_ratio:.4f}")

            st.caption(f"綜合異常參考分數: {anomaly_score:.4f} (目前參考門檻: {threshold:.2f})")
            st.divider()

            if anomaly_score > threshold:
                st.error("⚠️ **檢測警告**:此音訊含有疑似 AI 合成或深偽 (Deepfake) 的異常聲學特徵 , 僅供參考。")
            else:
                st.success("✅ **檢測結果**:此音訊特徵未達明顯異常門檻 , 偏向正常語音範圍 , 僅供參考。")
