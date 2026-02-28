import streamlit as st
import joblib
import pandas as pd

# モデルの読み込み
model = joblib.load("pta_model.pkl")

# タイトルと説明
st.title("PTA予測アプリ 大石内科クリニック")
st.write("各項目を選択・入力して「予測する」ボタンを押してください。")

# 入力フォーム（左側に項目名、右側に選択肢）
FV = st.number_input("FV (血流量)", value=300.0)
RI = st.number_input("RI (抵抗指数)", value=0.6)

# 選択肢 0:なし, 1:あり
PoorFlow = st.selectbox("脱血不良 (0:なし, 1:あり)", [0, 1])
IntermittentSound = st.selectbox("断続音 (0:なし, 1:あり)", [0, 1])
WeakSound = st.selectbox("シャント音減弱 (0:なし, 1:あり)", [0, 1])
Depression = st.selectbox("シャント挙上にて凹み (0:なし, 1:あり)", [0, 1])
StenosisSound = st.selectbox("狭窄音聴取 (0:なし, 1:あり)", [0, 1])
Edema = st.selectbox("浮腫み (0:なし, 1:あり)", [0, 1])
Recirculation = st.selectbox("再循環率上昇 (0:なし, 1:あり)", [0, 1])
PoorHemostasis = st.selectbox("止血不良 (0:なし, 1:あり)", [0, 1])
PalpableStenosis = st.selectbox("狭窄部触知 (0:なし, 1:あり)", [0, 1])

# 予測ボタン
if st.button("予測する"):
    # データをまとめる（モデルが学習した時と同じ順番にする必要があります）
    input_dict = {
        "PoorFlow": PoorFlow,
        "IntermittentSound": IntermittentSound,
        "WeakSound": WeakSound,
        "Depression": Depression,
        "StenosisSound": StenosisSound,
        "Edema": Edema,
        "Recirculation": Recirculation,
        "PoorHemostasis": PoorHemostasis,
        "PalpableStenosis": PalpableStenosis,
        "FV": FV,
        "RI": RI
    }
    
    input_df = pd.DataFrame([input_dict])

    try:
        # 確率を計算
        prob = model.predict_proba(input_df)[0][1]

        # 結果の表示
        st.subheader(f"PTAの確率: {prob:.2%}")

        if prob > 0.5:
            st.error("⚠️ PTA高リスク（治療の検討をお勧めします）")
        else:
            st.success("✅ PTA低リスク（経過観察）")
            
    except Exception as e:
        st.error(f"エラーが発生しました。モデルの項目設定を確認してください。")

