import streamlit as st
import joblib
import pandas as pd

model = joblib.load("pta_model.pkl")

st.title("PTA予測アプリ　大石内科クリニック")
st.write("各項目を入力してください")

FV = st.number_input("FV", value=300.0)
RI = st.number_input("RI", value=0.6)
PoorFlow = st.selectbox("脱血不良", [0,1])
IntermittentSound = st.selectbox("断続音", [0,1])
WeakSound = st.selectbox("シャント音減弱", [0,1])
Depression = st.selectbox("シャント挙上にて凹み", [0,1])
StenosisSound = st.selectbox("狭窄音聴取", [0,1])
Edema = st.selectbox("浮腫み", [0,1])
Recirculation = st.selectbox("再循環率上昇", [0,1])
PoorHemostasis = st.selectbox("止血不良", [0,1])
PalpableStenosis = st.selectbox("狭窄音聴取", [0,1])


if st.button("予測する"):
    input_data = pd.DataFrame([[PoorFlow, IntermittentSound, WeakSound,
                                Depression, StenosisSound, Edema,
                                Recirculation, PoorHemostasis,
                                PalpableStenosis, FV, RI]],
                              columns=["PoorFlow","IntermittentSound","WeakSound",
                                       "Depression","StenosisSound","Edema",
                                       "Recirculation","PoorHemostasis",
                                       "PalpableStenosis","FV","RI"])

    prob = model.predict_proba(input_data)[0][1]

    st.subheader(f"PTA確率: {prob:.2%}")

    if prob > 0.5:
        st.error("PTA高リスク")
    else:
        st.success("PTA低リスク")

