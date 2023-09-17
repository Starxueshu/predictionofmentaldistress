# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("An artificial intelligence tool to assess the risk of severe mental distress among college students: an externally validated study using machine learning.")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

Grade = st.sidebar.selectbox("Grade", ("First", "Second", "Third", "Fourth", "Delayed"))
Smoking = st.sidebar.selectbox("Smoking", ("No", "Quitted", "Current"))
Fatfood = st.sidebar.selectbox("Fat food", ("No", "Yes"))
Monthlyexpense = st.sidebar.selectbox("Monthly expense (￥)", ("﹤2000", "≧2000 and <5000", "≧5000 and <10000", "≧10000"))
Chronicdisease = st.sidebar.selectbox("Chronic disease", ("None", "Existed"))
Age = st.sidebar.slider("Age", 18, 28)
PSQI = st.sidebar.slider("PSQI", 0, 21)

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round.pkl")
    x = pd.DataFrame([[Grade, Smoking, Fatfood, Monthlyexpense, Chronicdisease, Age, PSQI]],
                     columns=["Grade", "Smoking", "Fatfood", "Monthlyexpense", "Chronicdisease", "Age", "PSQI"])
    x = x.replace(["First", "Second", "Third", "Fourth", "Delayed"], [1, 2, 3, 4, 5])
    x = x.replace(["No", "Quitted", "Current"], [1, 2, 3])
    x = x.replace(["No", "Yes"], [0, 1])
    x = x.replace(["﹤2000", "≧2000 and <5000", "≧5000 and <10000", "≧10000"], [1, 2, 3, 4])
    x = x.replace(["None", "Existed"], [2, 1])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.success(f"Probability of severe mental distress: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.568:
        st.success(f"Risk group: low-risk group")
    else:
        st.success(f"Risk group: High-risk group")
    if prediction < 0.568:
        st.markdown(f"For university students identified as high-risk individuals with severe mental distress, a comprehensive management approach is imperative to address their specific needs. Firstly, a multidisciplinary team comprising mental health professionals, counselors, and medical practitioners should be involved in their care. This team can collaborate to develop personalized treatment plans tailored to the individual’s condition. Intensive therapy sessions, such as cognitive-behavioral therapy (CBT) or dialectical behavior therapy (DBT), can be implemented to help these students develop coping mechanisms and improve their emotional well-being. Additionally, pharmacological interventions, under the guidance of a psychiatrist, may be considered to alleviate symptoms and stabilize their mental health. Regular follow-up appointments and close monitoring of their progress are crucial to ensure the effectiveness of the management plan. It is crucial to acknowledge that although the AI application offers risk estimates and recommendations, clinical decision-making should encompass the expertise of healthcare providers and take into account the unique context of each student.")
    else:
        st.markdown(f"University students identified as low-risk individuals in terms of severe anxiety and depression require a different approach to management. While their mental health concerns may be less severe, proactive measures should still be taken to promote their overall well-being and prevent the development of more significant issues. One key aspect is the provision of mental health education and awareness programs on campus. These initiatives can help students recognize the signs of distress and equip them with self-help strategies to manage stress and maintain good mental health. Additionally, establishing a supportive environment through peer support groups or mentoring programs can foster a sense of belonging and provide a platform for students to share their experiences and seek guidance. Regular check-ins with university counselors or mental health professionals can also be beneficial to address any emerging concerns promptly. By implementing these preventive measures, the university can create a nurturing environment that supports the mental well-being of all students, including those at low risk for severe mental distress. It is crucial to acknowledge that although the AI application offers risk estimates and recommendations, clinical decision-making should encompass the expertise of healthcare providers and take into account the unique context of each student.")
st.subheader('Model information')
st.markdown('This AI tool was developed based on the eXGBM model with the highest AUC value of 0.932 in the study. To further validate its efficacy, external validation was conducted, yielding an impressive AUC value of 0.918, thereby substantiating its exceptional predictive capabilities.')