import streamlit as st
import pandas as pd
import joblib
import os
import zipfile

zip_name = "archive.zip"
file_name = "consumer_complaints.csv"

def run():
    st.title("🔮 Complaint Dispute Prediction")
    st.write("Fill in the complaint details below to predict whether it will become a dispute.")

    # Load model
    @st.cache_resource
    def load_model():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return joblib.load(os.path.join(base_dir, 'modeling/best_model.pkl'))

    @st.cache_data
    def load_data():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            with zipfile.ZipFile(os.path.join(base_dir, 'modeling', zip_name), 'r') as zip_ref:
                        zip_ref.extractall(os.path.join(base_dir, 'modeling'))
        except:
             st.write("Error extracting the zip file. Please ensure the zip file exists and is not corrupted.")
             
        return pd.read_csv(os.path.join(base_dir,'modeling/consumer_complaints.csv'))
    
    df = load_data()
    model = load_model()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            product = st.selectbox("Product", [item for item in df['product'].unique() if pd.notnull(item)])
            sub_product = st.selectbox("Sub Product",[item for item in df['sub_product'].unique() if pd.notnull(item)])
            issue = st.selectbox("Issue", [item for item in df['issue'].unique() if pd.notnull(item)])
            sub_issue = st.selectbox("Sub Issue", [item for item in df['sub_issue'].unique() if pd.notnull(item)])
            company = st.selectbox("Company", [item for item in df['company'].unique() if pd.notnull(item)])
            state = st.selectbox("State", [item for item in df['state'].unique() if pd.notnull(item)])
            tags = st.selectbox("Tags", [item for item in df['tags'].unique() if pd.notnull(item)])
            received_week = st.number_input("Received Week", min_value=1, max_value=52, value=1)

        with col2:
            consumer_consent_provided = st.selectbox("Consumer Consent Provided",[item for item in df['consumer_consent_provided'].unique() if pd.notnull(item)])
            submitted_via = st.selectbox("Submitted Via", [item for item in df['submitted_via'].unique() if pd.notnull(item)])
            company_response_to_consumer = st.selectbox("Company Response to Consumer",[item for item in df['company_response_to_consumer'].unique() if pd.notnull(item)])
            company_public_response = st.selectbox(
                "Company Public Response",
                [item for item in df['company_public_response'].unique() if pd.notnull(item)]
            )
            timely_response = st.selectbox("Timely Response", ['Yes', 'No'])
            response_days = st.number_input("Response Days", min_value=0, max_value=365, value=5)
            received_month = st.number_input("Received Month", min_value=1, max_value=12, value=10)
            received_day = st.number_input("Received Day", min_value=1, max_value=31, value=1)

        month_in_quarter = st.number_input("Month in Quarter", min_value=1, max_value=4, value=1)
        consumer_complaint_narrative = st.selectbox(
            "Consumer Complaint Narrative", [item for item in df.consumer_complaint_narrative if pd.notnull(item)], index=0, help="Select a sample narrative from the dataset"
        )

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        data_inf = pd.DataFrame({
            'product': [product],
            'sub_product': [sub_product],
            'issue': [issue],
            'sub_issue': [sub_issue],
            'consumer_complaint_narrative': [consumer_complaint_narrative],
            'company_public_response': [company_public_response],
            'company': [company],
            'state': [state],
            'tags': [tags],
            'consumer_consent_provided': [consumer_consent_provided],
            'submitted_via': [submitted_via],
            'company_response_to_consumer': [company_response_to_consumer],
            'timely_response': [timely_response],
            'response_days': [response_days],
            'received_month': [received_month],
            'received_week' : [received_week],
            'received_day': [received_day],
            'month_in_quarter': [month_in_quarter]  
        })

        pred = model.predict(data_inf)[0]

        st.divider()
        if pred == 1:
            st.error("⚠️ Complaint diprediksi **AKAN** menjadi dispute.")
        else:
            st.success("✅ Complaint diprediksi **TIDAK** akan menjadi dispute.")
