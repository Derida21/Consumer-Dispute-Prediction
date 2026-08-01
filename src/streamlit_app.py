import streamlit as st
import eda
import prediction

st.set_page_config(
    page_title="Consumer Complaints App",
    page_icon="📋",
    layout="wide"
)

page = st.sidebar.selectbox(
    'Select Page',
    ['EDA', 'Prediction']
)

if page == 'EDA':
    eda.run()
else:
    prediction.run()
