import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import os
import zipfile

zip_name = "archive.zip"
file_name = "consumer_complaints.csv"

def run():
    st.title("📊 Exploratory Data Analysis")
    st.write("Consumer Complaints Dataset Overview")

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

    # Detect target column name (handle both variants)
    target_col = None
    for c in ['consumer_disputed?', 'consumer_disputed']:
        if c in df.columns:
            target_col = c
            break

    # ── Dataset Info ──────────────────────────────────────────────
    st.subheader("Dataset Info")
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # ── Missing Values ────────────────────────────────────────────
    st.subheader("Missing Values")
    missing = df.isnull().sum().reset_index()
    missing.columns = ['Column', 'Missing Count']
    missing = missing[missing['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    if missing.empty:
        st.success("No missing values found!")
    else:
        st.dataframe(missing, use_container_width=True)

    st.divider()

    # ── Analisis 1: Top 10 Companies ──────────────────────────────
    st.subheader("1. Top 10 Companies dengan Keluhan Terbanyak")

    result = (
        df.groupby('company').agg(
            total_complaint=('complaint_id', 'count'),
            product_name=('product', lambda x: x.mode().iloc[0]),
            total_product=('product', 'count'),
            sub_product_name=('sub_product', lambda x: x.mode().iloc[0] if not x.mode().empty else '-'),
            total_sub_product=('sub_product', 'count')
        ).reset_index().sort_values('total_complaint', ascending=False)
    )
    top10 = result.head(10)

    fig1, ax1 = plt.subplots(figsize=(14, 7))
    sns.barplot(data=top10, x='total_complaint', y='company', ax=ax1, palette='Blues_d')
    for i, row in enumerate(top10.itertuples()):
        ax1.text(
            row.total_complaint + 500,
            i,
            f"{row.product_name} | {row.sub_product_name}",
            va='center',
            fontsize=9
        )
    ax1.set_title('Top 10 Companies with Highest Complaints')
    ax1.set_xlabel('Total Complaints')
    ax1.set_ylabel('Company')
    plt.tight_layout()
    st.pyplot(fig1)

    st.dataframe(
        top10[['company', 'total_complaint', 'product_name', 'sub_product_name']].reset_index(drop=True),
        use_container_width=True
    )

    st.info("""
**Insight:**
- **Bank of America, Wells Fargo, JPMorgan Chase, Ocwen, dan Nationstar Mortgage** didominasi keluhan produk **Mortgage** (sub-produk: Other mortgage).
- **Equifax, Experian, dan TransUnion** didominasi produk **Credit Reporting**.
- **Citibank** dan **Capital One** paling banyak dikeluhkan pada produk **Credit Card**.

**Business Insight:** Keluhan terkonsentrasi pada tiga sektor utama — Mortgage, Credit Reporting, dan Credit Card.
""")

    st.divider()

    # ── Analisis 2: Chi-Square & Hubungan dengan Consumer Disputed ─
    if target_col:
        st.subheader("2. Hubungan Issue, State & Company Response terhadap Consumer Disputed")

        # Chi-Square test results
        col_tests = ['issue', 'state', 'company_response_to_consumer']
        chi_results = []
        for col in col_tests:
            if col in df.columns:
                contingency = pd.crosstab(df[col], df[target_col])
                chi2, p_value, dof, _ = chi2_contingency(contingency)
                chi_results.append({'Variabel': col, 'Chi2 Statistic': round(chi2, 2), 'p-value': round(p_value, 6), 'Signifikan (p<0.05)': '✅ Ya' if p_value < 0.05 else '❌ Tidak'})

        if chi_results:
            st.write("**Hasil Chi-Square Test of Independence:**")
            st.dataframe(pd.DataFrame(chi_results), use_container_width=True)

        # Plot 1: Company Response vs Dispute
        if 'company_response_to_consumer' in df.columns:
            response_dispute = pd.crosstab(
                df['company_response_to_consumer'],
                df[target_col],
                normalize='index'
            ) * 100

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            response_dispute.plot(kind='bar', stacked=True, ax=ax2, colormap='coolwarm')
            ax2.set_title('Consumer Dispute by Company Response')
            ax2.set_xlabel('Company Response')
            ax2.set_ylabel('Percentage (%)')
            ax2.legend(title='Consumer Disputed', bbox_to_anchor=(1.01, 1), loc='upper left')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig2)

        # Plot 2: Top 10 Issues with highest dispute rate
        if 'issue' in df.columns:
            issue_dispute = (
                pd.crosstab(df['issue'], df[target_col], normalize='index')
            )
            if 'Yes' in issue_dispute.columns:
                issue_dispute = issue_dispute['Yes'].sort_values(ascending=False).head(10)
            else:
                issue_dispute = issue_dispute.iloc[:, -1].sort_values(ascending=False).head(10)

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            issue_dispute.plot(kind='barh', ax=ax3, color='#DD8452')
            ax3.set_title('Top 10 Issues with Highest Dispute Rate')
            ax3.set_xlabel('Dispute Rate')
            ax3.set_ylabel('Issue')
            ax3.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig3)

        st.info("""
**Insight:**
- **Issue** memiliki hubungan signifikan terhadap status dispute (p-value < 0.05).
- **State** juga berkorelasi signifikan dengan status dispute konsumen.
- **Company Response** memiliki Chi-Square terbesar (~7961), menunjukkan jenis respons perusahaan sangat berkaitan dengan apakah konsumen melakukan dispute atau tidak.
""")
    else:
        st.warning("Kolom `consumer_disputed` tidak ditemukan di dataset.")
