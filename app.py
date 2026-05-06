import streamlit as st
from src.analysis import build_system

st.set_page_config(layout="wide")
st.title("Population-Aware Rare Disease Discovery Platform")

@st.cache_data
def load():
    return build_system()

data = load()

df = data["df"]

page = st.sidebar.radio("Navigation", [
    "Overview",
    "Founder Mutations",
    "Treatable Patients",
    "VUS Reclassification",
    "Patient Explorer"
])

# ─────────────────────────
if page == "Overview":
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Cases", data["total_cases"])
    col2.metric("Solved", data["solved"])
    col3.metric("Homozygous %", f"{data['hom_pct']*100:.1f}%")

    st.subheader("Key Insight")
    st.info("High homozygosity indicates strong consanguinity-driven disease architecture.")

# ─────────────────────────
elif page == "Founder Mutations":
    st.subheader("Top Recurrent Variants")

    st.dataframe(
        data["founders"].head(30)
    )

# ─────────────────────────
elif page == "Treatable Patients":
    st.subheader("Unsolved Patients with Treatable Genes")

    st.warning(f"{len(data['treatable_unsolved'])} patients could benefit TODAY")

    st.dataframe(
        data["treatable_unsolved"][
            ["case_id","gene_symbol","hpo_terms"]
        ].head(50)
    )

# ─────────────────────────
elif page == "VUS Reclassification":
    st.subheader("Top VUS Candidates")

    threshold = st.slider("Score threshold", 0.0, 10.0, 3.0)

    filtered = data["vus"][data["vus"]["score"] > threshold]

    st.dataframe(
        filtered[
            ["gene_symbol","hgvs_c","score","hpo_terms"]
        ].head(50)
    )

# ─────────────────────────
elif page == "Patient Explorer":
    st.subheader("Patient-Level Analysis")

    case = st.selectbox("Select Case", df["case_id"].unique())

    case_df = df[df["case_id"] == case]

    st.write(case_df.T)
