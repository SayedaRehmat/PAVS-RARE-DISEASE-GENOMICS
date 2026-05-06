import streamlit as st
from src.analysis import build_system

st.set_page_config(layout="wide")
st.title("Population-Aware Rare Disease Discovery Platform")

@st.cache_data
def load():
    return build_system()

data = load()

df = data["df"]

# Sidebar navigation
page = st.sidebar.radio("Navigation", [
    "Overview",
    "Founder Mutations",
    "Treatable Patients",
    "VUS Reclassification",
    "Patient Explorer"
])

# ─────────────────────────────────────
if page == "Overview":
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Cases", data["overview"]["total_cases"])
    col2.metric("Solved Cases", data["overview"]["solved_cases"])
    col3.metric("Homozygous %", f"{data['overview']['homozygous_pct']*100:.1f}%")

    st.info("High homozygosity suggests strong consanguinity-driven disease patterns.")

# ─────────────────────────────────────
elif page == "Founder Mutations":
    st.subheader("Top Recurrent Variants")

    st.dataframe(
        data["founders"].head(30),
        use_container_width=True
    )

# ─────────────────────────────────────
elif page == "Treatable Patients":
    st.subheader("Unsolved Patients with Treatable Conditions")

    st.warning(f"{len(data['treatable_unsolved'])} patients may benefit from existing treatments")

    st.dataframe(
        data["treatable_unsolved"][
            ["case_id", "gene_symbol", "treatment", "hpo_terms"]
        ].head(50),
        use_container_width=True
    )

# ─────────────────────────────────────
elif page == "VUS Reclassification":
    st.subheader("Top VUS Candidates")

    threshold = st.slider("Minimum Score", 0.0, 10.0, 3.0)

    filtered = data["vus"][data["vus"]["score"] > threshold]

    st.dataframe(
        filtered[
            ["gene_symbol", "hgvs_c", "score", "hpo_terms"]
        ].head(50),
        use_container_width=True
    )

# ─────────────────────────────────────
elif page == "Patient Explorer":
    st.subheader("Patient-Level View")

    case = st.selectbox("Select Case ID", df["case_id"].unique())

    case_df = df[df["case_id"] == case]

    st.dataframe(case_df.T)
