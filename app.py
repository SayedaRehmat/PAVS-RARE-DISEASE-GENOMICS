import streamlit as st
import pandas as pd
import plotly.express as px

from analysis_core import *

st.set_page_config(layout="wide")
st.title("🧬 PAVS Clinical Genomics Dashboard")

@st.cache_data
def load():
    return load_data("data/PAVS_cases.tsv")

df,_ = load()

# =========================
# SIDEBAR
# =========================
tab = st.sidebar.radio("Navigation", [
    "Overview","Population","Founders","Treatable",
    "Neuro","Similarity","Gene Model","Pathogenicity",
    "VUS","HPO","ADAT3","Search","Variant Filter"
])

# =========================
# OVERVIEW
# =========================
if tab=="Overview":
    st.plotly_chart(px.histogram(df,x="hpo_count"))

# =========================
# POPULATION
# =========================
elif tab=="Population":
    stats = get_population_stats(df)
    st.dataframe(stats)

# =========================
# FOUNDERS
# =========================
elif tab=="Founders":
    f = get_founders(df)
    st.dataframe(f)

# =========================
# TREATABLE
# =========================
elif tab=="Treatable":
    t = get_treatable_df(df)
    st.plotly_chart(px.bar(t["gene_symbol"].value_counts().reset_index(),
                           x="index",y="gene_symbol"))

# =========================
# NEURO
# =========================
elif tab=="Neuro":
    n = get_neuro(df)
    st.plotly_chart(px.bar(n["gene_symbol"].value_counts().reset_index(),
                           x="index",y="gene_symbol"))

# =========================
# SIMILARITY
# =========================
elif tab=="Similarity":
    sim,names = get_disease_similarity(df)
    st.plotly_chart(px.imshow(sim,x=names,y=names))

# =========================
# GENE MODEL
# =========================
elif tab=="Gene Model":
    if st.button("Run"):
        res = run_gene_model(df)
        st.write(res)

# =========================
# PATHOGENICITY
# =========================
elif tab=="Pathogenicity":
    if st.button("Run"):
        p,r = run_pathogenicity_model(df)
        st.plotly_chart(px.line(x=r,y=p))

# =========================
# VUS
# =========================
elif tab=="VUS":
    st.dataframe(get_vus(df).head(100))

# =========================
# HPO
# =========================
elif tab=="HPO":
    mat,terms = get_hpo_cooccurrence(df)
    st.plotly_chart(px.imshow(mat,x=terms,y=terms))

# =========================
# ADAT3
# =========================
elif tab=="ADAT3":
    st.plotly_chart(px.bar(get_adat3(df)))

# =========================
# SEARCH
# =========================
elif tab=="Search":
    q = st.text_input("Search")
    if q:
        st.dataframe(search_patient(df,q))

# =========================
# VARIANT FILTER
# =========================
elif tab=="Variant Filter":
    gene = st.text_input("Gene")
    hgvs = st.text_input("HGVS")
    st.dataframe(filter_variant(df,gene,hgvs))
