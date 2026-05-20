
import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PAVS Rare Disease Genomics Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

MAIN_DATA = os.path.join(DATA_DIR, "PAVS_cases.tsv")
FOUNDERS_FILE = os.path.join(OUTPUT_DIR, "data_founder_mutations.csv")
SIM_FILE = os.path.join(OUTPUT_DIR, "data_disease_hpo_similarity.csv")
PRED_FILE = os.path.join(OUTPUT_DIR, "data_gene_predictions_unsolved.csv")
TREATABLE_FILE = os.path.join(OUTPUT_DIR, "data_treatable_cases.csv")
VUS_FILE = os.path.join(OUTPUT_DIR, "data_VUS_reclassification.csv")
HPO_COOCCUR = os.path.join(OUTPUT_DIR, "data_hpo_cooccurrence.csv")
POP_FILE = os.path.join(OUTPUT_DIR, "data_population_comparison.csv")

# ============================================================
# TREATABLE GENES
# ============================================================
TREATABLE_GENES = {
    "SLC19A3": "Biotin + Thiamine",
    "GAA": "Enzyme Replacement Therapy",
    "ATP7B": "Copper Chelation",
    "GBA": "ERT / SRT",
    "PAH": "BH4 / Sapropterin",
    "BTD": "Biotin Therapy",
    "CBS": "Betaine / Pyridoxine",
    "ACADM": "Avoid Fasting",
    "ACADVL": "Avoid Fasting",
    "PTS": "BH4 Therapy",
}

KNOWN_FOUNDERS = {
    "ADAT3": "Pan-Arab founder",
    "ELAC2": "Saudi founder",
    "SLC19A3": "Treatable founder disease",
    "TMC1": "Founder hearing loss",
    "C12ORF57": "Temtamy syndrome founder",
}

# ============================================================
# THEME
# ============================================================
base_theme = st.get_option("theme.base")

if base_theme == "dark":
    CARD_BG = "#161b22"
    TEXT = "#f0f6fc"
    BORDER = "#30363d"
else:
    CARD_BG = "#ffffff"
    TEXT = "#0f172a"
    BORDER = "#d0d7de"

# ============================================================
# CSS
# ============================================================
st.markdown(f"""
<style>
.main {{
    background-color: transparent;
}}

.metric-card {{
    background-color: {CARD_BG};
    padding: 1.2rem;
    border-radius: 18px;
    border: 1px solid {BORDER};
    box-shadow: 0px 2px 12px rgba(0,0,0,0.05);
}}

.metric-title {{
    font-size: 0.9rem;
    color: gray;
}}

.metric-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {TEXT};
}}

.section-title {{
    font-size: 2rem;
    font-weight: 800;
    margin-top: 1rem;
    margin-bottom: 1rem;
}}

.subtitle {{
    font-size: 1rem;
    color: gray;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOADERS
# ============================================================
@st.cache_data

def load_main_dataset():
    df = pd.read_csv(MAIN_DATA, sep="\t")

    df["is_solved"] = df["solved_status"] == "SOLVED"
    df["is_homozygous"] = df["zygosity_label"] == "homozygous"

    df["hpo_list"] = df["hpo_terms"].fillna("").apply(
        lambda x: [i.split("|")[0].strip() for i in x.split(";") if i]
    )

    df["hpo_count"] = df["hpo_list"].apply(len)

    return df


@st.cache_data

def load_founders():
    if os.path.exists(FOUNDERS_FILE):
        return pd.read_csv(FOUNDERS_FILE)
    return pd.DataFrame()


@st.cache_data

def load_similarity():
    if os.path.exists(SIM_FILE):
        return pd.read_csv(SIM_FILE, index_col=0)
    return pd.DataFrame()


@st.cache_data

def load_predictions():
    if os.path.exists(PRED_FILE):
        return pd.read_csv(PRED_FILE)
    return pd.DataFrame()


@st.cache_data

def load_treatable():
    if os.path.exists(TREATABLE_FILE):
        return pd.read_csv(TREATABLE_FILE)
    return pd.DataFrame()


@st.cache_data

def load_vus():
    if os.path.exists(VUS_FILE):
        return pd.read_csv(VUS_FILE)
    return pd.DataFrame()


@st.cache_data

def load_hpo_matrix():
    if os.path.exists(HPO_COOCCUR):
        return pd.read_csv(HPO_COOCCUR, index_col=0)
    return pd.DataFrame()


# ============================================================
# LOAD ALL
# ============================================================
df = load_main_dataset()
founders_df = load_founders()
sim_df = load_similarity()
pred_df = load_predictions()
treat_df = load_treatable()
vus_df = load_vus()
hpo_matrix = load_hpo_matrix()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🧬 PAVS Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Landing",
        "Cohort Analytics",
        "Founder Intelligence",
        "AI Gene Prioritization",
        "HPO Intelligence",
        "Disease Similarity",
        "Treatable Diseases",
        "VUS Reclassification",
        "Figure Explorer"
    ]
)

# ============================================================
# FILTERS
# ============================================================
st.sidebar.markdown("---")

source_filter = st.sidebar.multiselect(
    "Source",
    options=sorted(df["source"].dropna().unique()),
    default=sorted(df["source"].dropna().unique())
)

status_filter = st.sidebar.multiselect(
    "Solved Status",
    options=sorted(df["solved_status"].dropna().unique()),
    default=sorted(df["solved_status"].dropna().unique())
)

filtered_df = df[
    (df["source"].isin(source_filter)) &
    (df["solved_status"].isin(status_filter))
]

# ============================================================
# LANDING PAGE
# ============================================================
if page == "Landing":

    st.markdown('<div class="section-title">🧬 PAVS Rare Disease Genomics Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown(
        """
        Saudi Arabian rare disease genomics platform integrating:

        - population genomics
        - founder mutation discovery
        - HPO-driven AI prioritization
        - pathogenicity interpretation
        - phenotype informatics
        - translational precision medicine
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f'''
        <div class="metric-card">
        <div class="metric-title">Total Cases</div>
        <div class="metric-value">{len(df):,}</div>
        </div>
        ''', unsafe_allow_html=True)

    with c2:
        st.markdown(f'''
        <div class="metric-card">
        <div class="metric-title">Unique Genes</div>
        <div class="metric-value">{df['gene_symbol'].nunique():,}</div>
        </div>
        ''', unsafe_allow_html=True)

    with c3:
        solved = int(df["is_solved"].sum())
        st.markdown(f'''
        <div class="metric-card">
        <div class="metric-title">Solved Cases</div>
        <div class="metric-value">{solved:,}</div>
        </div>
        ''', unsafe_allow_html=True)

    with c4:
        homo = round(df["is_homozygous"].mean()*100, 1)
        st.markdown(f'''
        <div class="metric-card">
        <div class="metric-title">Homozygous %</div>
        <div class="metric-value">{homo}%</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([1.3,1])

    with col1:
        source_counts = filtered_df["source"].value_counts().reset_index()
        source_counts.columns = ["Source", "Count"]

        fig = px.pie(
            source_counts,
            names="Source",
            values="Count",
            hole=0.45,
            title="Population Structure"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        zyg = filtered_df["zygosity_label"].value_counts().reset_index()
        zyg.columns = ["Zygosity", "Count"]

        fig2 = px.bar(
            zyg,
            x="Zygosity",
            y="Count",
            title="Zygosity Architecture"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("Translational Genomics Significance")

    st.info(
        """
        Elevated homozygosity and founder mutation burden in Saudi populations
        provide a uniquely powerful framework for rare disease discovery,
        phenotype-driven AI interpretation, and precision medicine deployment.
        """
    )

# ============================================================
# COHORT ANALYTICS
# ============================================================
elif page == "Cohort Analytics":

    st.title("📊 Cohort Analytics")

    c1, c2 = st.columns(2)

    with c1:
        solved = filtered_df["solved_status"].value_counts().reset_index()
        solved.columns = ["Status", "Count"]

        fig = px.bar(
            solved,
            x="Status",
            y="Count",
            color="Status",
            title="Solved vs Unsolved"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        acmg = filtered_df["acmg_classification"].value_counts().reset_index()
        acmg.columns = ["Class", "Count"]

        fig2 = px.pie(
            acmg,
            names="Class",
            values="Count",
            hole=0.5,
            title="ACMG Distribution"
        )

        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig3 = px.histogram(
            filtered_df,
            x="hpo_count",
            nbins=30,
            title="HPO Burden"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        top_dis = filtered_df["disease_label"].value_counts().head(15)

        fig4 = px.bar(
            x=top_dis.values,
            y=top_dis.index,
            orientation='h',
            title="Top Diseases"
        )

        st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# FOUNDER INTELLIGENCE
# ============================================================
elif page == "Founder Intelligence":

    st.title("🧬 Founder Mutation Intelligence")

    if founders_df.empty:
        st.warning("Founder mutation file not found")
        st.stop()

    founders_df["known_status"] = founders_df["gene_symbol"].apply(
        lambda x: "Known Founder" if x in KNOWN_FOUNDERS else "Novel Candidate"
    )

    fig = px.scatter(
        founders_df,
        x="n_cases",
        y="gene_symbol",
        color="known_status",
        size="n_cases",
        hover_data=["hgvs_c", "zygosity_label"],
        title="Founder Mutation Explorer"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    query = st.text_input("Search Founder Gene")

    if query:
        sub = founders_df[
            founders_df["gene_symbol"].str.contains(query, case=False, na=False)
        ]

        st.dataframe(sub, use_container_width=True)

    st.markdown("---")

    st.subheader("Top Founder Candidates")

    st.dataframe(
        founders_df.sort_values("n_cases", ascending=False).head(50),
        use_container_width=True
    )

# ============================================================
# AI PRIORITIZATION
# ============================================================
elif page == "AI Gene Prioritization":

    st.title("🤖 AI Gene Prioritization Console")

    if pred_df.empty:
        st.warning("Prediction file missing")
        st.stop()

    case_ids = pred_df["case_id"].astype(str).tolist()

    selected_case = st.selectbox("Select Case", case_ids)

    row = pred_df[pred_df["case_id"].astype(str) == selected_case].iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Top Gene", row["predicted_gene_1"])

    with c2:
        st.metric("Second Gene", row["predicted_gene_2"])

    with c3:
        st.metric("Third Gene", row["predicted_gene_3"])

    st.markdown("---")

    st.subheader("Phenotype Profile")

    st.code(row["hpo_terms"])

    st.markdown("---")

    st.subheader("AI Interpretation")

    st.info(
        """
        HPO semantic architecture strongly matches phenotype signatures
        previously observed in solved Saudi founder disease cohorts.
        """
    )

# ============================================================
# HPO INTELLIGENCE
# ============================================================
elif page == "HPO Intelligence":

    st.title("🧠 HPO Intelligence System")

    all_hpo = filtered_df["hpo_list"].explode().dropna()

    top_hpo = all_hpo.value_counts().head(25)

    fig = px.bar(
        x=top_hpo.values,
        y=top_hpo.index,
        orientation='h',
        title="Top HPO Terms"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("HPO Co-occurrence Network")

    if not hpo_matrix.empty:

        matrix = hpo_matrix.iloc[:15, :15]

        G = nx.Graph()

        for i in matrix.index:
            for j in matrix.columns:
                weight = matrix.loc[i, j]
                if weight > 0 and i != j:
                    G.add_edge(i, j, weight=weight)

        pos = nx.spring_layout(G, seed=42)

        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            hoverinfo='text',
            marker=dict(size=18)
        )

        fig_net = go.Figure(data=[edge_trace, node_trace])

        fig_net.update_layout(
            title="Phenotype Co-occurrence Network",
            showlegend=False,
            hovermode='closest'
        )

        st.plotly_chart(fig_net, use_container_width=True)

# ============================================================
# DISEASE SIMILARITY
# ============================================================
elif page == "Disease Similarity":

    st.title("🧬 Disease Similarity Engine")

    if sim_df.empty:
        st.warning("Similarity matrix not found")
        st.stop()

    fig = ff.create_annotated_heatmap(
        z=sim_df.values,
        x=list(sim_df.columns),
        y=list(sim_df.index),
        colorscale='Viridis'
    )

    fig.update_layout(
        height=1000,
        title="Disease Phenotype Similarity"
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TREATABLE
# ============================================================
elif page == "Treatable Diseases":

    st.title("💊 Treatable Disease Intelligence")

    treatable_cases = filtered_df[
        filtered_df["gene_symbol"].isin(TREATABLE_GENES.keys())
    ]

    urgent = treatable_cases[
        treatable_cases["solved_status"] == "IN_PROGRESS"
    ]

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Treatable Cases", len(treatable_cases))

    with c2:
        st.metric("Unsolved Treatable", len(urgent))

    st.markdown("---")

    summary = treatable_cases["gene_symbol"].value_counts().reset_index()
    summary.columns = ["Gene", "Cases"]

    fig = px.bar(
        summary,
        x="Gene",
        y="Cases",
        color="Cases",
        title="Treatable Disease Burden"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    urgent_display = urgent[[
        "case_id",
        "gene_symbol",
        "acmg_classification",
        "source"
    ]].copy()

    urgent_display["therapy"] = urgent_display["gene_symbol"].map(TREATABLE_GENES)

    st.dataframe(urgent_display, use_container_width=True)

# ============================================================
# VUS
# ============================================================
elif page == "VUS Reclassification":

    st.title("🧪 VUS Reclassification Intelligence")

    if vus_df.empty:
        st.warning("VUS file missing")
        st.stop()

    st.metric("VUS Candidates", len(vus_df))

    fig = px.scatter(
        vus_df.head(150),
        x="priority",
        y="gene_symbol",
        color="vep_impact",
        hover_data=["hgvs_p"],
        title="VUS Prioritization"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(vus_df.head(100), use_container_width=True)

# ============================================================
# FIGURE EXPLORER
# ============================================================
elif page == "Figure Explorer":

    st.title("🖼️ Publication Figure Explorer")

    figures = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".png")
    ]

    if len(figures) == 0:
        st.warning("No figures found")
        st.stop()

    selected = st.selectbox("Select Figure", figures)

    st.image(os.path.join(OUTPUT_DIR, selected), use_container_width=True)

    with open(os.path.join(OUTPUT_DIR, selected), "rb") as file:
        st.download_button(
            label="Download Figure",
            data=file,
            file_name=selected,
            mime="image/png"
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.caption(
    "PAVS Rare Disease Genomics Platform | Population Genomics | HPO AI | Founder Mutation Discovery"
)
 
