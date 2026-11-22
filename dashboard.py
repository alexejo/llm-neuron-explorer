import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="LLM Neuron Explorer", layout="wide")
st.title("LLM Synaptic Pruning · Neuron Explorer")
st.caption("Measured MLP channel ablation effects. Positive Δ loss means disabling a channel worsens reference-answer likelihood. This is a small pilot, not validated task specialisation.")

uploaded = st.file_uploader("Upload results.csv from Google Colab", type="csv")
if uploaded is None:
    st.info("Upload measured experiment results to explore them. No simulated measurements are shown.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

required = {"layer", "channel", "task", "importance"}
if not required.issubset(df.columns):
    st.error(f"Missing columns: {sorted(required - set(df.columns))}")
    st.stop()
if df.duplicated(["layer", "channel", "task"]).any():
    st.error("Duplicate channel/task measurements. Aggregate replicates before uploading.")
    st.stop()
for col in ["layer", "channel", "importance"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
if df[["layer", "channel", "importance"]].isna().any().any():
    st.error("Layer, channel and importance must contain valid numbers.")
    st.stop()

tasks = sorted(df["task"].unique())
task = st.selectbox("Task", tasks)
subset = df[df["task"] == task].copy()
st.metric("Sampled channels", len(subset))
st.metric("Positive loss changes", int((subset["importance"] > 0).sum()))

if "group" in df.columns and "stage" in df.columns:
    comparison = df[df["stage"].isin(["layer1_test", "target_test"])].copy()
    if not comparison.empty:
        st.subheader("Target layer: linked versus random")
        summary = comparison.groupby("group")["importance"].agg(["mean", "median", "count"])
        st.dataframe(summary, use_container_width=True)
        fig_compare = px.box(comparison, x="group", y="importance", points="all",
                             labels={"importance": "Δ loss", "group": "Selection group"},
                             title="Target-layer ablation loss by selection group")
        fig_compare.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_compare, use_container_width=True)
        st.caption("Exploratory comparison only: the same five coding examples were used for channel selection and evaluation.")

st.subheader("Channel map by transformer layer")
st.caption("Only sampled channels appear. Their horizontal positions are true channel indices; empty space is unmeasured, not zero.")
fig = px.scatter(
    subset, x="channel", y="layer", color="importance",
    color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
    hover_data={"channel": True, "layer": True, "importance": ":.6f"},
    labels={"importance": "Δ loss", "channel": "MLP channel index", "layer": "Transformer layer"},
)
fig.update_traces(marker={"size": 12, "symbol": "square"})
fig.update_layout(height=550)
fig.update_yaxes(autorange="reversed", dtick=1)
st.plotly_chart(fig, use_container_width=True)

wide = df.pivot(index=["layer", "channel"], columns="task", values="importance").reset_index()
if {"python", "translation"}.issubset(wide.columns):
    st.subheader("Coding versus translation")
    paired = wide.dropna(subset=["python", "translation"])
    if len(paired) >= 2:
        st.metric("Pearson correlation", f"{paired['python'].corr(paired['translation']):.3f}")
    fig2 = px.scatter(paired, x="python", y="translation", hover_data=["layer", "channel"],
                      labels={"python": "Python Δ loss", "translation": "Translation Δ loss"})
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Ranked sampled channels")
st.dataframe(subset.sort_values("importance", ascending=False), use_container_width=True, hide_index=True)
st.download_button("Download filtered results", subset.to_csv(index=False), "filtered_results.csv", "text/csv")
