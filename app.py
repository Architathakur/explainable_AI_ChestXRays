import streamlit as st

st.set_page_config(
    page_title="PneumoXAI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

from pathlib import Path

from model_utils import (  # noqa: E402
    generate_gradcam,
    generate_gradcampp,
    generate_ig,
    load_model,
    preprocess_image,
    run_inference,
)


BASE_DIR = Path(__file__).parent
CHECKPOINT_PATH = BASE_DIR / "outputs/cxr_foundation_densenet121-res224-all/best_cxr_foundation.pt"
THRESHOLD_PATH = BASE_DIR / "outputs/threshold_tuning_densenet121-res224-all/threshold_tuning.json"


@st.cache_resource
def get_model():
    base = Path(__file__).parent
    return load_model(
        str(base / "outputs/cxr_foundation_densenet121-res224-all/best_cxr_foundation.pt"),
        str(base / "outputs/threshold_tuning_densenet121-res224-all/threshold_tuning.json")
    )


def inject_global_css() -> None:
    st.markdown(
        """
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
[data-testid="stFileUploader"] {
  border: 1.5px dashed #2a3347;
  border-radius: 10px;
  padding: 8px;
  background: #161b27;
}
.stTabs [data-baseweb="tab"] {
  font-size: 13px;
  padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
  border-bottom: 2px solid #2563eb;
  color: #e2e8f0;
}
[data-testid="stSidebar"] {
  background: #161b27;
  border-right: 0.5px solid #2a3347;
}
[data-testid="stImage"] img {
  border-radius: 8px;
}
[data-testid="metric-container"] {
  background: #161b27;
  border: 0.5px solid #2a3347;
  border-radius: 10px;
  padding: 12px 14px;
}
.px-section-label {
  font-size:10px;
  color:#64748b;
  text-transform:uppercase;
  letter-spacing:0.08em;
  margin:16px 0 8px;
}
.px-divider {
  height:1px;
  background:#2a3347;
  margin:16px 0;
}
.px-muted { color:#64748b; font-size:11px; }
.px-card {
  background:#161b27;
  border:0.5px solid #2a3347;
  border-radius:10px;
  padding:14px 16px;
}
.px-pill {
  display:inline-flex;
  align-items:center;
  border-radius:999px;
  padding:4px 9px;
  background:#0f2a5e;
  color:#bfdbfe;
  font-size:11px;
  margin-bottom:6px;
}
.px-placeholder {
  min-height:360px;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  color:#64748b;
  border:1px dashed #2a3347;
  border-radius:10px;
  background:#121723;
}
.px-file-meta {
  background:#161b27;
  border:0.5px solid #2a3347;
  border-radius:8px;
  padding:10px 12px;
  color:#94a3b8;
  font-size:12px;
}
.px-prob-chip {
  background:#161b27;
  border:0.5px solid #2a3347;
  border-radius:10px;
  padding:12px 14px;
}
.px-warning {
  background:rgba(120, 53, 15, 0.36);
  border:0.5px solid rgba(245, 158, 11, 0.45);
  border-radius:10px;
  padding:12px 14px;
  color:#f8d9a0;
  font-size:13px;
}
.px-xai-guide {
  background:#121723;
  border:0.5px solid #2a3347;
  border-radius:10px;
  padding:14px 16px;
  margin:12px 0 18px;
}
.px-xai-guide-title {
  color:#e2e8f0;
  font-size:13px;
  font-weight:600;
  margin-bottom:8px;
}
.px-xai-guide-grid {
  display:grid;
  grid-template-columns:repeat(3, minmax(0, 1fr));
  gap:10px;
}
.px-xai-guide-item {
  color:#94a3b8;
  font-size:12px;
  line-height:1.45;
}
.px-dot {
  display:inline-block;
  width:9px;
  height:9px;
  border-radius:999px;
  margin-right:6px;
}
.px-method-desc {
  font-size:12px;
  color:#94a3b8;
  line-height:1.45;
  min-height:54px;
}
.px-method-note {
  font-size:10px;
  color:#64748b;
  margin-top:6px;
}
.px-method-tech {
  font-size:11px;
  color:#7c8da8;
  line-height:1.4;
  margin-top:8px;
  padding-top:8px;
  border-top:0.5px solid #2a3347;
}
.px-tech-label {
  color:#bfdbfe;
  font-weight:600;
}
@media (max-width: 900px) {
  .px-xai-guide-grid { grid-template-columns:1fr; }
}
.px-figure-missing {
  height:220px;
  display:flex;
  align-items:center;
  justify-content:center;
  background:#1f2937;
  border:0.5px solid #374151;
  border-radius:8px;
  color:#94a3b8;
  font-size:13px;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def lung_svg(size: int = 34) -> str:
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M31.5 8v48" stroke="#bfdbfe" stroke-width="3" stroke-linecap="round"/>
  <path d="M28 28c-7-13-17-12-19-4-3 12 1 26 13 28 7 1 9-7 9-17 0-3-1-5-3-7Z" stroke="#bfdbfe" stroke-width="3" fill="none"/>
  <path d="M36 28c7-13 17-12 19-4 3 12-1 26-13 28-7 1-9-7-9-17 0-3 1-5 3-7Z" stroke="#bfdbfe" stroke-width="3" fill="none"/>
</svg>
"""


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
<div style="display:flex; gap:10px; align-items:center;">
  <div style="width:42px; height:42px; border-radius:10px; background:#0f2a5e; display:flex; align-items:center; justify-content:center;">
    {lung_svg(30)}
  </div>
  <div>
    <div style="font-size:14px; font-weight:500; color:#e2e8f0;">PneumoXAI</div>
    <div style="font-size:11px; color:#64748b;">Chest X-Ray Analyzer</div>
  </div>
</div>
<div class="px-divider"></div>
<div class="px-section-label">XAI METHODS</div>
            """,
            unsafe_allow_html=True,
        )
        st.toggle("Grad-CAM", value=True, key="show_gradcam")
        st.toggle("Grad-CAM++", value=True, key="show_gradcampp")
        st.toggle("Integrated Gradients", value=True, key="show_ig")
        st.markdown(
            """
<div class="px-divider"></div>
<div class="px-section-label">MODEL</div>
<div style="font-size:13px; color:#e2e8f0; margin-bottom:3px;">DenseNet121</div>
<div class="px-muted">densenet121-res224-all</div>
<div class="px-muted">Threshold: 0.731</div>
<div class="px-divider"></div>
<div class="px-section-label">LEGEND</div>
<div style="height:9px; border-radius:999px; background:linear-gradient(90deg,#2563eb,#06b6d4,#22c55e,#eab308,#ef4444); margin-top:8px;"></div>
<div style="display:flex; justify-content:space-between; color:#64748b; font-size:10px; margin-top:5px;">
  <span>Low</span><span>High</span>
</div>
<div class="px-divider"></div>
<div style="font-size:10px; color:#64748b;">Research use only. Not a clinical tool.</div>
            """,
            unsafe_allow_html=True,
        )


def metric_card(title: str, value: str, color: str, subtitle: str, width: int) -> str:
    return f"""
<div style="background:#161b27; border:0.5px solid #2a3347; border-radius:10px; padding:14px 16px;">
  <div style="font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">{title}</div>
  <div style="font-size:24px; font-weight:500; color:#e2e8f0; line-height:1;">{value}</div>
  <div style="height:3px; background:{color}; border-radius:2px; width:{width}%; margin-top:9px;"></div>
  <div style="font-size:10px; color:#64748b; margin-top:5px;">{subtitle}</div>
</div>
"""


def render_prediction(label: str, confidence: float, prob_pneumonia: float, prob_normal: float) -> None:
    color = "#ef4444" if label == "Pneumonia" else "#22c55e"
    progress_color = "#ef4444" if label == "Pneumonia" else "#22c55e"
    st.markdown(
        f"""
<style>
.stProgress > div > div > div > div {{ background-color:{progress_color}; }}
</style>
<div class="px-card">
  <div style="font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;">Prediction</div>
  <div style="font-size:30px; color:{color}; font-weight:600; line-height:1.1;">● {label}</div>
  <div style="font-size:13px; color:#94a3b8; margin-top:8px;">Confidence</div>
  <div style="font-size:28px; color:#e2e8f0; font-weight:500; margin-bottom:10px;">{confidence * 100:.1f}%</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(int(confidence * 100))
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            f"""<div class="px-prob-chip"><div class="px-muted">Normal prob</div><div style="font-size:20px; color:#e2e8f0;">{prob_normal * 100:.1f}%</div></div>""",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""<div class="px-prob-chip"><div class="px-muted">Pneumonia prob</div><div style="font-size:20px; color:#e2e8f0;">{prob_pneumonia * 100:.1f}%</div></div>""",
            unsafe_allow_html=True,
        )
    st.markdown(
        """<div class="px-warning">⚠️ Research tool only. Consult a qualified radiologist for clinical decisions.</div>""",
        unsafe_allow_html=True,
    )


def render_xai_grid(model, tensor, display_np) -> None:
    methods = []
    if st.session_state.get("show_gradcam", True):
        methods.append((
            "Grad-CAM",
            generate_gradcam,
            "Shows the main broad area the AI looked at before making its decision.",
            "Useful for a quick, big-picture check.",
            "Technical: uses gradients from the last convolutional layer to locate broad image regions linked to the prediction. Mean IoU: 0.225.",
        ))
    if st.session_state.get("show_gradcampp", True):
        methods.append((
            "Grad-CAM++",
            generate_gradcampp,
            "A sharper version that can highlight more than one suspicious region.",
            "Best localization score in this project.",
            "Technical: uses higher-order gradient weighting, which can separate multiple important regions better than standard Grad-CAM. Best IoU: 0.283.",
        ))
    if st.session_state.get("show_ig", True):
        methods.append((
            "Integrated Gradients",
            generate_ig,
            "Marks smaller image details that changed the AI's score most strongly.",
            "Often more speckled, so read it together with the other maps.",
            "Technical: compares the X-ray with a blank baseline and accumulates pixel-level attribution along that path. Mean IoU: 0.189.",
        ))

    if not methods:
        st.info("Enable at least one XAI method in the sidebar.")
        return

    overlays = []
    with st.spinner("Generating XAI overlays..."):
        for method_name, fn, description, note, tech_note in methods:
            try:
                overlays.append((method_name, fn(model, tensor, display_np), description, note, tech_note))
            except Exception as exc:
                st.warning(f"{method_name} failed: {exc}")

    if not overlays:
        return
    cols = st.columns(min(3, len(overlays)))
    for idx, (method_name, overlay, description, note, tech_note) in enumerate(overlays):
        with cols[idx % len(cols)]:
            st.image(overlay, use_container_width=True)
            st.markdown(
                f"""<div class="px-pill">{method_name}</div><div class="px-method-desc">{description}</div><div class="px-method-note">{note}</div><div class="px-method-tech"><span class="px-tech-label">Technical note:</span> {tech_note}</div>""",
                unsafe_allow_html=True,
            )


def render_confusion_matrix() -> None:
    st.markdown("**Confusion matrix**")
    st.markdown(
        """
<div style="display:grid; grid-template-columns:90px 1fr 1fr; gap:8px; align-items:stretch;">
  <div></div>
  <div style="color:#94a3b8; font-size:11px; text-align:center;">Pred Normal</div>
  <div style="color:#94a3b8; font-size:11px; text-align:center;">Pred Pneumonia</div>
  <div style="color:#94a3b8; font-size:11px; display:flex; align-items:center;">Normal</div>
  <div style="background:#1d4ed8; border-radius:8px; padding:18px; text-align:center;"><div style="font-size:28px; font-weight:600;">2683</div><div style="font-size:11px;">TN</div></div>
  <div style="background:#a16207; border-radius:8px; padding:18px; text-align:center;"><div style="font-size:28px; font-weight:600;">418</div><div style="font-size:11px;">FP</div></div>
  <div style="color:#94a3b8; font-size:11px; display:flex; align-items:center;">Pneumonia</div>
  <div style="background:#b91c1c; border-radius:8px; padding:18px; text-align:center;"><div style="font-size:28px; font-weight:600;">264</div><div style="font-size:11px;">FN</div></div>
  <div style="background:#15803d; border-radius:8px; padding:18px; text-align:center;"><div style="font-size:28px; font-weight:600;">638</div><div style="font-size:11px;">TP</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_xai_table() -> None:
    st.markdown("**XAI localization comparison**")
    st.markdown(
        """
<table style="width:100%; border-collapse:collapse; background:#161b27; border:0.5px solid #2a3347; border-radius:10px; overflow:hidden;">
  <thead>
    <tr style="color:#64748b; font-size:10px; text-transform:uppercase; letter-spacing:0.06em;">
      <th style="text-align:left; padding:11px;">Method</th><th style="text-align:left; padding:11px;">Mean IoU</th><th style="text-align:left; padding:11px;">Time</th><th style="text-align:left; padding:11px;">Consistency</th>
    </tr>
  </thead>
  <tbody style="font-size:13px; color:#e2e8f0;">
    <tr><td style="padding:11px; border-top:0.5px solid #2a3347;">Grad-CAM</td><td style="padding:11px; border-top:0.5px solid #2a3347;">0.225</td><td style="padding:11px; border-top:0.5px solid #2a3347;">456 ms</td><td style="padding:11px; border-top:0.5px solid #2a3347;">1.000</td></tr>
    <tr style="background:#0f2a5e;"><td style="padding:11px; border-top:0.5px solid #2a3347;">Grad-CAM++ <span style="background:#2563eb; border-radius:999px; padding:2px 7px; font-size:10px;">best</span></td><td style="padding:11px; border-top:0.5px solid #2a3347;"><b>0.283</b></td><td style="padding:11px; border-top:0.5px solid #2a3347;">538 ms</td><td style="padding:11px; border-top:0.5px solid #2a3347;">1.000</td></tr>
    <tr><td style="padding:11px; border-top:0.5px solid #2a3347;">Integr. Gradients</td><td style="padding:11px; border-top:0.5px solid #2a3347;">0.189</td><td style="padding:11px; border-top:0.5px solid #2a3347;">2255 ms</td><td style="padding:11px; border-top:0.5px solid #2a3347;">1.000</td></tr>
  </tbody>
</table>
        """,
        unsafe_allow_html=True,
    )


def render_figure(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.markdown("""<div class="px-figure-missing">Figure not available</div>""", unsafe_allow_html=True)
        st.caption(caption)


inject_global_css()
render_sidebar()

tab_predict, tab_perf = st.tabs(["🔬 Predict & Explain", "📊 Model Performance"])

with tab_predict:
    st.markdown(
        """
<div style="margin-bottom:14px;">
  <div style="font-size:28px; font-weight:600; color:#e2e8f0; line-height:1.15;">🫁 Chest X-Ray Pneumonia Analyzer</div>
  <div style="font-size:14px; color:#94a3b8; margin-top:6px;">Upload a frontal chest X-ray to classify and explain the model's decision.</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Drop chest X-ray here",
        type=["dcm", "png", "jpg", "jpeg"],
        help="Frontal view recommended. DICOM or standard image formats accepted."
    )

    if uploaded_file is None:
        st.markdown(
            f"""<div class="px-placeholder">{lung_svg(82)}<div style="margin-top:14px;">Upload a chest X-ray to get started</div></div>""",
            unsafe_allow_html=True,
        )
    else:
        if not CHECKPOINT_PATH.exists():
            st.error("Model checkpoint not found. Please ensure outputs/ directory is present.")
            st.stop()
        try:
            model, threshold = get_model()
        except FileNotFoundError:
            st.error("Model checkpoint not found. Please ensure outputs/ directory is present.")
            st.stop()
        except Exception as exc:
            st.error(f"Model could not be loaded: {exc}")
            st.stop()

        try:
            tensor, display_np = preprocess_image(uploaded_file)
            label, confidence, prob_pneumonia, prob_normal = run_inference(model, tensor, threshold)
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")
            st.stop()

        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            st.markdown("**Input X-ray**")
            st.image(display_np, use_container_width=True)
            st.markdown(
                f"""<div class="px-file-meta">{uploaded_file.name} · {display_np.shape[1]} × {display_np.shape[0]}</div>""",
                unsafe_allow_html=True,
            )
        with right:
            render_prediction(label, confidence, prob_pneumonia, prob_normal)

        st.divider()
        st.markdown(
            """
<div style="font-size:20px; font-weight:600; color:#e2e8f0;">Explainability maps</div>
<div style="font-size:13px; color:#94a3b8; margin:3px 0 10px;">These heatmaps show which parts of the X-ray most influenced the AI's prediction.</div>
<div class="px-xai-guide">
  <div class="px-xai-guide-title">How to read these maps</div>
  <div class="px-xai-guide-grid">
    <div class="px-xai-guide-item"><span class="px-dot" style="background:#ef4444;"></span><b style="color:#e2e8f0;">Red / yellow</b> areas had the strongest influence on the AI's answer.</div>
    <div class="px-xai-guide-item"><span class="px-dot" style="background:#2563eb;"></span><b style="color:#e2e8f0;">Blue / dark</b> areas mattered less for this prediction.</div>
    <div class="px-xai-guide-item"><span class="px-dot" style="background:#22c55e;"></span>Compare all enabled maps. If several point to the same region, the explanation is more consistent.</div>
  </div>
  <div style="font-size:11px; color:#64748b; margin-top:10px;">Important: these maps show model attention, not a confirmed disease location. A radiologist must interpret the scan clinically.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        render_xai_grid(model, tensor, display_np)

with tab_perf:
    st.markdown(
        """
<div style="display:flex; align-items:center; gap:10px;">
  <div style="font-size:24px; font-weight:600; color:#e2e8f0;">Model performance</div>
  <div class="px-pill" style="margin-bottom:0;">DenseNet121-all</div>
</div>
<div style="font-size:13px; color:#94a3b8; margin:6px 0 18px;">Evaluated on RSNA validation set · 4,003 images</div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(4)
    metrics = [
        ("ACCURACY", "82.9%", "#3b82f6", "overall correct", 83),
        ("PRECISION", "60.8%", "#22c55e", "of predicted +ve", 61),
        ("RECALL", "68.3%", "#f97316", "of actual +ve", 68),
        ("AUC-ROC", "0.873", "#a855f7", "F1: 0.643", 87),
    ]
    for col, metric in zip(metric_cols, metrics):
        with col:
            st.markdown(metric_card(*metric), unsafe_allow_html=True)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    row2_left, row2_right = st.columns(2, gap="large")
    with row2_left:
        render_confusion_matrix()
    with row2_right:
        render_xai_table()

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown("**Example figures**")
    figures_dir = Path(__file__).parent / "outputs" / "figures"
    fig_cols = st.columns(3)
    figure_specs = [
        ("xai_comparison_pneumonia.png", "Pneumonia cases: Grad-CAM | Grad-CAM++ | Integrated Gradients"),
        ("xai_comparison_normal.png", "Normal cases: all three methods"),
        ("xai_failure_cases.png", "Failure cases: misclassified examples with XAI overlays"),
    ]
    for col, (filename, caption) in zip(fig_cols, figure_specs):
        with col:
            render_figure(figures_dir / filename, caption)
