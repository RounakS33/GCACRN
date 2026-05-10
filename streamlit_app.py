import io as _io
import os
import sys
import argparse

import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ── project root on path ──────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import create_model  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_opt(ckpt_dir: str, exp_name: str, epoch: str, use_gpu: bool):
    opt = argparse.Namespace()
    opt.model = "GCACRN"
    opt.name = exp_name
    opt.checkpoints_dir = ckpt_dir
    opt.epoch = epoch
    opt.load_iter = 0
    opt.isTrain = False
    opt.gpu_ids = [0] if (use_gpu and torch.cuda.is_available()) else []
    opt.input_nc = 3
    opt.output_nc = 3
    opt.ngf = 64
    opt.ndf = 64
    opt.netG = "gen_drop"
    opt.netD = "basic"
    opt.n_layers_D = 3
    opt.norm = "instance"
    opt.init_type = "normal"
    opt.init_gain = 0.02
    opt.no_dropout = False
    opt.preprocess = "resize_and_crop"
    opt.verbose = False
    opt.batch_size = 1
    opt.blurKernel = 5
    return opt


@st.cache_resource(show_spinner="Loading model weights…")
def _load_model(ckpt_dir: str, exp_name: str, epoch: str, use_gpu: bool):
    opt = _make_opt(ckpt_dir, exp_name, epoch, use_gpu)
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    return model


def _preprocess(img: Image.Image) -> torch.Tensor:
    """PIL → [1, 3, 256, 256] float tensor in [0, 1]."""
    return T.Compose([T.Resize((256, 256)), T.ToTensor()])(
        img.convert("RGB")
    ).unsqueeze(0)


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] tensor → [H, W, C] uint8."""
    return (t.permute(1, 2, 0).cpu().float().numpy().clip(0, 1) * 255).astype(np.uint8)


def _run_inference(model, img_tensor: torch.Tensor, n_cascade: int = 3) -> list[dict]:
    dev = model.device
    model.real_I = img_tensor.to(dev)

    # replicate model.init()
    model.t_h = model.t_c = model.r_h = model.r_c = None
    model.fake_T = model.real_I.clone()
    model.fake_Ts = [model.fake_T]
    model.fake_R = torch.ones_like(model.real_I) * 0.1
    model.fake_Rs = [model.fake_R]

    steps: list[dict] = []

    with torch.no_grad():
        for _ in range(n_cascade):
            inp_T = torch.cat(
                [model.real_I, model.fake_Ts[-1], model.fake_Rs[-1]], dim=1
            )
            fT, model.t_h, model.t_c, fT2, fT4 = model.netG_T(
                inp_T, model.t_h, model.t_c
            )
            model.fake_T = fT
            model.fake_T2 = fT2
            model.fake_T4 = fT4
            model.fake_Ts.append(fT)

            inp_R = torch.cat(
                [model.real_I, model.fake_Ts[-1], model.fake_Rs[-1]], dim=1
            )
            fR, model.r_h, model.r_c, fR2, fR4 = model.netG_R(
                inp_R, model.r_h, model.r_c
            )
            model.fake_R = fR
            model.fake_R2 = fR2
            model.fake_R4 = fR4
            model.fake_Rs.append(fR)

            steps.append(
                {
                    "T_256": fT.clamp(0, 1).squeeze(0).cpu(),
                    "T_128": fT2.clamp(0, 1).squeeze(0).cpu(),
                    "T_64": fT4.clamp(0, 1).squeeze(0).cpu(),
                    "R_256": fR.clamp(0, 1).squeeze(0).cpu(),
                    "R_128": fR2.clamp(0, 1).squeeze(0).cpu(),
                    "R_64": fR4.clamp(0, 1).squeeze(0).cpu(),
                }
            )

    return steps


def _compute_metrics(pred: torch.Tensor, gt: torch.Tensor, dev):
    """pred / gt: [3, H, W] tensors. Returns (ssim, psnr, lpips)."""
    p = pred.unsqueeze(0).to(dev)
    g = gt.unsqueeze(0).to(dev)
    ssim = StructuralSimilarityIndexMeasure(
        data_range=1.0).to(dev)(p, g).item()
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev)(p, g).item()
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", normalize=True).to(dev)(p, g).item()
    return ssim, psnr, lpips


def _compute_losses(model, gt_tensor: torch.Tensor) -> dict:
    """Run compute_losses() given a GT transmission tensor [1,3,256,256]."""
    dev = model.device
    model.real_T = gt_tensor.to(dev)
    model.real_T2 = T.Resize((128, 128))(gt_tensor).to(dev)
    model.real_T4 = T.Resize((64, 64))(gt_tensor).to(dev)
    model.isNatural = True
    with torch.no_grad():
        model.compute_losses()
    return model.get_current_losses()


# ─────────────────────────────────────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Reflection Removal Using GCACRN",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("GCACRN — Reflection Removal Visualizer")
st.caption(
    "Upload a reflection-corrupted image. The model runs specified number of cascade iterations "
    "with both pretrained Transmission (T) and Reflection (R) generators. All intermediate "
    "multi-scale outputs are shown."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Configuration")

    ckpt_dir = st.text_input("Checkpoints dir", "./checkpoints")

    _avail: list[str] = []
    if os.path.isdir(ckpt_dir):
        _avail = sorted(
            d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))
        )
    if not _avail:
        _avail = ["GCACRN"]

    exp_name = st.selectbox("Experiment", _avail)
    epoch = st.text_input("Epoch suffix", "latest")
    n_cascade = st.number_input(
        "Cascade stages", min_value=1, max_value=10, value=3, step=1)

    # ADD THIS NEW SLIDER
    alpha_blend = st.slider(
        "Alpha Blending Factor (α)",
        min_value=0.5, max_value=1.0, value=1.0, step=0.05,
        help="Adjusts the transmission attenuation: Î = αT + R. Set to 1.0 for real images, or ~0.8 for synthetic."
    )

    use_gpu = st.checkbox("Use GPU", value=torch.cuda.is_available())

    st.divider()
    st.header("Image Uploads")
    up_I = st.file_uploader("Reflection-corrupted image (I)",
                            type=["png", "jpg", "jpeg", "bmp", "tiff"])
    up_T = st.file_uploader(
        "Ground-truth transmission (T) [Optional]", type=["png", "jpg", "jpeg", "bmp", "tiff"])
    up_R = st.file_uploader(
        "Ground-truth reflection (R) [Optional]", type=["png", "jpg", "jpeg", "bmp", "tiff"])

# ── Gate on upload ─────────────────────────────────────────────────────────────
if up_I is None:
    st.info("Upload a reflection-corrupted image using the sidebar to begin.")
    st.stop()

img_I = Image.open(up_I).convert("RGB")
img_T_gt = Image.open(up_T).convert("RGB") if up_T else None
img_R_gt = Image.open(up_R).convert("RGB") if up_R else None

has_gt = img_T_gt is not None
has_r_gt = img_R_gt is not None

# ── Upload preview + run button ───────────────────────────────────────────────
col_prev1, col_prev2, col_run = st.columns([1, 1, 1])
with col_prev1:
    st.subheader("Input (I)")
    st.image(img_I, use_container_width=True)
with col_prev2:
    st.subheader("Ground Truths Provided:")
    if has_gt:
        st.write("✅ Transmission (T)")
    else:
        st.write("❌ Transmission (T)")
    if has_r_gt:
        st.write("✅ Reflection (R)")
    else:
        st.write("❌ Reflection (R)")
with col_run:
    st.subheader("Run")
    dev_label = "GPU (CUDA)" if (
        use_gpu and torch.cuda.is_available()) else "CPU"
    st.write(f"**Device:** `{dev_label}`")
    run_btn = st.button("Run Inference", type="primary",
                        use_container_width=True)

# ── Load model ────────────────────────────────────────────────────────────────
try:
    model = _load_model(ckpt_dir, exp_name, epoch, use_gpu)
except Exception as exc:
    st.error(f"**Failed to load model:** {exc}")
    st.stop()

# ── Inference ─────────────────────────────────────────────────────────────────
_key = f"steps_{exp_name}_{epoch}_{n_cascade}_{up_I.name}"
if run_btn or _key not in st.session_state:
    tensor_I = _preprocess(img_I)
    with st.spinner(f"Running {n_cascade}-stage cascade inference…"):
        steps = _run_inference(model, tensor_I, n_cascade=int(n_cascade))
    st.session_state[_key] = steps
    st.session_state["tensor_I"] = tensor_I
    st.session_state["tensor_T_gt"] = _preprocess(img_T_gt) if has_gt else None
    st.session_state["tensor_R_gt"] = _preprocess(
        img_R_gt) if has_r_gt else None
elif _key in st.session_state:
    steps = st.session_state[_key]
else:
    st.info("Press **Run Inference** to start.")
    st.stop()

tensor_I: torch.Tensor = st.session_state["tensor_I"]
tensor_T_gt: torch.Tensor | None = st.session_state["tensor_T_gt"]
tensor_R_gt: torch.Tensor | None = st.session_state.get("tensor_R_gt")

init_T_np = _to_uint8(tensor_I.squeeze(0))
init_R_np = np.full((256, 256, 3), 26, dtype=np.uint8)  # 0.1 × 255 ≈ 26

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Cascade Progression (Horizontal Grid)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("1 · Cascade Progression")
st.caption(
    "Top row: Transmission (T) progression. Bottom row: Reflection (R) progression.")

n_stages = len(steps)

# Determine if we need an extra column for Ground Truths at the end
show_gt_col = has_gt or has_r_gt
n_cols = n_stages + 2 if show_gt_col else n_stages + 1

fig_casc, axes_casc = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))

# Prepare base lists (Init + All Cascade Stages)
t_labels = ["Init: $T_0$ (Input I)"] + \
    [f"Iter {i+1}: $\hat{{T}}_{i+1}$" for i in range(n_stages)]
t_imgs = [init_T_np] + [_to_uint8(s["T_256"]) for s in steps]

r_labels = ["Init: $R_0$ (Constant)"] + \
    [f"Iter {i+1}: $\hat{{R}}_{i+1}$" for i in range(n_stages)]
r_imgs = [init_R_np] + [_to_uint8(s["R_256"]) for s in steps]

# Handle the final "Target" column if applicable
if show_gt_col:
    # --- Transmission GT Row ---
    if has_gt and tensor_T_gt is not None:
        t_labels.append("Target: $T$ (Ground Truth)")
        t_imgs.append(_to_uint8(tensor_T_gt.squeeze(0)))
    else:
        # If no T ground truth is provided but R is, just repeat the final estimate to keep grid square
        t_labels.append("$\hat{T}$ (Final Estimate)")
        t_imgs.append(_to_uint8(steps[-1]["T_256"]))

    # --- Reflection GT Row ---
    if has_r_gt and tensor_R_gt is not None:
        # Scenario A: User uploaded the actual Ground Truth Reflection
        r_labels.append("Target: $R$ (Ground Truth)")
        r_imgs.append(_to_uint8(tensor_R_gt.squeeze(0)))
    elif has_gt and tensor_T_gt is not None:
        # Scenario B: No GT Reflection, but we have GT Transmission (Calculate Proxy)
        r_labels.append("Target: $I - T$ (Proxy GT)")
        gt_r = (tensor_I.squeeze(0) - tensor_T_gt.squeeze(0)).clamp(0, 1)
        r_imgs.append(_to_uint8(gt_r))
    else:
        # Scenario C: Just repeat the final estimate
        r_labels.append("$\hat{R}$ (Final Estimate)")
        r_imgs.append(_to_uint8(steps[-1]["R_256"]))

# Plot the grid
for c in range(n_cols):
    # Top Row: Transmission
    axes_casc[0, c].imshow(t_imgs[c])
    axes_casc[0, c].set_title(t_labels[c], fontsize=15,
                              fontweight="bold", pad=12)
    axes_casc[0, c].axis("off")

    # Bottom Row: Reflection
    axes_casc[1, c].imshow(r_imgs[c])
    axes_casc[1, c].set_title(r_labels[c], fontsize=15,
                              fontweight="bold", pad=12)
    axes_casc[1, c].axis("off")

fig_casc.tight_layout()
st.pyplot(fig_casc, use_container_width=True)
plt.close(fig_casc)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Multi-Scale Intermediates (per-iteration)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("2 · Multi-Scale Generator Outputs")
st.caption("Each generator produces outputs at three resolutions (256, 128, 64). Expand to view the intermediate feature maps packed into RGB channels.")

_MS_KEYS = ["T_256", "T_128", "T_64", "R_256", "R_128", "R_64"]
_MS_LABELS = ["T · 256×256", "T · 128×128", "T · 64×64",
              "R · 256×256", "R · 128×128", "R · 64×64"]

for i, step in enumerate(steps):
    # Using st.expander to make them collapsible. We keep the final iteration expanded by default.
    with st.expander(f"Iteration {i + 1} Multi-Scale Features", expanded=(i == len(steps) - 1)):
        cols = st.columns(6)
        for col, key, label in zip(cols, _MS_KEYS, _MS_LABELS):
            with col:
                img_show = _to_uint8(step[key])
                st.image(img_show, caption=label, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Layer Decomposition & Reconstruction Residual
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("3 · Layer Decomposition & Reconstruction Residual")
st.caption("Visualizing the decoupling accuracy at each step. The residual (I - T̂ - R̂) highlights unaccounted signals (grey=0 error).")

inp_np = _to_uint8(tensor_I.squeeze(0))
for i, step in enumerate(steps):
    t_np = _to_uint8(step["T_256"])
    r_np = _to_uint8(step["R_256"])
    resid = tensor_I.squeeze(0) - step["T_256"] - step["R_256"]
    resid_display = ((resid.permute(1, 2, 0).numpy() + 1.0) /
                     2.0 * 255).astype(np.uint8)

    with st.expander(f"Iteration {i + 1} Decomposition", expanded=(i == len(steps) - 1)):
        c1, c2, c3, c4 = st.columns(4)
        c1.image(inp_np, caption="Input (I)", use_container_width=True)
        c2.image(t_np, caption=f"T̂  (iter {i+1})", use_container_width=True)
        c3.image(r_np, caption=f"R̂  (iter {i+1})", use_container_width=True)
        c4.image(resid_display, caption="Residual I−T̂−R̂",
                 use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Final output & Reconstruction
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("4 · Final Output & Network Synergy")

final_T = steps[-1]["T_256"]
final_R = steps[-1]["R_256"]
# reconstructed_I = (final_T + final_R).clamp(0, 1)
# FACTOR ALPHA INTO THE FINAL RECONSTRUCTION HERE
reconstructed_I = ((alpha_blend * final_T) + final_R).clamp(0, 1)

# --- Row 1: Transmission ---
st.subheader("Transmission")
cols_T = st.columns(4)
cols_T[0].image(inp_np, caption="Input (I)", use_container_width=True)

if has_gt and tensor_T_gt is not None:
    gt_cpu_T = tensor_T_gt.squeeze(0).cpu()
    err_T = (final_T - gt_cpu_T).abs().mean(0).numpy()
    err_norm_T = (err_T - err_T.min()) / (err_T.max() - err_T.min() + 1e-8)
    err_color_T = (cm.hot(err_norm_T)[:, :, :3] * 255).astype(np.uint8)

    cols_T[1].image(_to_uint8(gt_cpu_T),
                    caption="Ground Truth T", use_container_width=True)
    cols_T[2].image(_to_uint8(final_T), caption="Predicted T",
                    use_container_width=True)
    cols_T[3].image(err_color_T, caption="Pixel Error Map (T)",
                    use_container_width=True)
else:
    cols_T[1].image(_to_uint8(final_T), caption="Predicted T",
                    use_container_width=True)

# --- Row 2: Reflection & Synergy ---
st.subheader("Reflection & Synergy")
cols_R = st.columns(4)
cols_R[0].image(_to_uint8(reconstructed_I),
                caption="Reconstructed Î (T + R)", use_container_width=True)

if has_r_gt and tensor_R_gt is not None:
    gt_cpu_R = tensor_R_gt.squeeze(0).cpu()
    err_R = (final_R - gt_cpu_R).abs().mean(0).numpy()
    err_norm_R = (err_R - err_R.min()) / (err_R.max() - err_R.min() + 1e-8)
    err_color_R = (cm.hot(err_norm_R)[:, :, :3] * 255).astype(np.uint8)

    cols_R[1].image(_to_uint8(gt_cpu_R),
                    caption="Ground Truth R", use_container_width=True)
    cols_R[2].image(_to_uint8(final_R), caption="Predicted R",
                    use_container_width=True)
    cols_R[3].image(err_color_R, caption="Pixel Error Map (R)",
                    use_container_width=True)
else:
    cols_R[1].image(_to_uint8(final_R), caption="Predicted R",
                    use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Quantitative Metrics (per-iteration + final)
# ─────────────────────────────────────────────────────────────────────────────
if has_gt or has_r_gt:
    st.divider()
    st.header("5 · Quantitative Metrics")
    dev = model.device
    iters = list(range(1, len(steps) + 1))

    fig_m, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

    if has_gt and tensor_T_gt is not None:
        gt_cpu_T = tensor_T_gt.squeeze(0).cpu()
        iter_ssim_T, iter_psnr_T, iter_lpips_T = [], [], []
        for step in steps:
            s, p, l = _compute_metrics(step["T_256"], gt_cpu_T, dev)
            iter_ssim_T.append(s)
            iter_psnr_T.append(p)
            iter_lpips_T.append(l)

        ax1.plot(iters, iter_ssim_T, "o-",
                 color="#4C9BE8", label="Transmission")
        ax2.plot(iters, iter_psnr_T, "o-",
                 color="#5DB85D", label="Transmission")
        ax3.plot(iters, iter_lpips_T, "o-",
                 color="#E8784C", label="Transmission")

        st.subheader("Transmission Final Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("SSIM ↑", f"{iter_ssim_T[-1]:.4f}")
        m2.metric("PSNR ↑", f"{iter_psnr_T[-1]:.2f} dB")
        m3.metric("LPIPS ↓", f"{iter_lpips_T[-1]:.4f}", delta_color="inverse")

    if has_r_gt and tensor_R_gt is not None:
        gt_cpu_R = tensor_R_gt.squeeze(0).cpu()
        iter_ssim_R, iter_psnr_R, iter_lpips_R = [], [], []
        for step in steps:
            s, p, l = _compute_metrics(step["R_256"], gt_cpu_R, dev)
            iter_ssim_R.append(s)
            iter_psnr_R.append(p)
            iter_lpips_R.append(l)

        ax1.plot(iters, iter_ssim_R, "x--", color="#4C9BE8",
                 alpha=0.6, label="Reflection")
        ax2.plot(iters, iter_psnr_R, "x--", color="#5DB85D",
                 alpha=0.6, label="Reflection")
        ax3.plot(iters, iter_lpips_R, "x--", color="#E8784C",
                 alpha=0.6, label="Reflection")

        st.subheader("Reflection Final Metrics")
        m4, m5, m6 = st.columns(3)
        m4.metric("SSIM ↑", f"{iter_ssim_R[-1]:.4f}")
        m5.metric("PSNR ↑", f"{iter_psnr_R[-1]:.2f} dB")
        m6.metric("LPIPS ↓", f"{iter_lpips_R[-1]:.4f}", delta_color="inverse")

    ax1.set_title("SSIM ↑")
    ax1.set_xlabel("Cascade iteration")
    ax1.legend()
    ax2.set_title("PSNR ↑ (dB)")
    ax2.set_xlabel("Cascade iteration")
    ax2.legend()
    ax3.set_title("LPIPS ↓")
    ax3.set_xlabel("Cascade iteration")
    ax3.legend()

    for ax in (ax1, ax2, ax3):
        ax.grid(True, alpha=0.3)
        ax.set_xticks(iters)

    fig_m.tight_layout()
    st.pyplot(fig_m, use_container_width=True)
    plt.close(fig_m)

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Loss values (requires GT Transmission)
# ─────────────────────────────────────────────────────────────────────────────
if has_gt and tensor_T_gt is not None:
    st.divider()
    st.header("6 · Loss Decomposition")
    st.caption(
        "Computed in eval mode with isNatural=True. "
        "Discriminator (D_syn) and adversarial (G) losses are 0 during inference."
    )

    with st.spinner("Computing losses…"):
        losses = _compute_losses(model, tensor_T_gt)

    import pandas as pd

    df_loss = pd.DataFrame(
        [(k, v) for k, v in losses.items()],
        columns=["Component", "Value"],
    )
    df_loss["Value"] = df_loss["Value"].map(lambda x: f"{x:.6f}")

    st.dataframe(df_loss, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Download final outputs
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("7 · Download")


def _to_png_bytes(t: torch.Tensor) -> bytes:
    buf = _io.BytesIO()
    Image.fromarray(_to_uint8(t)).save(buf, format="PNG")
    return buf.getvalue()


dc1, dc2 = st.columns(2)
with dc1:
    st.download_button(
        "Download predicted T (PNG)",
        data=_to_png_bytes(final_T),
        file_name="predicted_transmission.png",
        mime="image/png",
        use_container_width=True,
    )
with dc2:
    st.download_button(
        "Download predicted R (PNG)",
        data=_to_png_bytes(final_R),
        file_name="predicted_reflection.png",
        mime="image/png",
        use_container_width=True,
    )
