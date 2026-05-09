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
    """
    Replicates model.forward() step-by-step and captures full-res, half-res,
    and quarter-res outputs for BOTH generators at every cascade iteration.

    Returns a list of n_cascade dicts, one per iteration:
        { 'T_256', 'T_128', 'T_64', 'R_256', 'R_128', 'R_64' }
    All tensors are [3, H, W], clamped to [0, 1], on CPU.
    """
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
        net_type="vgg", normalize=True
    ).to(dev)(p, g).item()
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

st.title("GCACRN — Reflection Removal Cascade Visualizer")
st.caption(
    "Upload a reflection-corrupted image. The model runs specified number of cascade iterations "
    "with both pretrained Transmission (T) and Reflection (R) generators. All intermediate "
    "multi-scale outputs are shown."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model")

    ckpt_dir = st.text_input("Checkpoints dir", "./checkpoints")

    _avail: list[str] = []
    if os.path.isdir(ckpt_dir):
        _avail = sorted(
            d
            for d in os.listdir(ckpt_dir)
            if os.path.isdir(os.path.join(ckpt_dir, d))
        )
    if not _avail:
        _avail = ["GCACRN"]

    exp_name = st.selectbox("Experiment", _avail)
    epoch = st.text_input("Epoch suffix", "latest")
    n_cascade = st.number_input(
        "Cascade stages", min_value=1, max_value=10, value=3, step=1,
        help="Number of T/R generator iterations. Display shows n+1 images (init + n stages).",
    )
    use_gpu = st.checkbox("Use GPU", value=torch.cuda.is_available())

    st.divider()
    st.header("Images")
    up_I = st.file_uploader(
        "Reflection-corrupted image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
    )
    up_T = st.file_uploader(
        "Ground-truth transmission",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
    )

# ── Gate on upload ─────────────────────────────────────────────────────────────
if up_I is None:
    st.info("Upload a reflection-corrupted image using the sidebar to begin.")
    st.stop()

img_I = Image.open(up_I).convert("RGB")
img_T_gt = Image.open(up_T).convert("RGB") if up_T else None
has_gt = img_T_gt is not None

# ── Upload preview + run button ───────────────────────────────────────────────
col_prev1, col_prev2, col_run = st.columns([1, 1, 1])
with col_prev1:
    st.subheader("Input (I)")
    st.image(img_I, use_container_width=True)
with col_prev2:
    st.subheader("Transmission (T)" if has_gt else "GT Transmission")
    if has_gt:
        st.image(img_T_gt, use_container_width=True)
    else:
        st.caption("Not provided — metrics and losses will be skipped.")
with col_run:
    st.subheader("Configuration")
    st.write(f"**Experiment:** `{exp_name}`")
    st.write(f"**Epoch:** `{epoch}`")
    dev_label = "GPU (CUDA)" if (
        use_gpu and torch.cuda.is_available()) else "CPU"
    st.write(f"**Device:** `{dev_label}`")
    st.write("")
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
elif _key in st.session_state:
    steps = st.session_state[_key]
else:
    st.info("Press **Run Inference** to start.")
    st.stop()

tensor_I: torch.Tensor = st.session_state["tensor_I"]
tensor_T_gt: torch.Tensor | None = st.session_state["tensor_T_gt"]
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
# Total columns = Init + all cascade stages + Ground Truth (if provided)
n_cols = n_stages + 2 if has_gt else n_stages + 1

fig_casc, axes_casc = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))

# Prepare Transmission (T) images and labels
t_labels = ["$T_0$"] + [f"$\hat{{T}}_{i+1}$" for i in range(n_stages)]
t_imgs = [init_T_np] + [_to_uint8(s["T_256"]) for s in steps]

# Prepare Reflection (R) images and labels
r_labels = ["$R_0$"] + [f"$\hat{{R}}_{i+1}$" for i in range(n_stages)]
r_imgs = [init_R_np] + [_to_uint8(s["R_256"]) for s in steps]

# Append Ground Truths to the end if available
if has_gt and tensor_T_gt is not None:
    t_labels.append("$T$ (GT)")
    t_imgs.append(_to_uint8(tensor_T_gt.squeeze(0)))

    r_labels.append("$\hat{R}$ (GT)")
    # Approximate GT reflection as I - T (clamped to prevent artifacting)
    gt_r = (tensor_I.squeeze(0) - tensor_T_gt.squeeze(0)).clamp(0, 1)
    r_imgs.append(_to_uint8(gt_r))

# Plot the grid
for c in range(n_cols):
    # Top Row: Transmission
    axes_casc[0, c].imshow(t_imgs[c])
    axes_casc[0, c].set_title(t_labels[c], fontsize=16, pad=12)
    axes_casc[0, c].axis("off")

    # Bottom Row: Reflection
    axes_casc[1, c].imshow(r_imgs[c])
    axes_casc[1, c].set_title(r_labels[c], fontsize=16, pad=12)
    axes_casc[1, c].axis("off")

fig_casc.tight_layout()
# Change to use_column_width=True if your Streamlit is outdated
st.pyplot(fig_casc, use_container_width=True)
plt.close(fig_casc)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Multi-Scale Intermediates (per-iteration)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("2 · Multi-Scale Generator Outputs")
st.caption(
    "Each generator produces outputs at three resolutions (256, 128, 64) "
    "used for hierarchical perceptual losses. Shown here for every iteration."
)

_MS_KEYS = ["T_256", "T_128", "T_64", "R_256", "R_128", "R_64"]
_MS_LABELS = ["T · 256×256", "T · 128×128", "T · 64×64",
              "R · 256×256", "R · 128×128", "R · 64×64"]

for i, step in enumerate(steps):
    st.subheader(f"Iteration {i + 1}")
    cols = st.columns(6)
    for col, key, label in zip(cols, _MS_KEYS, _MS_LABELS):
        with col:
            img_show = _to_uint8(step[key])
            st.image(img_show, caption=label, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — T vs R comparison across iterations (delta view)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("3 · T + R + Residual  (per iteration)")
st.caption(
    "At each iteration: predicted T, predicted R, and the residual "
    "I - T - R (highlights what the model hasn't yet explained)."
)

inp_np = _to_uint8(tensor_I.squeeze(0))
for i, step in enumerate(steps):
    t_np = _to_uint8(step["T_256"])
    r_np = _to_uint8(step["R_256"])
    # Residual in float before clamping so we can see the signed error
    resid = tensor_I.squeeze(0) - step["T_256"] - step["R_256"]
    resid_display = ((resid.permute(1, 2, 0).numpy() + 1.0) / 2.0 * 255).astype(
        np.uint8
    )

    with st.expander(f"Iteration {i + 1}", expanded=(i == len(steps) - 1)):
        c1, c2, c3, c4 = st.columns(4)
        c1.image(inp_np, caption="Input (I)", use_container_width=True)
        c2.image(t_np, caption=f"T̂  (iter {i+1})", use_container_width=True)
        c3.image(r_np, caption=f"R̂  (iter {i+1})", use_container_width=True)
        c4.image(
            resid_display, caption="Residual I−T̂−R̂  (grey=0)", use_container_width=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Final output (+ GT comparison if available)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("4 · Final Output")

final_T = steps[-1]["T_256"]
final_R = steps[-1]["R_256"]
inp_np = _to_uint8(tensor_I.squeeze(0))  # Ensure the input array is ready

if has_gt and tensor_T_gt is not None:
    gt_cpu = tensor_T_gt.squeeze(0).cpu()

    # Calculate error map
    err = (final_T - gt_cpu).abs().mean(0).numpy()
    err_norm = (err - err.min()) / (err.max() - err.min() + 1e-8)
    err_color = (cm.hot(err_norm)[:, :, :3] * 255).astype(np.uint8)

    # 5-column layout to include the Original Input
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.image(inp_np, caption="Input (I)", use_container_width=True)
    c2.image(_to_uint8(gt_cpu), caption="Ground TruthTransmission (T)",
             use_container_width=True)
    c3.image(_to_uint8(final_T), caption="Predicted T (final)",
             use_container_width=True)
    c4.image(_to_uint8(final_R), caption="Predicted R (final)",
             use_container_width=True)
    c5.image(err_color, caption="Pixel error map (T)",
             use_container_width=True)
else:
    # 3-column layout if no Ground Truth is provided
    c1, c2, c3 = st.columns(3)
    c1.image(inp_np, caption="Input (I)", use_container_width=True)
    c2.image(_to_uint8(final_T), caption="Predicted T (final)",
             use_container_width=True)
    c3.image(_to_uint8(final_R), caption="Predicted R (final)",
             use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Metrics (per-iteration + final)
# ─────────────────────────────────────────────────────────────────────────────
if has_gt and tensor_T_gt is not None:
    st.divider()
    st.header("5 · Metrics  (vs GT Transmission)")

    gt_cpu = tensor_T_gt.squeeze(0).cpu()
    dev = model.device

    iter_ssim, iter_psnr, iter_lpips = [], [], []
    with st.spinner("Computing per-iteration SSIM / PSNR / LPIPS…"):
        for step in steps:
            s, p, l = _compute_metrics(step["T_256"], gt_cpu, dev)
            iter_ssim.append(s)
            iter_psnr.append(p)
            iter_lpips.append(l)

    # Summary cards for the final iteration
    m1, m2, m3 = st.columns(3)
    delta_ssim = iter_ssim[-1] - iter_ssim[0] if len(iter_ssim) > 1 else None
    delta_psnr = iter_psnr[-1] - iter_psnr[0] if len(iter_psnr) > 1 else None
    delta_lpips = iter_lpips[-1] - \
        iter_lpips[0] if len(iter_lpips) > 1 else None
    m1.metric("SSIM ↑  (final)",  f"{iter_ssim[-1]:.4f}",
              delta=f"{delta_ssim:+.4f} vs iter 1" if delta_ssim is not None else None)
    m2.metric("PSNR ↑  (final)",  f"{iter_psnr[-1]:.2f} dB",
              delta=f"{delta_psnr:+.2f} dB vs iter 1" if delta_psnr is not None else None)
    m3.metric("LPIPS ↓  (final)", f"{iter_lpips[-1]:.4f}",
              delta=f"{delta_lpips:+.4f} vs iter 1" if delta_lpips is not None else None,
              delta_color="inverse")

    # Per-iteration trend charts
    iters = list(range(1, len(steps) + 1))
    fig_m, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

    ax1.plot(iters, iter_ssim, "o-", color="#4C9BE8")
    ax1.set_title("SSIM ↑")
    ax1.set_xlabel("Cascade iteration")
    ax1.set_xticks(iters)
    ax1.set_ylim(max(0, min(iter_ssim) - 0.02), min(1, max(iter_ssim) + 0.02))

    ax2.plot(iters, iter_psnr, "o-", color="#5DB85D")
    ax2.set_title("PSNR ↑  (dB)")
    ax2.set_xlabel("Cascade iteration")
    ax2.set_xticks(iters)

    ax3.plot(iters, iter_lpips, "o-", color="#E8784C")
    ax3.set_title("LPIPS ↓")
    ax3.set_xlabel("Cascade iteration")
    ax3.set_xticks(iters)

    for ax in (ax1, ax2, ax3):
        ax.grid(True, alpha=0.3)
    fig_m.tight_layout()
    st.pyplot(fig_m, use_container_width=True)
    plt.close(fig_m)

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Loss values (requires GT)
# ─────────────────────────────────────────────────────────────────────────────
if has_gt and tensor_T_gt is not None:
    st.divider()
    st.header("6 · Loss Decomposition  (final iteration vs GT)")
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

    # Display only the table, spanning the center of the screen
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
