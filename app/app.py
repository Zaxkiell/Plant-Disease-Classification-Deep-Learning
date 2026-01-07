import os
import yaml
import torch
import timm
import numpy as np
import cv2
import streamlit as st

from PIL import Image
from torchvision import datasets, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_last_conv_layer(model):
    for _, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise RuntimeError("No Conv2d layer found")


@st.cache_resource
def load_model_and_classes():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_path = os.path.join(cfg["dataset"]["root_dir"], cfg["dataset"]["val_dir"])
    ds = datasets.ImageFolder(val_path)
    classes = ds.classes

    model = timm.create_model(
        cfg["model"]["name"],
        pretrained=False,
        num_classes=len(classes)
    ).to(device)

    weights_path = os.path.join(cfg["output"]["model_dir"], "best_model.pth")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    target_layer = get_last_conv_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer])

    tfm = transforms.Compose([
        transforms.Resize((cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])),
        transforms.ToTensor(),
    ])

    return cfg, device, model, classes, cam, tfm


def pil_to_rgb_np(pil_img):
    img = np.array(pil_img.convert("RGB"))
    return img


def preprocess(pil_img, tfm, device):
    x = tfm(pil_img).unsqueeze(0).to(device)
    return x


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)


def main():
    st.set_page_config(page_title="Plant Disease Classification", layout="wide")
    st.title("Plant Disease Classification (EfficientNet + Grad-CAM)")

    cfg, device, model, classes, cam, tfm = load_model_and_classes()

    uploaded = st.file_uploader("Upload an image (leaf)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to start.")
        return

    pil_img = Image.open(uploaded)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(pil_img, use_container_width=True)

    x = preprocess(pil_img, tfm, device)

    with torch.no_grad():
        logits = model(x).cpu().numpy()[0]

    probs = softmax(logits)
    topk = 5
    top_idx = probs.argsort()[::-1][:topk]
    top_probs = probs[top_idx]

    pred_idx = int(top_idx[0])
    pred_label = classes[pred_idx]
    pred_conf = float(top_probs[0])

    grayscale_cam = cam(input_tensor=x)[0]

    img_np = pil_to_rgb_np(pil_img)
    img_np = cv2.resize(img_np, (cfg["dataset"]["image_size"], cfg["dataset"]["image_size"]))
    img_np = img_np.astype(np.float32) / 255.0

    cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    with col2:
        st.subheader("Prediction")
        st.write(f"**Top-1:** {pred_label}")
        st.write(f"**Confidence:** {pred_conf:.4f}")

        st.subheader("Top-5")
        for i, (idx, p) in enumerate(zip(top_idx, top_probs), start=1):
            st.write(f"{i}. {classes[int(idx)]} — {float(p):.4f}")

        st.subheader("Grad-CAM")
        st.image(cam_img, use_container_width=True)


if __name__ == "__main__":
    main()
