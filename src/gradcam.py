import os
import yaml
import torch
import timm
import numpy as np
import cv2

from torchvision import datasets, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_last_conv_layer(model):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise RuntimeError("No Conv2d layer found")


def main():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root_dir = cfg["dataset"]["root_dir"]
    val_dir = cfg["dataset"]["val_dir"]
    img_size = cfg["dataset"]["image_size"]

    val_path = os.path.join(root_dir, val_dir)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    ds = datasets.ImageFolder(val_path, transform=tfm)
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

    os.makedirs(cfg["output"]["plot_dir"], exist_ok=True)

    idx = np.random.randint(0, len(ds))
    img_tensor, label = ds[idx]

    img = img_tensor.unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=img)[0]

    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    cam_img = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)
    cam_img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        logits = model(img)
        pred = int(torch.argmax(logits, dim=1).item())

    out_path = os.path.join(cfg["output"]["plot_dir"], f"gradcam_sample_idx{idx}.png")
    cv2.imwrite(out_path, cam_img_bgr)

    print("Saved:", out_path)
    print("GT:", classes[label])
    print("Pred:", classes[pred])


if __name__ == "__main__":
    main()
