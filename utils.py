import os
import torch
import torchvision.transforms as transforms
from PIL import Image

EN_US = os.getenv("LANG") != "zh_CN.UTF-8"

ZH2EN = {
    "上传录音": "Upload a recording",
    "选择模型": "Select a model",
    "状态栏": "Status",
    "音频文件名": "Audio filename",
    "中国乐器识别": "Chinese instrument recognition",
    "建议录音时长保持在 3s 左右": "It is recommended to keep the recording length around 3s.",
    "引用": "Cite",
}

if EN_US:
    import huggingface_hub

    MODEL_DIR = huggingface_hub.snapshot_download(
        "ccmusic-database/CTIS",
        cache_dir="./__pycache__",
    )

else:
    import modelscope

    MODEL_DIR = modelscope.snapshot_download(
        "ccmusic-database/CTIS",
        cache_dir="./__pycache__",
    )


def _L(zh_txt: str):
    return ZH2EN[zh_txt] if EN_US else zh_txt


def toCUDA(x):
    if hasattr(x, "cuda"):
        if torch.cuda.is_available():
            return x.cuda()

    return x


def find_files(folder_path=f"{MODEL_DIR}/examples", ext=".wav"):
    wav_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(ext):
                file_path = os.path.join(root, file)
                wav_files.append(file_path)

    return wav_files


def get_modelist(model_dir=MODEL_DIR, assign_model=""):
    output = []
    for entry in os.listdir(model_dir):
        # 获取完整路径
        full_path = os.path.join(model_dir, entry)
        # 跳过'.git'文件夹
        if entry == ".git" or entry == "examples":
            print(f"跳过 .git 或 examples 文件夹: {full_path}")
            continue

        # 检查条目是文件还是目录
        if os.path.isdir(full_path):
            model = os.path.basename(full_path)
            if assign_model and assign_model.lower() in model:
                output.insert(0, model)
            else:
                output.append(model)

    return output


def embed_img(img_path: str, input_size=224):
    transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)
