import argparse
import os
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "kartiknarayan/PETALface"
AVAILABLE_MODELS = [
    "swin_arcface_webface4m_tinyface",
    "swin_cosface_webface4m_tinyface",
    "swin_cosface_webface4m_briar",
    "swin_cosface_webface12m_briar",
    "swin_arcface_webface4m",
    "swin_cosface_webface4m",
    "swin_arcface_webface12m",
    "swin_cosface_webface12m",
]


def list_models():
    print("可用模型:")
    for i, name in enumerate(AVAILABLE_MODELS, 1):
        print(f"  {i}. {name}")
    print(f"\n总数: {len(AVAILABLE_MODELS)}")


def download_model(model_name: str, output_dir: str = "models"):
    if model_name not in AVAILABLE_MODELS:
        print(f"错误: 未知模型 '{model_name}'")
        list_models()
        return False

    os.makedirs(output_dir, exist_ok=True)

    filename = f"{model_name}/model.pt"
    local_path = os.path.join(output_dir, model_name, "model.pt")

    print(f"下载模型: {model_name}")
    print(f"  从: {REPO_ID}/{filename}")
    print(f"  到: {local_path}")

    try:
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        file_size = os.path.getsize(downloaded)
        print(f"下载完成! 文件大小: {file_size / 1024 / 1024:.2f} MB")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 PETALface 模型权重")
    parser.add_argument("--model", type=str, default="", help="模型名称")
    parser.add_argument("--output-dir", type=str, default="models", help="输出目录")
    parser.add_argument("--list", action="store_true", help="列出可用模型")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.model:
        download_model(args.model, args.output_dir)
    else:
        print("请指定模型名称 (--model) 或使用 --list 查看可用模型")


if __name__ == "__main__":
    main()
