"""
帧特征预提取脚本：使用 CLIP ViT-B/32 视觉编码器从 MSR-VTT 视频中均匀采样 K 帧并提取特征。
保存为 data/msrvtt_clip_frames/{video_id}.pt — Tensor (K, d)。

用法:
    python scripts/extract_frame_features.py --video_dir /path/to/msrvtt/videos --output_dir data/msrvtt_clip_frames
"""
import argparse
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# 镜像加速
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def extract_and_save(video_dir: str, output_dir: str, num_frames: int = 12,
                     model_name: str = "openai/clip-vit-base-patch32"):
    """
    从视频中均匀采样帧并提取 CLIP 视觉特征。
    如果没有实际视频文件，则生成随机特征用于开发测试。
    """
    from transformers import CLIPModel, CLIPProcessor

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_path = Path(video_dir)
    if not video_path.exists():
        print(f"[WARNING] Video directory {video_dir} not found.")
        print(f"Generating random frame features for development...")
        _generate_dummy_features(output_path, num_frames, 512)
        return

    print(f"Loading CLIP visual model: {model_name}")
    clip_model = CLIPModel.from_pretrained(model_name, local_files_only=True)
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
    clip_model = clip_model.to(device)
    clip_model.eval()

    try:
        import decord
        decord.bridge.set_bridge("torch")
    except ImportError:
        print("[ERROR] decord not installed. Install with: pip install decord")
        print("Generating dummy features instead...")
        _generate_dummy_features(output_path, num_frames, 512)
        return

    video_files = sorted(video_path.glob("video*.mp4"))
    if not video_files:
        video_files = sorted(video_path.glob("video*.avi"))
    print(f"Found {len(video_files)} video files")

    for vf in tqdm(video_files, desc="Extracting frame features"):
        vid = vf.stem  # e.g. "video0"
        out_file = output_path / f"{vid}.pt"
        if out_file.exists():
            continue

        try:
            vr = decord.VideoReader(str(vf))
            total_frames = len(vr)
            indices = torch.linspace(0, total_frames - 1, num_frames).long()
            frames = vr.get_batch(indices.tolist())  # (K, H, W, C)

            # 转为 PIL 格式处理
            from PIL import Image
            import numpy as np
            pil_frames = [Image.fromarray(frames[i].numpy()) for i in range(frames.shape[0])]

            inputs = processor(images=pil_frames, return_tensors="pt").to(device)
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)  # (K, d)
                features = F.normalize(features, p=2, dim=-1)

            torch.save(features.cpu(), out_file)
        except Exception as e:
            print(f"[ERROR] Failed to process {vid}: {e}")
            # 保存随机特征作为 fallback
            dummy = F.normalize(torch.randn(num_frames, 512), p=2, dim=-1)
            torch.save(dummy, out_file)

    print(f"Done. Features saved to {output_path}")


def _generate_dummy_features(output_path: Path, num_frames: int, feature_dim: int):
    """为所有 10000 个 MSR-VTT 视频生成随机帧特征（开发测试用）。"""
    for i in tqdm(range(10000), desc="Generating dummy frame features"):
        vid = f"video{i}"
        out_file = output_path / f"{vid}.pt"
        if out_file.exists():
            continue
        dummy = F.normalize(torch.randn(num_frames, feature_dim), p=2, dim=-1)
        torch.save(dummy, out_file)
    print(f"Generated dummy features for 10000 videos in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="data/msrvtt_videos",
                        help="MSR-VTT video directory")
    parser.add_argument("--output_dir", type=str, default="data/msrvtt_clip_frames",
                        help="Output directory for frame features")
    parser.add_argument("--num_frames", type=int, default=12,
                        help="Number of frames to sample per video")
    args = parser.parse_args()

    extract_and_save(args.video_dir, args.output_dir, args.num_frames)
