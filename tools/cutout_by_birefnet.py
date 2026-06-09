from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation


IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("指定了 --device cuda，但当前环境不可用 CUDA。")
    return device


def _resolve_model_source(model_name: str | Path) -> str:
    p = Path(model_name).expanduser()
    if p.exists():
        return str(p.resolve())

    c1 = Path("./models/BiRefNet") / p.name
    if c1.exists():
        return str(c1.resolve())

    c2 = Path("./models") / p.name
    if c2.exists():
        return str(c2.resolve())

    return str(model_name)


def _collect_images(input_path: Path) -> tuple[list[Path], Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"不支持的图片格式: {input_path.suffix}")
        return [input_path], input_path.parent

    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    images = sorted(
        p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return images, input_path


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )


def _unwrap_prediction(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output

    if isinstance(output, (list, tuple)):
        for item in reversed(output):
            try:
                return _unwrap_prediction(item)
            except TypeError:
                continue
        raise TypeError("模型输出中未找到可用 tensor。")

    if isinstance(output, dict):
        for key in ("logits", "preds", "mask", "masks", "output"):
            if key in output:
                return _unwrap_prediction(output[key])

    for attr in ("logits", "preds", "mask", "masks"):
        value = getattr(output, attr, None)
        if value is not None:
            return _unwrap_prediction(value)

    raise TypeError(f"无法识别的模型输出类型: {type(output)!r}")


def _predict_alpha(
    model,
    transform_image: transforms.Compose,
    image_rgb: Image.Image,
    device: str,
    use_fp16: bool,
    alpha_threshold: float,
) -> Image.Image:
    input_tensor = transform_image(image_rgb).unsqueeze(0).to(device)
    if use_fp16:
        input_tensor = input_tensor.half()

    with torch.inference_mode():
        output = model(input_tensor)
        pred = _unwrap_prediction(output)
        pred = pred.sigmoid().float().cpu()[0].squeeze().numpy()

    pred_img = Image.fromarray((np.clip(pred, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    alpha = pred_img.resize(image_rgb.size, Image.Resampling.BILINEAR)
    if alpha_threshold > 0:
        a = np.array(alpha, dtype=np.uint8)
        th = int(max(0, min(255, round(alpha_threshold * 255.0))))
        a = np.where(a >= th, a, 0).astype(np.uint8)
        alpha = Image.fromarray(a, mode="L")
    return alpha


@dataclass
class BiRefNetSegmenter:
    model: object
    transform_image: transforms.Compose
    device: str
    use_fp16: bool

    def predict_alpha(self, image_rgb: Image.Image, alpha_threshold: float = 0.13) -> Image.Image:
        return _predict_alpha(
            model=self.model,
            transform_image=self.transform_image,
            image_rgb=image_rgb.convert("RGB"),
            device=self.device,
            use_fp16=self.use_fp16,
            alpha_threshold=alpha_threshold,
        )

    def cutout(self, image: Image.Image, alpha_threshold: float = 0.13) -> Image.Image:
        rgba = image.convert("RGBA")
        alpha = self.predict_alpha(rgba.convert("RGB"), alpha_threshold=alpha_threshold)
        rgba.putalpha(alpha)
        arr = np.array(rgba, dtype=np.uint8)
        transparent = arr[:, :, 3] == 0
        if np.any(transparent):
            arr[transparent, 0:3] = 0
        return Image.fromarray(arr, mode="RGBA")


def load_birefnet_segmenter(
    model_name: str | Path = "./models/BiRefNet",
    device: str = "auto",
    resolution: int = 1024,
    no_fp16: bool = False,
) -> BiRefNetSegmenter:
    torch.set_float32_matmul_precision("high")
    real_device = _resolve_device(device)
    use_fp16 = real_device == "cuda" and not no_fp16
    model_source = _resolve_model_source(model_name)
    model = AutoModelForImageSegmentation.from_pretrained(
        model_source,
        trust_remote_code=True,
    )
    model.to(real_device)
    model.eval()
    if use_fp16:
        model.half()
    else:
        model.float()
    return BiRefNetSegmenter(
        model=model,
        transform_image=_build_transform(resolution),
        device=real_device,
        use_fp16=use_fp16,
    )


def _build_output_path(
    image_path: Path,
    base_input: Path,
    input_is_dir: bool,
    output_dir: Path,
) -> Path:
    if input_is_dir:
        rel = image_path.relative_to(base_input)
        return (output_dir / rel).with_suffix(".png")
    return output_dir / f"{image_path.stem}.png"


def run_cutout(
    input_dir: str | Path = "./datasets/raw-img",
    output_dir: str | Path = "./datasets/raw-img-cutout",
    model_name: str | Path = "./models/BiRefNet",
    device: str = "auto",
    resolution: int = 1024,
    no_fp16: bool = False,
    alpha_threshold: float = 0.0,
) -> dict[str, int | str]:
    input_path = Path(input_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images, base_input = _collect_images(input_path)
    if not images:
        raise RuntimeError(f"未找到可处理图片: {input_path}")

    segmenter = load_birefnet_segmenter(
        model_name=model_name,
        device=device,
        resolution=resolution,
        no_fp16=no_fp16,
    )
    input_is_dir = input_path.is_dir()
    ok = 0
    skipped = 0
    errors: list[str] = []

    for image_path in tqdm(images, desc="BiRefNet 去背景"):
        try:
            with Image.open(image_path) as image:
                cutout = segmenter.cutout(image, alpha_threshold=alpha_threshold)

            out_path = _build_output_path(
                image_path=image_path,
                base_input=base_input,
                input_is_dir=input_is_dir,
                output_dir=out_dir,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cutout.save(out_path)
            ok += 1
        except Exception as exc:
            skipped += 1
            if len(errors) < 5:
                errors.append(f"{image_path}: {exc}")

    if ok == 0 and skipped > 0:
        detail = "\n".join(errors) if errors else "无详细错误。"
        raise RuntimeError(f"BiRefNet 未成功输出任何图片，失败 {skipped} 张。\n{detail}")

    return {
        "total": len(images),
        "ok": ok,
        "skipped": skipped,
        "output_dir": str(out_dir),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 BiRefNet 对图片执行背景去除。")
    parser.add_argument("--input_dir", "--input-dir", dest="input_dir", default="./result/front_parts", help="输入图片或目录。")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="./result/img-cutout", help="输出目录。")
    parser.add_argument("--model_name", "--model-name", "--weight", dest="model_name", default="./models/BiRefNet", help="模型目录或名称。")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="推理设备。")
    parser.add_argument("--resolution", type=int, default=1024, help="推理分辨率。")
    parser.add_argument("--no_fp16", action="store_true", help="禁用 fp16。")
    parser.add_argument(
        "--alpha_threshold",
        type=float,
        default=0.13,
        help="alpha 阈值(0~1)，>0 时将低于阈值的 alpha 置 0。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_cutout(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        resolution=int(args.resolution),
        no_fp16=bool(args.no_fp16),
        alpha_threshold=float(args.alpha_threshold),
    )
    print(
        f"完成: total={stats['total']} ok={stats['ok']} skipped={stats['skipped']} "
        f"output_dir={stats['output_dir']}"
    )


if __name__ == "__main__":
    main()
