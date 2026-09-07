from __future__ import annotations

from pathlib import Path
from huggingface_hub import hf_hub_download


class ModelManager:
    """Downloads public model artifacts once and keeps them in Drive."""

    SIGN_REPO = "liamxdev/vtsr"
    HELMET_REPO = "nnsohamnn/helmet-detection-yolo11"
    SCENE_REPO = "Ultralytics/YOLO11"

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _hf(self, repo_id: str, filename: str) -> str:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(self.models_dir),
        )

    def sign_model(self) -> str:
        # FP16 TorchScript is a good fit for a Colab T4 and avoids a
        # hardware-specific TensorRT engine.
        return self._hf(self.SIGN_REPO, "vtsr.torchscript")

    def sign_mapping(self) -> str:
        return self._hf(self.SIGN_REPO, "label-mapping.json")

    def helmet_model(self, prefer_finetuned: bool = True) -> str:
        fine = self.models_dir / "helmet_best.pt"
        if prefer_finetuned and fine.exists():
            return str(fine)
        # Small checkpoint is the default for video speed; the same repository
        # also provides a larger 100-epoch YOLO11m checkpoint.
        return self._hf(self.HELMET_REPO, "yolov11s(80 epochs).pt")

    def helmet_model_m(self) -> str:
        return self._hf(self.HELMET_REPO, "yolov11m(100epochs).pt")

    def scene_model(self) -> str:
        return self._hf(self.SCENE_REPO, "yolo11n.pt")
