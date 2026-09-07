from __future__ import annotations

from pathlib import Path
from huggingface_hub import hf_hub_download


class ModelManager:
    """Downloads public model artifacts once and keeps them in Drive."""

    SIGN_REPO = "liamxdev/vtsr"
    PLATE_REPO = "Koushim/yolov8-license-plate-detection"
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
        # FP16 TorchScript is a good fit for Colab T4 and avoids a
        # hardware-specific TensorRT engine.
        return self._hf(self.SIGN_REPO, "vtsr.torchscript")

    def sign_mapping(self) -> str:
        return self._hf(self.SIGN_REPO, "label-mapping.json")

    def plate_model(self, prefer_finetuned: bool = True) -> str:
        # If the user fine-tunes on a Vietnamese car-plate dataset, this file
        # automatically overrides the general pretrained detector.
        fine = self.models_dir / "plate_best.pt"
        if prefer_finetuned and fine.exists():
            return str(fine)
        return self._hf(self.PLATE_REPO, "best.pt")

    def scene_model(self) -> str:
        return self._hf(self.SCENE_REPO, "yolo11n.pt")
