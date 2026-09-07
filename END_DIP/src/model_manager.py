from __future__ import annotations

from pathlib import Path
from huggingface_hub import hf_hub_download


class ModelManager:
    """Download once to Drive, with one subdirectory per model family."""

    SIGN_REPO = "star092304/traffic-sign-detection-vietnam-yolo"
    HELMET_REPO = "nnsohamnn/helmet-detection-yolo11"
    PLATE_REPO = "Koushim/yolov8-license-plate-detection"
    SCENE_REPO = "Ultralytics/YOLO11"

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _hf(self, repo_id: str, filename: str, family: str) -> str:
        local = self.models_dir / family
        local.mkdir(parents=True, exist_ok=True)
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local))

    def sign_model(self) -> str:
        fine = self.models_dir / "traffic_sign_best.pt"
        if fine.exists():
            return str(fine)
        return self._hf(self.SIGN_REPO, "best.pt", "traffic_sign")

    def scene_model(self) -> str:
        return self._hf(self.SCENE_REPO, "yolo11n.pt", "scene")

    def helmet_model(self) -> str:
        fine = self.models_dir / "helmet_best.pt"
        if fine.exists():
            return str(fine)
        return self._hf(self.HELMET_REPO, "yolov11s(80 epochs).pt", "helmet")

    def helmet_base_model(self) -> str:
        return self._hf(self.HELMET_REPO, "yolov11s(80 epochs).pt", "helmet")

    def plate_model(self) -> str:
        fine = self.models_dir / "plate_best.pt"
        if fine.exists():
            return str(fine)
        return self._hf(self.PLATE_REPO, "best.pt", "license_plate")
