from __future__ import annotations

from pathlib import Path
from huggingface_hub import hf_hub_download


class ModelManager:
    """Download/cache the single traffic-sign model used by the final project."""

    SIGN_REPO = "liamxdev/vtsr"

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def sign_model(self) -> str:
        # FP16 TorchScript works well on Colab T4/L4/A100 and avoids
        # hardware-specific TensorRT engine compatibility issues.
        return hf_hub_download(
            repo_id=self.SIGN_REPO,
            filename="vtsr.torchscript",
            local_dir=str(self.models_dir),
        )
