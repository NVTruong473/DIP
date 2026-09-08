from __future__ import annotations

from pathlib import Path
from huggingface_hub import hf_hub_download


class ModelManager:
    """Cache the traffic-sign model in Google Drive and reuse it after Colab restarts."""

    SIGN_REPO = "liamxdev/vtsr"

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.sign_dir = self.models_dir / "traffic_sign"
        self.sign_dir.mkdir(parents=True, exist_ok=True)

    def sign_model(self) -> str:
        return hf_hub_download(
            repo_id=self.SIGN_REPO,
            filename="vtsr.torchscript",
            local_dir=str(self.sign_dir),
        )
