"""Grounding DINO client for object detection.

Supports three runtime modes:
- ``mock``: deterministic fake detections, useful for CI/offline tests.
- ``api``: HTTP endpoint that returns label/bbox/confidence JSON.
- ``local``: lazy-loading stub that logs once and returns empty detections.
  Real checkpoint loading is intentionally not implemented to avoid
  downloading large weights.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from loguru import logger

from common.schema import DetectedObject
from rethinker_promptforge.config import load_config


class DINOClient:
    """Object detector backed by a grounding model.

    Configuration is read from ``configs/models.yaml`` under the ``dino`` key.
    ``image_size`` and ``patch_size`` are used to resize inputs to dimensions
    compatible with the vision transformer before running detection.
    """

    VALID_MODES = {"mock", "api", "local"}

    def __init__(
        self,
        config_path: str | Path | None = None,
        mode: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 8.0,
    ) -> None:
        if config_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            config_path = repo_root / "configs" / "models.yaml"
        cfg = load_config(config_path).get("dino", {})

        self.mode = (mode or cfg.get("mode", "mock")).lower()
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid DINO mode {self.mode}. "
                f"Choose one of {sorted(self.VALID_MODES)}."
            )

        self.model_id = cfg.get("model_id", "facebook/dino-vitb16")
        self.device = cfg.get("device", "cuda")
        self.patch_size = int(cfg.get("patch_size", 16))
        self.image_size = int(cfg.get("image_size", 518))
        self.base_url = str(cfg.get("base_url", "http://localhost:8002")).rstrip("/")
        self.api_key = api_key or cfg.get("api_key") or None

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        self._model: Any | None = None
        self._model_loaded = False

        logger.info(
            "DINOClient initialized: mode={}, model_id={}, device={}, "
            "image_size={}, patch_size={}, base_url={}",
            self.mode,
            self.model_id,
            self.device,
            self.image_size,
            self.patch_size,
            self.base_url,
        )

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode an RGB numpy image as a base64 PNG data URL."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected an RGB or grayscale image, got shape {image.shape}"
            )
        from PIL import Image

        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Resize ``image`` to a ViT-friendly size and return it plus scale.

        The longer side is scaled to ``image_size`` (rounded down to a
        multiple of ``patch_size``) while preserving aspect ratio. Returned
        detections are scaled back to the original image coordinates.
        """
        if image.ndim == 2:
            rgb = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb = image[:, :, :3]
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb = image.copy()
        else:
            raise ValueError(f"Expected an RGB or grayscale image, got shape {image.shape}")

        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        if self.image_size <= 0 or self.patch_size <= 0:
            return rgb, 1.0

        orig_h, orig_w = rgb.shape[:2]
        max_side = max(orig_h, orig_w)
        target = (self.image_size // self.patch_size) * self.patch_size
        if target == 0:
            target = self.patch_size
        scale = target / max_side

        new_h = max(int(round(orig_h * scale)) // self.patch_size * self.patch_size, self.patch_size)
        new_w = max(int(round(orig_w * scale)) // self.patch_size * self.patch_size, self.patch_size)

        if (new_h, new_w) != (orig_h, orig_w):
            from PIL import Image

            pil_image = Image.fromarray(rgb)
            resized = pil_image.resize((new_w, new_h), Image.BILINEAR)
            rgb = np.array(resized)
            logger.info(
                "Resized DINO input from {}x{} to {}x{} (scale={:.4f})",
                orig_w,
                orig_h,
                new_w,
                new_h,
                scale,
            )

        return rgb, scale

    def _mock_detect(self, image: np.ndarray) -> list[DetectedObject]:
        """Return deterministic fake detections scaled to image dimensions."""
        height, width = image.shape[:2]
        return [
            DetectedObject(
                label="mock_object",
                bbox=[
                    width * 0.25,
                    height * 0.25,
                    width * 0.75,
                    height * 0.75,
                ],
                confidence=0.95,
            )
        ]

    def _api_detect(self, image: np.ndarray) -> list[DetectedObject]:
        """Send the image to an API endpoint and parse detections."""
        payload = {
            "model": self.model_id,
            "image": self._encode_image(image),
        }
        url = f"{self.base_url}/detect"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                detections = data.get("detections", [])
                if not isinstance(detections, list):
                    raise ValueError(f"detections must be a list, got {type(detections)}")
                return [self._parse_detection(item) for item in detections]
            except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
                last_exception = exc
                logger.warning(
                    "DINO API request failed (attempt {}/{}): {}",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    time.sleep(delay)
        raise RuntimeError(
            f"DINO API request failed after {self.max_retries + 1} attempts"
        ) from last_exception

    @staticmethod
    def _parse_detection(item: Any) -> DetectedObject:
        """Convert a raw detection dict into a validated ``DetectedObject``."""
        if isinstance(item, DetectedObject):
            return item
        if not isinstance(item, dict):
            raise ValueError(f"Detection item must be a dict, got {type(item)}")
        bbox = item.get("bbox") or item.get("box")
        if bbox is None:
            raise ValueError("Detection item missing bbox or box")
        return DetectedObject(
            label=str(item.get("label", "unknown")),
            bbox=[float(v) for v in bbox],
            confidence=float(item.get("confidence", item.get("score", 0.0))),
        )

    def _local_detect(self, image: np.ndarray) -> list[DetectedObject]:
        """Stub for local checkpoint inference; no weights are downloaded."""
        if not self._model_loaded:
            logger.info(
                "Local DINO checkpoint not loaded (stub). model_id={}",
                self.model_id,
            )
            self._model_loaded = True
        return []

    def detect(self, image: np.ndarray) -> list[DetectedObject]:
        """Run object detection on ``image`` and return labeled boxes.

        Args:
            image: RGB or grayscale numpy array. ``np.uint8`` preferred.

        Returns:
            A list of ``DetectedObject`` instances. Empty list when no
            objects are detected or when using the local stub.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be a numpy ndarray, got {type(image)}")
        if image.ndim not in (2, 3):
            raise ValueError(f"image must be 2D or 3D, got shape {image.shape}")

        preprocessed, scale = self._preprocess_image(image)

        if self.mode == "mock":
            detections = self._mock_detect(preprocessed)
        elif self.mode == "api":
            detections = self._api_detect(preprocessed)
        elif self.mode == "local":
            detections = self._local_detect(preprocessed)
        else:
            # Defensive: should never happen because __init__ validates mode.
            raise RuntimeError(f"Unsupported DINO mode: {self.mode}")

        if scale != 1.0:
            detections = [
                det.model_copy(update={"bbox": [coord / scale for coord in det.bbox]})
                for det in detections
            ]

        if not detections:
            logger.info("DINO returned no detections.")
        return detections
