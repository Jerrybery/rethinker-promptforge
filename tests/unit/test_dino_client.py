"""Unit tests for the DINO object-detection client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests
import responses

from common.schema import DetectedObject
from perception.dino_client import DINOClient

REPO_ROOT = Path(__file__).resolve().parents[2]
DINO_API_URL = "http://localhost:8002"


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(
        """
dino:
  model_id: test-dino
  device: cpu
  patch_size: 16
  image_size: 224
  mode: mock
  base_url: http://localhost:8002
  api_key: null
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def local_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "models-local.yaml"
    path.write_text(
        """
dino:
  model_id: test-dino
  device: cpu
  patch_size: 16
  image_size: 0
  mode: local
  checkpoint_path: /fake/checkpoint
  base_url: http://localhost:8002
  api_key: null
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.zeros((100, 200, 3), dtype=np.uint8)


@pytest.fixture
def client(config_path: Path) -> DINOClient:
    return DINOClient(config_path=config_path, max_retries=0)


def _detection_response(detections: list[dict[str, Any]]) -> dict[str, Any]:
    return {"model": "test-dino", "detections": detections}


class TestInitialization:
    def test_defaults_to_mock_mode(self, config_path: Path) -> None:
        client = DINOClient(config_path=config_path)
        assert client.mode == "mock"

    def test_mode_override(self, config_path: Path) -> None:
        client = DINOClient(config_path=config_path, mode="local")
        assert client.mode == "local"

    def test_invalid_mode_raises(self, config_path: Path) -> None:
        with pytest.raises(ValueError, match="Invalid DINO mode"):
            DINOClient(config_path=config_path, mode="bad_mode")

    def test_logs_configuration(self, config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_logger = MagicMock()
        monkeypatch.setattr("perception.dino_client.logger", mock_logger)
        client = DINOClient(config_path=config_path)
        mock_logger.info.assert_called_once()
        log_args = mock_logger.info.call_args.args
        assert client.mode in log_args
        assert client.model_id in log_args
        assert client.device in log_args
        assert client.patch_size in log_args


class TestMockMode:
    def test_returns_one_detection(self, client: DINOClient, sample_image: np.ndarray) -> None:
        results = client.detect(sample_image)
        assert len(results) == 1
        det = results[0]
        assert det.label == "mock_object"
        assert det.confidence == pytest.approx(0.95)
        assert det.bbox == pytest.approx([50.0, 25.0, 150.0, 75.0])

    def test_bbox_scales_with_image_size(self, client: DINOClient) -> None:
        image = np.zeros((400, 800, 3), dtype=np.uint8)
        det = client.detect(image)[0]
        assert det.bbox == pytest.approx([200.0, 100.0, 600.0, 300.0])

    def test_bbox_scales_back_to_original_coordinates(self, client: DINOClient) -> None:
        """Non-square images exercise separate scale_x / scale_y handling."""
        image = np.zeros((123, 456, 3), dtype=np.uint8)
        det = client.detect(image)[0]
        assert det.bbox == pytest.approx(
            [
                456 * 0.25,
                123 * 0.25,
                456 * 0.75,
                123 * 0.75,
            ]
        )

    def test_grayscale_image(self, client: DINOClient) -> None:
        gray = np.zeros((100, 200), dtype=np.uint8)
        det = client.detect(gray)[0]
        assert det.label == "mock_object"


class TestPreprocessing:
    def test_resizes_to_multiple_of_patch_size(self, client: DINOClient) -> None:
        image = np.zeros((123, 456, 3), dtype=np.uint8)
        preprocessed, scale_x, scale_y = client._preprocess_image(image)
        assert preprocessed.shape[2] == 3
        assert preprocessed.shape[0] % client.patch_size == 0
        assert preprocessed.shape[1] % client.patch_size == 0
        assert scale_x != 1.0
        assert scale_y != 1.0

    def test_separate_xy_scales_for_non_square_image(self, client: DINOClient) -> None:
        image = np.zeros((123, 456, 3), dtype=np.uint8)
        preprocessed, scale_x, scale_y = client._preprocess_image(image)
        assert scale_x == pytest.approx(preprocessed.shape[1] / 456)
        assert scale_y == pytest.approx(preprocessed.shape[0] / 123)
        assert scale_x != pytest.approx(scale_y)

    def test_respects_disable_image_size(self, tmp_path: Path) -> None:
        path = tmp_path / "models.yaml"
        path.write_text(
            """
dino:
  model_id: test-dino
  device: cpu
  patch_size: 16
  image_size: 0
  mode: mock
  base_url: http://localhost:8002
  api_key: null
""",
            encoding="utf-8",
        )
        client = DINOClient(config_path=path)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        preprocessed, scale_x, scale_y = client._preprocess_image(image)
        assert preprocessed.shape == image.shape
        assert scale_x == 1.0
        assert scale_y == 1.0

    def test_logs_resize(self, client: DINOClient, sample_image: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_logger = MagicMock()
        monkeypatch.setattr("perception.dino_client.logger", mock_logger)
        client.detect(sample_image)
        resize_calls = [
            call for call in mock_logger.info.call_args_list
            if "Resized DINO input" in str(call.args)
        ]
        assert len(resize_calls) == 1


class TestAPIMode:
    @pytest.fixture
    def api_client(self, config_path: Path) -> DINOClient:
        return DINOClient(config_path=config_path, mode="api", max_retries=0)

    @responses.activate
    def test_detect_parses_response(self, api_client: DINOClient, sample_image: np.ndarray) -> None:
        responses.post(
            f"{DINO_API_URL}/detect",
            json=_detection_response(
                [
                    {
                        "label": "mug",
                        "bbox": [10.0, 20.0, 30.0, 40.0],
                        "confidence": 0.87,
                    }
                ]
            ),
        )
        results = api_client.detect(sample_image)
        assert len(results) == 1
        det = results[0]
        assert det.label == "mug"
        _, scale_x, scale_y = api_client._preprocess_image(sample_image)
        assert det.bbox == pytest.approx(
            [10.0 / scale_x, 20.0 / scale_y, 30.0 / scale_x, 40.0 / scale_y]
        )
        assert det.confidence == pytest.approx(0.87)

    @responses.activate
    def test_detect_accepts_box_alias(self, api_client: DINOClient, sample_image: np.ndarray) -> None:
        responses.post(
            f"{DINO_API_URL}/detect",
            json=_detection_response(
                [
                    {
                        "label": "cup",
                        "box": [1.0, 2.0, 3.0, 4.0],
                        "score": 0.77,
                    }
                ]
            ),
        )
        results = api_client.detect(sample_image)
        assert len(results) == 1
        assert results[0].label == "cup"
        _, scale_x, scale_y = api_client._preprocess_image(sample_image)
        assert results[0].bbox == pytest.approx(
            [1.0 / scale_x, 2.0 / scale_y, 3.0 / scale_x, 4.0 / scale_y]
        )
        assert results[0].confidence == pytest.approx(0.77)

    @responses.activate
    def test_empty_detections(self, api_client: DINOClient, sample_image: np.ndarray) -> None:
        responses.post(f"{DINO_API_URL}/detect", json=_detection_response([]))
        results = api_client.detect(sample_image)
        assert results == []

    @responses.activate
    def test_retry_on_network_error(self, config_path: Path, sample_image: np.ndarray) -> None:
        client = DINOClient(
            config_path=config_path,
            mode="api",
            max_retries=2,
            base_delay=0.0,
            max_delay=0.0,
        )
        responses.add(
            responses.POST,
            f"{DINO_API_URL}/detect",
            body=requests.ConnectionError("boom"),
        )
        responses.add(
            responses.POST,
            f"{DINO_API_URL}/detect",
            json=_detection_response(
                [{"label": "spoon", "bbox": [0, 0, 1, 1], "confidence": 0.5}]
            ),
        )
        results = client.detect(sample_image)
        assert len(results) == 1
        assert results[0].label == "spoon"
        assert len(responses.calls) == 2

    @responses.activate
    def test_retry_exhausted_raises(self, config_path: Path, sample_image: np.ndarray) -> None:
        client = DINOClient(
            config_path=config_path,
            mode="api",
            max_retries=1,
            base_delay=0.0,
            max_delay=0.0,
        )
        responses.add(
            responses.POST,
            f"{DINO_API_URL}/detect",
            body=requests.Timeout("timeout"),
        )
        responses.add(
            responses.POST,
            f"{DINO_API_URL}/detect",
            body=requests.ConnectionError("boom"),
        )
        with pytest.raises(RuntimeError):
            client.detect(sample_image)

    @responses.activate
    def test_malformed_detection_item_raises(self, api_client: DINOClient, sample_image: np.ndarray) -> None:
        responses.post(
            f"{DINO_API_URL}/detect",
            json=_detection_response([{"label": "mug"}]),
        )
        with pytest.raises(ValueError):
            api_client.detect(sample_image)

    @responses.activate
    def test_bad_detections_type_raises(self, api_client: DINOClient, sample_image: np.ndarray) -> None:
        responses.post(
            f"{DINO_API_URL}/detect",
            json={"model": "test-dino", "detections": "not-a-list"},
        )
        with pytest.raises(ValueError):
            api_client.detect(sample_image)

    @responses.activate
    def test_validation_errors_are_not_retried(self, config_path: Path, sample_image: np.ndarray) -> None:
        client = DINOClient(
            config_path=config_path,
            mode="api",
            max_retries=2,
            base_delay=0.0,
            max_delay=0.0,
        )
        responses.post(
            f"{DINO_API_URL}/detect",
            json={"model": "test-dino", "detections": "not-a-list"},
        )
        with pytest.raises(ValueError):
            client.detect(sample_image)
        assert len(responses.calls) == 1

    @responses.activate
    def test_no_authorization_when_api_key_missing(self, api_client: DINOClient, sample_image: np.ndarray) -> None:
        responses.post(
            f"{DINO_API_URL}/detect",
            json=_detection_response([]),
        )
        api_client.detect(sample_image)
        assert "Authorization" not in responses.calls[0].request.headers

    @responses.activate
    def test_authorization_sent_when_api_key_set(self, config_path: Path, sample_image: np.ndarray) -> None:
        client = DINOClient(config_path=config_path, mode="api", api_key="secret123")
        responses.post(
            f"{DINO_API_URL}/detect",
            json=_detection_response([]),
        )
        client.detect(sample_image)
        assert responses.calls[0].request.headers["Authorization"] == "Bearer secret123"

    @responses.activate
    def test_image_encoded_as_png(self, api_client: DINOClient, sample_image: np.ndarray) -> None:
        responses.post(f"{DINO_API_URL}/detect", json=_detection_response([]))
        api_client.detect(sample_image)
        payload = json.loads(responses.calls[0].request.body)
        assert payload["image"].startswith("data:image/png;base64,")


class TestLocalMode:
    @staticmethod
    def _patch_transformers(
        monkeypatch: pytest.MonkeyPatch,
        torch: Any,
        bboxes: list[list[float]] | None = None,
        scores: list[float] | None = None,
        labels: list[int] | None = None,
    ) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
        import transformers

        bboxes = bboxes or []
        scores = scores or []
        labels = labels or []

        mock_model = MagicMock()
        mock_model.config.id2label = {0: "mock_cat"}
        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros((1, 3, 224, 224))}

        result = {
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
        }
        mock_processor.post_process_object_detection.return_value = [result]

        model_from_pretrained = MagicMock(return_value=mock_model)
        processor_from_pretrained = MagicMock(return_value=mock_processor)
        monkeypatch.setattr(
            transformers.AutoModelForObjectDetection, "from_pretrained", model_from_pretrained
        )
        monkeypatch.setattr(
            transformers.AutoProcessor, "from_pretrained", processor_from_pretrained
        )
        return mock_model, mock_processor, model_from_pretrained, processor_from_pretrained

    def test_local_without_checkpoint_raises(
        self, config_path: Path, sample_image: np.ndarray
    ) -> None:
        client = DINOClient(config_path=config_path, mode="local")
        with pytest.raises(NotImplementedError, match="checkpoint_path is not configured"):
            client.detect(sample_image)

    def test_local_loads_and_caches_checkpoint(
        self,
        local_config_path: Path,
        sample_image: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        transformers = pytest.importorskip("transformers")
        torch = pytest.importorskip("torch")
        mock_model, _mock_processor, model_from_pretrained, processor_from_pretrained = (
            self._patch_transformers(monkeypatch, torch)
        )

        client = DINOClient(config_path=local_config_path)
        client.detect(sample_image)
        client.detect(sample_image)

        model_from_pretrained.assert_called_once_with("/fake/checkpoint")
        processor_from_pretrained.assert_called_once_with("/fake/checkpoint")
        mock_model.eval.assert_called_once()
        assert client._local_model is mock_model
        assert client._local_processor is not None

    def test_local_forward_returns_detections(
        self,
        local_config_path: Path,
        sample_image: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch = pytest.importorskip("torch")
        self._patch_transformers(
            monkeypatch,
            torch,
            bboxes=[[10.0, 20.0, 30.0, 40.0]],
            scores=[0.88],
            labels=[0],
        )

        client = DINOClient(config_path=local_config_path)
        results = client.detect(sample_image)

        assert len(results) == 1
        det = results[0]
        assert isinstance(det, DetectedObject)
        assert det.label == "mock_cat"
        assert det.confidence == pytest.approx(0.88)
        # image_size=0 disables resizing, so coordinates are preserved.
        assert det.bbox == pytest.approx([10.0, 20.0, 30.0, 40.0])


class TestInputValidation:
    def test_non_ndarray_raises(self, client: DINOClient) -> None:
        with pytest.raises(TypeError):
            client.detect([1, 2, 3])

    def test_bad_dimensions_raises(self, client: DINOClient) -> None:
        with pytest.raises(ValueError):
            client.detect(np.zeros((1, 2, 3, 4)))

    def test_invalid_bbox_order_raises(self) -> None:
        with pytest.raises(ValueError):
            DetectedObject(label="mug", bbox=[10, 20, 5, 40], confidence=0.9)


class TestRepoConfig:
    def test_repo_config_has_dino_keys(self) -> None:
        cfg = DINOClient(config_path=REPO_ROOT / "configs" / "models.yaml")
        assert cfg.model_id == "IDEA-Research/grounding-dino-base"
        assert cfg.mode == "mock"
        assert cfg.device == "cuda"
        assert cfg.image_size == 518
        assert cfg.patch_size == 16
        assert cfg.checkpoint_path is None
