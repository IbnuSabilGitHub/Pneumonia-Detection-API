"""
Machine learning prediction service.
"""

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

from ..core.logger import get_logger
from ..core.settings import model_config, settings
from ..utils.exceptions import ModelLoadError, PredictionError

logger = get_logger(__name__)


class PneumoniaPredictionService:
    """Service for pneumonia prediction using ONNX model."""

    def __init__(
        self, model_path: Optional[str] = None, stats_path: Optional[str] = None
    ):
        """
        Initialize the prediction service.

        Args:
            model_path: Path to the ONNX model file
            stats_path: Path to the model statistics JSON file
        """
        self.model_path = model_path or settings.model_path
        self.stats_path = stats_path or settings.model_stats_path
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.mean: float = model_config.DEFAULT_MEAN
        self.std: float = model_config.DEFAULT_STD
        self.label_map = model_config.LABEL_MAP
        self.model_type: str = self._extract_model_type()

    def _extract_model_type(self) -> str:
        """Extract model type from model path."""
        try:
            return self.model_path.split("model_")[-1].split(".")[0]
        except (IndexError, AttributeError):
            logger.warning("Could not extract model type from path, using 'unknown'")
            return "unknown"

    def load_model(self) -> None:
        """Load the ONNX model and statistics."""
        try:
            # Load ONNX model
            self.session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            # Load model statistics
            self._load_model_stats()

            logger.info(
                "Model loaded successfully from %s (mean=%.4f, std=%.4f)",
                self.model_path,
                self.mean,
                self.std,
            )

        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def _load_model_stats(self) -> None:
        """Load model normalization statistics."""
        if not os.path.exists(self.stats_path):
            logger.warning(
                "Model stats file not found at %s, using default values (mean=%s, std=%s)",
                self.stats_path,
                self.mean,
                self.std,
            )
            return

        try:
            with open(self.stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
                self.mean = stats.get("mean", self.mean)
                self.std = stats.get("std", self.std)
        except (ValueError, KeyError, IOError) as e:
            logger.warning("Failed to load model stats: %s, using defaults", e)

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model inference.

        Args:
            image: PIL Image object

        Returns:
            Preprocessed image tensor
        """
        # Determine target size based on model type
        target_size = self._get_target_size()

        # Convert to grayscale and resize
        image = image.convert("L").resize(target_size)

        # Normalize to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Apply dataset normalization
        img_array = (img_array - self.mean) / self.std

        # Add batch and channel dimensions [batch, channel, height, width]
        img_tensor = img_array[np.newaxis, np.newaxis, :, :]

        return img_tensor

    def _get_target_size(self) -> tuple:
        """Get target size based on model type."""
        if "efficientnet_b0" in self.model_path.lower():
            return model_config.TARGET_SIZE_B0
        return model_config.TARGET_SIZE

    def _get_architecture_description(self) -> str:
        """Get model architecture description based on model type."""
        if "efficientnet_b0" in self.model_type.lower():
            return (
                "EfficientNet-B0 based pneumonia detection model with transfer learning"
            )
        elif "standard" in self.model_type.lower():
            return "CNN-based pneumonia detection model with custom architecture"
        else:
            return "Deep learning pneumonia detection model"

    def _generate_medical_recommendation(
        self, prediction: str, confidence: float
    ) -> str:
        """
        Generate medical recommendation based on prediction and confidence.

        Args:
            prediction: Predicted class
            confidence: Prediction confidence

        Returns:
            Medical recommendation string
        """
        if prediction == "PNEUMONIA":
            if confidence > model_config.HIGH_CONFIDENCE_THRESHOLD:
                return "URGENT: Immediate medical consultation required"
            elif confidence > model_config.MEDIUM_CONFIDENCE_THRESHOLD:
                return "edical consultation strongly recommended"
            else:
                return "Consider medical consultation"
        else:
            return "Normal results - maintain regular health checkups"

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Perform pneumonia prediction on chest X-ray image.

        Args:
            image: PIL Image object of chest X-ray

        Returns:
            Dictionary containing prediction results

        Raises:
            PredictionError: If prediction fails
        """
        if self.session is None:
            raise PredictionError("Model not loaded")

        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image)

            # Run inference
            output = self.session.run([self.output_name], {self.input_name: img_tensor})
            logits = output[0][0]

            # Apply softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            # Get prediction
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])
            prediction = self.label_map[pred_idx]

            # Generate medical recommendation
            recommendation = self._generate_medical_recommendation(
                prediction, confidence
            )

            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": {
                    "NORMAL": float(probs[0]),
                    "PNEUMONIA": float(probs[1]),
                },
                "medical_recommendation": recommendation,
                "model_info": {
                    "model_type": self.model_type,
                    "model_version": "v1.0",
                    "architecture": self._get_architecture_description(),
                },
            }

        except Exception as e:
            logger.error("Prediction error: %s", e)
            raise PredictionError(f"Prediction failed: {e}") from e

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.session is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.is_loaded():
            return {"loaded": False}

        return {
            "loaded": True,
            "model_path": self.model_path,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "mean": self.mean,
            "std": self.std,
            "target_size": model_config.TARGET_SIZE,
            "labels": list(self.label_map.values()),
        }
