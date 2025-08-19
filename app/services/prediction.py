"""
Machine learning prediction service.
"""
from typing import Dict, Any, Optional
import onnxruntime as ort
import numpy as np
import json
import os
from PIL import Image

from ..core.settings import settings, model_config
from ..core.logger import get_logger
from ..utils.exceptions import ModelLoadError, PredictionError

logger = get_logger(__name__)


class PneumoniaPredictionService:
    """Service for pneumonia prediction using ONNX model."""
    
    def __init__(self, model_path: Optional[str] = None, stats_path: Optional[str] = None):
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
        
    def load_model(self) -> None:
        """Load the ONNX model and statistics."""
        try:
            # Load ONNX model
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Load model statistics
            self._load_model_stats()
            
            logger.info(
                f"Model loaded successfully from {self.model_path} "
                f"(mean={self.mean:.4f}, std={self.std:.4f})"
            )
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Failed to load model: {e}")
    
    def _load_model_stats(self) -> None:
        """Load model normalization statistics."""
        if not os.path.exists(self.stats_path):
            logger.warning(
                f"Model stats file not found at {self.stats_path}, "
                f"using default values (mean={self.mean}, std={self.std})"
            )
            return
        
        try:
            with open(self.stats_path, 'r') as f:
                stats = json.load(f)
                self.mean = stats.get('mean', self.mean)
                self.std = stats.get('std', self.std)
        except Exception as e:
            logger.warning(f"Failed to load model stats: {e}, using defaults")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to grayscale and resize
        image = image.convert("L").resize(model_config.TARGET_SIZE)
        
        # Normalize to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Apply dataset normalization
        img_array = (img_array - self.mean) / self.std
        
        # Add batch and channel dimensions [batch, channel, height, width]
        img_tensor = img_array[np.newaxis, np.newaxis, :, :]
        
        return img_tensor
    
    def _generate_medical_recommendation(
        self, 
        prediction: str, 
        confidence: float
    ) -> str:
        """
        Generate medical recommendation based on prediction and confidence.
        
        Args:
            prediction: Predicted class
            confidence: Prediction confidence
            
        Returns:
            Medical recommendation string
        """
        if prediction == 'PNEUMONIA':
            if confidence > model_config.HIGH_CONFIDENCE_THRESHOLD:
                return "🚨 URGENT: Immediate medical consultation required"
            elif confidence > model_config.MEDIUM_CONFIDENCE_THRESHOLD:
                return "⚠️ Medical consultation strongly recommended"
            else:
                return "⚠️ Consider medical consultation"
        else:
            return "✅ Normal results - maintain regular health checkups"
    
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
            output = self.session.run(
                [self.output_name], 
                {self.input_name: img_tensor}
            )
            logits = output[0][0]
            
            # Apply softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # Get prediction
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])
            prediction = self.label_map[pred_idx]
            
            # Generate medical recommendation
            recommendation = self._generate_medical_recommendation(prediction, confidence)
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": {
                    "NORMAL": float(probs[0]),
                    "PNEUMONIA": float(probs[1])
                },
                "medical_recommendation": recommendation
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise PredictionError(f"Prediction failed: {e}")
    
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
            "labels": list(self.label_map.values())
        }
