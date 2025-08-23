model_path: str = "models/pneumonia_model_standard.onnx"
model_path_efficientnet_b0: str = "models/pneumonia_model_efficientnet_b0.onnx"

sliceing = model_path_efficientnet_b0.split('model_')[-1].split('.')[0]  # Extract model type from path
print(sliceing)  # Output: pneumonia_model_standard