import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Preprocessing pipeline matching ResNet18 training standards
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_brain_model(model_path='brain_mri_model.pth', device='cpu'):
    """Rebuilds the ResNet18 backbone and loads trained weights."""
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    
    # Initialize unweighted ResNet18 structure
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, class_names

class GradCAM:
    """Extracts activations and gradients to produce explainability heatmaps."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # Backward pass from the target class score
        score = output[0, target_class]
        score.backward()

        # Global average pooling on feature gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().cpu().numpy()

        # Apply ReLU to focus only on positive contributions
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize heatmap back to standard image dimensions
        cam = cv2.resize(cam, (224, 224))
        probabilities = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        return cam, target_class, probabilities

def generate_superimposed_mri(pil_img, model, class_names, device='cpu'):
    """Runs inference and returns the prediction details along with the blended heatmap."""
    rgb_img = pil_img.convert('RGB')
    tensor = PREPROCESS(rgb_img).unsqueeze(0).to(device)

    # ResNet18 target: final convolutional layer
    target_layer = model.layer4[1].conv2
    cam_engine = GradCAM(model, target_layer)

    heatmap, pred_idx, probs = cam_engine.generate_heatmap(tensor)

    # Superimpose heatmap onto the original image
    np_img = np.array(rgb_img.resize((224, 224)))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = np.uint8(0.6 * np_img + 0.4 * heatmap_colored)

    return {
        'prediction': class_names[pred_idx],
        'confidence': float(probs[pred_idx]),
        'all_probabilities': {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        'superimposed_image': blended
    }