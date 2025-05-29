# Combined Channel ResNet Model for Biometric Recognition
# This module implements a modified ResNet architecture that accepts 6-channel input
# for processing fused image pairs in siamese-style networks

import torch
import torch.nn as nn
import torchvision.models as models

class CombinedChannelModel(nn.Module):
    """
    A modified ResNet model that processes 6-channel input for biometric verification.
    
    This model takes two RGB images concatenated as a 6-channel tensor and produces
    a similarity score between 0 and 1, where values > 0.5 indicate genuine pairs
    and values <= 0.5 indicate impostor pairs.
    """
    
    def __init__(self, resnet_type='resnet18', pretrained=True, embedding_dim=128):
        """
        Initialize the Combined Channel Model.
        
        Args:
            resnet_type (str): Type of ResNet architecture ('resnet18', 'resnet34', 'resnet50')
            pretrained (bool): Whether to use ImageNet pretrained weights
            embedding_dim (int): Dimension of the feature embedding layer
        """
        super(CombinedChannelModel, self).__init__()
        
        # Select the appropriate ResNet architecture
        if resnet_type == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
        elif resnet_type == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
        elif resnet_type == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")
        
        print(f"Creating model with architecture: {resnet_type}")
        
        # Modify the first convolutional layer to accept 6 channels instead of 3
        # This allows processing of concatenated image pairs (2 RGB images = 6 channels)
        original_weight = base_model.conv1.weight.clone()
        base_model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the new 6-channel weights by duplicating the original 3-channel weights
        # This preserves pretrained features for both image pairs
        with torch.no_grad():
            base_model.conv1.weight[:, :3] = original_weight  # First 3 channels (first image)
            base_model.conv1.weight[:, 3:] = original_weight  # Last 3 channels (second image)
        
        # Create feature extractor by removing the final classification layer
        modules = list(base_model.children())[:-1]  # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(*modules)
        
        # Dynamically determine the feature size after the feature extractor
        # This ensures compatibility across different ResNet architectures
        with torch.no_grad():
            dummy_input = torch.zeros(1, 6, 256, 128)  # Batch size 1, 6 channels, 256x128 image
            dummy_output = self.feature_extractor(dummy_input)
            feature_size = dummy_output.view(1, -1).shape[1]  # Flatten to get feature dimension
        
        # Create embedding layer to reduce dimensionality to specified embedding_dim
        self.embedding_layer = nn.Linear(feature_size, embedding_dim)
        
        # Create final classifier for binary verification (genuine vs impostor)
        self.classifier = nn.Sequential(
            nn.ReLU(),                    # Non-linear activation
            nn.Dropout(0.5),              # Regularization to prevent overfitting
            nn.Linear(embedding_dim, 1),  # Binary classification output
            nn.Sigmoid()                  # Sigmoid activation for probability output [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 6, height, width)
                             representing concatenated image pairs
        
        Returns:
            torch.Tensor: Similarity scores of shape (batch_size, 1) with values in [0, 1]
                         where > 0.5 indicates genuine pairs, <= 0.5 indicates impostor pairs
        """
        # Extract features using the modified ResNet backbone
        features = self.feature_extractor(x)
        
        # Flatten features for the fully connected layers
        features = features.view(features.size(0), -1)
        
        # Generate embedding representation
        embedding = self.embedding_layer(features)
        
        # Compute final similarity score
        output = self.classifier(embedding)
        
        return output