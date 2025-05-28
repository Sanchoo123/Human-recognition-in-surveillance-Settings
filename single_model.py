import torch
import torch.nn as nn
import torchvision.models as models

class CombinedChannelModel(nn.Module):
    def __init__(self, resnet_type='resnet18', pretrained=True, embedding_dim=128):
        super(CombinedChannelModel, self).__init__()
        
        # Escolher a arquitetura ResNet
        if resnet_type == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
        elif resnet_type == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
        elif resnet_type == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Tipo de ResNet não suportado: {resnet_type}")
        
        print(f"Criando modelo com arquitetura: {resnet_type}")
        
        # Modificar a primeira camada para aceitar 6 canais
        original_weight = base_model.conv1.weight.clone()
        base_model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            base_model.conv1.weight[:, :3] = original_weight
            base_model.conv1.weight[:, 3:] = original_weight
        
        # Extrator de características
        modules = list(base_model.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        
        # Determinar o tamanho das características dinamicamente
        with torch.no_grad():
            dummy_input = torch.zeros(1, 6, 256, 128)
            dummy_output = self.feature_extractor(dummy_input)
            feature_size = dummy_output.view(1, -1).shape[1]
        
        # Classificador com embedding_dim (mantendo a estrutura original)
        self.embedding_layer = nn.Linear(feature_size, embedding_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        embedding = self.embedding_layer(features)
        output = self.classifier(embedding)
        return output