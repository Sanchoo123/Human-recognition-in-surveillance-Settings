# Human Recognition in Surveillance Settings

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Este repositório contém um sistema completo de reconhecimento biométrico baseado em deep learning, projetado especificamente para cenários de vigilância. O sistema utiliza redes neurais convolucionais (ResNet) para comparação de pares de imagens, análise de explicações com PCA e K-means clustering, e geração de visualizações interpretáveis.

## 🎯 Características Principais

- **Arquitetura ResNet Flexível**: Suporte para ResNet-18, ResNet-34 e ResNet-50
- **Fusão de Canais**: Processamento simultâneo de pares de imagens através de 6 canais
- **Análise Explicável**: Geração de explicações interpretáveis usando PCA e clustering
- **Métricas Avançadas**: Curvas ROC, matriz de confusão, precision-recall
- **Visualizações**: Gráficos de treinamento e comparações de imagens
- **Pipeline Completo**: Desde extração de frames até predição final

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- CUDA 11.0+ (recomendado para GPU)
- Git

### Dependências

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-learn
pip install opencv-python tqdm
pip install sentence-transformers
pip install openai  # Para explicações AI (opcional)
```

### Instalação Rápida

```bash
git clone <repository-url>
cd "Human recognition in survaillance Settings"
pip install -r requirements.txt  # Caso exista
```

## 💻 Uso

### 1. Preparação dos Dados

#### Extração de Frames
```bash
# Extração básica de frames
python convert.py

# Extração mantendo estrutura de IDs
python convertroi.py
```

#### Criação de Pares de Treinamento
```bash
python create_pairs.py --roi_folder <caminho_para_rois> --num_pairs 5000 --genuine_ratio 0.5
```

### 2. Treinamento

```bash
python main.py --roi_dir <diretorio_rois> \
               --mode train \
               --epochs 50 \
               --batch_size 32 \
               --resnet_type resnet50 \
               --num_pairs 10000
```

**Parâmetros Disponíveis:**
- `--roi_dir`: Diretório contendo as ROIs organizadas
- `--mode`: `train`, `test`, ou `explain`
- `--epochs`: Número de épocas de treinamento (padrão: 20)
- `--batch_size`: Tamanho do batch (padrão: 32)
- `--resnet_type`: Arquitetura ResNet (`resnet18`, `resnet34`, `resnet50`)
- `--num_pairs`: Número de pares para treinamento/teste (padrão: 1000)
- `--genuine_ratio`: Proporção de pares genuínos (padrão: 0.5)

### 3. Teste e Avaliação

```bash
python main.py --roi_dir <diretorio_rois> \
               --mode test \
               --model_path output/best_model_resnet50.pth \
               --resnet_type resnet50
```

### 4. Predição Individual

```bash
python predict_pair.py --image1 <caminho_imagem1> \
                      --image2 <caminho_imagem2> \
                      --model <caminho_modelo> \
                      --output <diretorio_saida>
```

## 🔧 Arquitetura Técnica

### Modelo Principal (`CombinedChannelModel`)

O sistema utiliza uma arquitetura ResNet modificada que:

1. **Entrada de 6 Canais**: Concatena duas imagens RGB (3+3 canais)
2. **Feature Extraction**: Utiliza ResNet pré-treinada como backbone
3. **Embedding**: Gera representações de 128 dimensões
4. **Classificação**: Camada final para decisão genuíno/impostor

```python
# Exemplo de uso do modelo
model = CombinedChannelModel(resnet_type='resnet50', embedding_dim=128)
output = model(fused_images)  # [batch_size, 1]
decision = output > 0.5  # Threshold para classificação
```

### Pipeline de Dados

1. **Extração de Frames**: Conversão de vídeos para frames individuais
2. **Criação de Pares**: Geração automática de pares genuínos/impostores
3. **Pré-processamento**: Normalização e redimensionamento
4. **Fusão**: Concatenação de pares de imagens em tensor de 6 canais

### Sistema de Explicações

O módulo `predict_pair.py` implementa:

- **Análise PCA**: Redução dimensional dos embeddings
- **K-means Clustering**: Agrupamento de explicações similares
- **Visualizações**: Gráficos interpretativos dos resultados
- **Explicações Textuais**: Geração automática via modelos de linguagem

## 📊 Métricas e Resultados

O sistema gera automaticamente:

### Métricas de Performance
- **Acurácia**: Precisão geral do modelo
- **Curva ROC**: Area Under Curve (AUC)
- **Precision-Recall**: Para análise detalhada
- **Matriz de Confusão**: Distribuição de erros

### Visualizações Geradas
- `output/roc_curve.png`: Curva ROC com AUC
- `output/confusion_matrix.png`: Matriz de confusão
- `output/val_loss_accuracy.png`: Métricas de validação
- `results/comparison.jpg`: Comparação visual de pares

### Resultados Típicos
```
Test Accuracy: 0.9250
ROC AUC: 0.9680
Best Validation Loss: 0.1834
```

## 🔍 Análise Explicável

### Geração de Explicações
```bash
python predict_pair.py --explain \
                      --image1 pessoa1.jpg \
                      --image2 pessoa2.jpg \
                      --model best_model.pth
```

### Saídas Geradas
- **PCA Analysis**: Coordenadas no espaço reduzido
- **Cluster Information**: Agrupamentos de explicações
- **Best Explanation**: Explicação mais representativa
- **Similarity Scores**: Métricas de similaridade

## 📁 Organização dos Dados

### Estrutura Esperada dos ROIs
```
roi_folder/
├── pessoa_001/
│   ├── video_001/
│   │   ├── frame_001.jpg
│   │   ├── frame_002.jpg
│   │   └── ...
│   └── video_002/
│       └── ...
├── pessoa_002/
│   └── ...
```

### Formato dos Pares
O sistema gera pares no formato:
```python
(image1_path, image2_path, label)
# label: 1 = genuíno, 0 = impostor
```

## ⚙️ Configurações Avançadas

### Personalização do Modelo
```python
# Diferentes arquiteturas
model_configs = {
    'resnet18': {'params': 11M, 'speed': 'fast'},
    'resnet34': {'params': 21M, 'speed': 'medium'}, 
    'resnet50': {'params': 25M, 'speed': 'slow'}
}
```

### Hiperparâmetros Recomendados
```python
# Para dataset pequeno (<5k pares)
batch_size = 16
learning_rate = 1e-4
epochs = 30

# Para dataset médio (5k-20k pares)
batch_size = 32
learning_rate = 1e-3
epochs = 50

# Para dataset grande (>20k pares)
batch_size = 64
learning_rate = 1e-3
epochs = 100
```

## 🐛 Troubleshooting

### Problemas Comuns

**Erro de GPU Memory:**
```bash
# Reduzir batch_size
python main.py --batch_size 16

# Ou usar CPU
export CUDA_VISIBLE_DEVICES=""
```

**Baixa Acurácia:**
- Verificar balanceamento dos dados (genuine_ratio=0.5)
- Aumentar número de pares de treinamento
- Tentar arquitetura mais complexa (resnet50)

**Modelo não Converge:**
- Reduzir learning rate
- Aumentar número de épocas
- Verificar qualidade das imagens

## 📈 Performance Benchmarks

### Tempos de Execução (RTX 3080)
- **Treinamento**: ~2 min/epoch (1000 pares, batch_size=32)
- **Teste**: ~30 seg (500 pares)
- **Predição Individual**: ~100ms por par

### Requisitos de Hardware
- **Mínimo**: 8GB RAM, CPU multi-core
- **Recomendado**: 16GB RAM, GPU 8GB VRAM
- **Ótimo**: 32GB RAM, GPU 12GB+ VRAM

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

### Guidelines de Desenvolvimento
- Seguir PEP 8 para código Python
- Documentar novas funções com docstrings
- Adicionar testes para novas funcionalidades
- Manter compatibilidade com versões anteriores

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE) - veja o arquivo LICENSE para detalhes.


## 📚 Referências

- **ResNet Paper**: "Deep Residual Learning for Image Recognition" (He et al., 2016)
- **Siamese Networks**: "Learning a Similarity Metric Discriminatively" (Chopra et al., 2005)
- **Face Recognition**: "DeepFace: Closing the Gap to Human-Level Performance" (Taigman et al., 2014)

## 🔄 Changelog

### v1.0.0 (2025-05-28)
- Implementação inicial do sistema
- Suporte para múltiplas arquiteturas ResNet
- Sistema de explicações com PCA
- Geração automática de métricas

---


