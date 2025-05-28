import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random


def resize_and_pad(image, target_shape=(256, 128)):
    """
    Redimensiona a imagem mantendo a proporção e adiciona padding para atingir a forma alvo
    
    Args:
        image: Imagem de entrada (numpy array)
        target_shape: Tamanho alvo como (altura, largura)
    
    Returns:
        Imagem redimensionada com padding
    """
    h, w = image.shape[:2]
    target_h, target_w = target_shape

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def create_fused_instance(img1, img2, output_shape=(256, 128)):
    """
    Cria uma instância fundida de duas imagens para entrada do modelo
    
    Args:
        img1: Primeira imagem (numpy array)
        img2: Segunda imagem (numpy array)
        output_shape: Forma de saída como (altura, largura)
    
    Returns:
        Tensor PyTorch com img1 e img2 concatenadas em profundidade
    """
    img1_resized = resize_and_pad(img1, output_shape)
    img2_resized = resize_and_pad(img2, output_shape)
    
    # Concatenar as imagens em profundidade: img1 (3 canais) + img2 (3 canais) = 6 canais
    fused_image = np.dstack((img1_resized, img2_resized))
    
    # Converter para tensor PyTorch
    fused_tensor = torch.from_numpy(fused_image.transpose(2, 0, 1).astype(np.float32) / 255.0)
    
    return fused_tensor


def apply_augmentation(image):
    """
    Aplica transformações de data augmentation para melhorar a generalização
    
    Args:
        image: Imagem de entrada (numpy array RGB)
    
    Returns:
        Imagem com augmentation aplicado
    """
    # Probabilidade de aplicar cada transformação
    p_brightness = 0.5
    p_contrast = 0.3
    p_blur = 0.2
    p_noise = 0.2
    
    # Cópia da imagem
    result = image.copy()
    
    # 1. Alteração de brilho
    if random.random() < p_brightness:
        brightness_factor = random.uniform(0.8, 1.2)
        result = cv2.convertScaleAbs(result, alpha=brightness_factor, beta=0)
    
    # 2. Alteração de contraste
    if random.random() < p_contrast:
        contrast_factor = random.uniform(0.8, 1.2)
        mean = np.mean(result, axis=(0, 1), keepdims=True)
        result = (result - mean) * contrast_factor + mean
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 3. Desfoque gaussiano
    if random.random() < p_blur:
        kernel_size = random.choice([3, 5])
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
    
    # 4. Ruído gaussiano
    if random.random() < p_noise:
        noise = np.random.normal(0, 5, result.shape).astype(np.uint8)
        result = cv2.add(result, noise)
    
    return result


class FusedFramePairsDataset(Dataset):
    """
    PyTorch Dataset que cria pares fundidos de imagens para reconhecimento biométrico
    """
    def __init__(self, pairs, transform=None, output_shape=(256, 128), use_augmentation=False):
        """
        Args:
            pairs: Lista de tuplas (frame1_path, frame2_path, label)
            transform: Transformações PyTorch a serem aplicadas após a fusão (opcional)
            output_shape: Forma de saída das imagens como (altura, largura)
            use_augmentation: Se True, aplica data augmentation nas imagens
        """
        self.pairs = pairs
        self.transform = transform
        self.output_shape = output_shape
        self.use_augmentation = use_augmentation
        
        if use_augmentation:
            print("Data augmentation enabled in dataset")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        frame1_path, frame2_path, label = self.pairs[idx]
        
        # Carregar imagens
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)
        
        if frame1 is None:
            raise ValueError(f"Could not read image: {frame1_path}")
        
        if frame2 is None:
            raise ValueError(f"Could not read image: {frame2_path}")
        
        # Converter de BGR para RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # Aplicar data augmentation se habilitado
        if self.use_augmentation and label == 1:  # Aplica apenas em pares genuínos
            # Aplica augmentation com 50% de chance em cada imagem
            if random.random() < 0.5:
                frame1 = apply_augmentation(frame1)
            if random.random() < 0.5:
                frame2 = apply_augmentation(frame2)
        
        # Criar instância fundida
        fused_tensor = create_fused_instance(frame1, frame2, self.output_shape)
        
        # Aplicar transformações adicionais se necessário
        if self.transform:
            fused_tensor = self.transform(fused_tensor)
        
        # Converter label para tensor
        label = torch.tensor(label, dtype=torch.float32)
        
        return fused_tensor, label