"""
Biometric Pair Prediction with AI-Powered Explanation System

This module provides comprehensive biometric verification capabilities with intelligent 
explanation generation using GPT-4 Vision. It performs person re-identification 
between image pairs and generates detailed forensic analysis explanations.

Key Features:
- Deep learning-based biometric verification using fused channel ResNet models
- AI-powered explanation generation with multiple prompt strategies
- Advanced clustering analysis for explanation quality assessment
- Comprehensive visualization and reporting capabilities
- Forensic-grade analysis with security-focused insights

Author: Biometric Recognition Research Team
Version: 2.0
Date: 2024

Dependencies:
- PyTorch for deep learning model inference
- OpenAI GPT-4 Vision for explanation generation
- SentenceTransformers for text embedding analysis
- scikit-learn for clustering and similarity analysis
- OpenCV for image processing and visualization
"""

import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from single_model import CombinedChannelModel
from dataset_fusion import create_fused_instance
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import image as skimage
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
import base64
from io import BytesIO

# Configure comprehensive logging for system monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default OpenAI API key (replace with your actual key)
DEFAULT_API_KEY = "yourkey"

def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 format for API transmission.
    
    This function reads an image file and converts it to a base64-encoded string,
    which is required for sending images to the OpenAI Vision API.
    
    Args:
        image_path (str): Path to the image file to encode
        
    Returns:
        str: Base64-encoded string representation of the image
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If there's an error reading the image file
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_explanation_with_gpt4(image_path1, image_path2, decision, score, custom_prompt=None):
    """
    Generate detailed forensic analysis explanation using GPT-4 Vision.
    
    This function leverages GPT-4's vision capabilities to provide expert-level
    analysis of biometric image pairs, including forensic insights, security
    assessments, and technical evaluations.
    
    Args:
        image_path1 (str): Path to the first biometric image
        image_path2 (str): Path to the second biometric image  
        decision (str): Model's classification decision ("Genuine" or "Impostor")
        score (float): Confidence score from the biometric model
        custom_prompt (str, optional): Custom analysis prompt to override default
        
    Returns:
        str: Detailed expert analysis explanation in Portuguese
        
    Raises:
        Exception: If GPT-4 API call fails or returns invalid response
    """
    try:
        # Encode both images to base64 for API transmission
        base64_image1 = encode_image_to_base64(image_path1)
        base64_image2 = encode_image_to_base64(image_path2)
        
        # Use custom prompt if provided, otherwise use comprehensive default
        if custom_prompt:
            prompt = custom_prompt.format(decision=decision, score=score)
        else:
            # Comprehensive forensic analysis prompt with roleplay
            prompt = f"""You are a biometric analysis and security expert with years of experience in forensic image analysis. 
            Your task is to analyze this biometric image pair that was classified as {decision} with a score of {score:.4f}.
            
            As an expert, provide a detailed and technical analysis considering:
            
            Before starting these points, I want a description of the person's gender, environment, clothing colors, hairstyle, etc.
            
            1. Forensic Analysis:
               - Similar or different biometric characteristics
               - Image quality and resolution
               - Possible artifacts or distortions
            
            2. Security Analysis:
               - Confidence level in the classification
               - Possible vulnerability points
               - Recommendations for additional verification
            
            3. Technical Analysis:
               - Specific details that led to the classification
               - Factors that influenced the score
               - Current analysis limitations
            
            Maintain a professional and technical tone, but accessible. Use specialized terminology when appropriate."""
        
        # Configure OpenAI API call with high-resolution image analysis
        client = openai.OpenAI(api_key=DEFAULT_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are a biometric analysis and security expert with deep knowledge in forensic image analysis and biometric authentication systems."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image1}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image2}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"Error generating explanation with GPT-4.1: {str(e)}")
        raise

def predict_image_pair(image_path1, image_path2, model_path, resnet_type='resnet50', threshold=0.5):
    """
    Perform biometric verification between two images using trained deep learning model.
    
    This function loads a pre-trained ResNet-based model and performs person 
    re-identification between two input images, returning both the classification
    decision and confidence score.
    
    Args:
        image_path1 (str): Path to the first biometric image
        image_path2 (str): Path to the second biometric image
        model_path (str): Path to the trained model weights file
        resnet_type (str): ResNet architecture type ('resnet18', 'resnet50', etc.)
        threshold (float): Decision threshold for genuine/impostor classification
        
    Returns:
        tuple: (decision, score) where:
            - decision (str): "Genuine" if same person, "Impostor" if different
            - score (float): Model confidence score between 0 and 1
            
    Raises:
        FileNotFoundError: If image files or model file don't exist
        ValueError: If images cannot be loaded or processed
    """
    # Validate input file existence
    if not os.path.exists(image_path1) or not os.path.exists(image_path2):
        raise FileNotFoundError("One or both image paths do not exist.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("The specified model was not found.")
    
    # Configure computing device (GPU if available, CPU otherwise)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the pre-trained biometric model
    model = CombinedChannelModel(resnet_type=resnet_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Load and preprocess input images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    if img1 is None or img2 is None:
        raise ValueError("Could not load one or both images.")
    
    # Convert from BGR to RGB color space
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Create fused tensor combining both images (6-channel input)
    fused_tensor = create_fused_instance(img1, img2, output_shape=(256, 128))
    
    # Apply ImageNet normalization to the fused tensor
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],  # RGB means for both images
        std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225]   # RGB stds for both images
    )
    fused_tensor = normalize(fused_tensor)
    
    # Perform model inference
    with torch.no_grad():
        fused_input = fused_tensor.unsqueeze(0).to(device)  # Add batch dimension
        prediction = model(fused_input).item()  # Get scalar prediction
    
    # Make decision based on threshold
    decision = "Genuine" if prediction > threshold else "Impostor"
    return decision, prediction

def create_visualization(image_path1, image_path2, decision, score, output_path):
    """
    Create a visual comparison of the image pair with prediction results.
    
    This function generates a side-by-side visualization of the two input images
    with overlaid prediction information, useful for manual verification and
    reporting purposes.
    
    Args:
        image_path1 (str): Path to the first image
        image_path2 (str): Path to the second image
        decision (str): Model's classification decision ("Genuine" or "Impostor")
        score (float): Confidence score from the model
        output_path (str): Path where the visualization will be saved
        
    Raises:
        ValueError: If images cannot be loaded
    """
    # Load input images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not load one or both images.")
    
    # Resize images maintaining aspect ratio
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    target_height = 300
    img1 = cv2.resize(img1, (int(w1 * target_height / h1), target_height))
    img2 = cv2.resize(img2, (int(w2 * target_height / h2), target_height))
    
    # Create white space between images (50 pixels)
    h_max = max(img1.shape[0], img2.shape[0])
    w_total = img1.shape[1] + img2.shape[1] + 50
    
    # Create visualization canvas with space for text
    vis = np.ones((h_max + 100, w_total, 3), dtype=np.uint8) * 255
    
    # Place images on canvas
    vis[50:50+img1.shape[0], 0:img1.shape[1]] = img1
    vis[50:50+img2.shape[0], img1.shape[1]+50:img1.shape[1]+50+img2.shape[1]] = img2
    
    # Add labels and information
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Image titles
    cv2.putText(vis, "Image 1", (10, 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(vis, "Image 2", (img1.shape[1]+60, 30), font, 0.7, (0, 0, 0), 2)
    
    # Prediction result with color coding
    color = (0, 128, 0) if decision == "Genuine" else (0, 0, 255)  # Green for genuine, red for impostor
    result_text = f"Result: {decision} (score: {score:.4f})"
    cv2.putText(vis, result_text, (10, h_max+80), font, 0.8, color, 2)
    
    # Save visualization
    cv2.imwrite(output_path, vis)

def get_image_similarity(img1_path, img2_path):
    """
    Calculate visual similarity between two images using HOG features.
    
    This function computes similarity using Histogram of Oriented Gradients (HOG)
    features and cosine similarity, providing a baseline comparison independent
    of the deep learning model.
    
    Args:
        img1_path (str): Path to the first image
        img2_path (str): Path to the second image
    
    Returns:
        float: Similarity score between 0 and 1 (higher = more similar)
    """
    # Load input images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return 0.0
    
    # Convert to RGB color space
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Resize to standard dimensions for comparison
    size = (224, 224)
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)
    
    # Extract HOG (Histogram of Oriented Gradients) features
    hog1 = skimage.hog(img1, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualize=False)
    hog2 = skimage.hog(img2, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualize=False)
    
    # Calculate cosine similarity between HOG features
    similarity = cosine_similarity([hog1], [hog2])[0][0]
    return float(similarity)
    vis = np.ones((h_max + 100, w_total, 3), dtype=np.uint8) * 255
    
    # Adicionar imagens
    vis[50:50+img1.shape[0], 0:img1.shape[1]] = img1
    vis[50:50+img2.shape[0], img1.shape[1]+50:img1.shape[1]+50+img2.shape[1]] = img2
    
    # Adicionar rótulos e informações
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Títulos das imagens
    cv2.putText(vis, "Imagem 1", (10, 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(vis, "Imagem 2", (img1.shape[1]+60, 30), font, 0.7, (0, 0, 0), 2)
    
    # Resultado da predição
    color = (0, 128, 0) if decision == "Genuine" else (0, 0, 255)  # Verde para genuine, vermelho para impostor
    result_text = f"Resultado: {decision} (score: {score:.4f})"
    cv2.putText(vis, result_text, (10, h_max+80), font, 0.8, color, 2)
    
    # Salvar visualização
    cv2.imwrite(output_path, vis)

def get_image_similarity(img1_path, img2_path):
    """
    Calcula a similaridade entre duas imagens usando características visuais.
    
    Args:
        img1_path: Caminho da primeira imagem
        img2_path: Caminho da segunda imagem
    
    Returns:
        float: Score de similaridade entre 0 e 1
    """
    # Carregar imagens
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return 0.0
    
    # Converter para RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Redimensionar para tamanho padrão
    size = (224, 224)
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)
    
    # Extrair características usando HOG
    hog1 = skimage.hog(img1, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualize=False)
    hog2 = skimage.hog(img2, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualize=False)
    
    # Calcular similaridade do cosseno
    similarity = cosine_similarity([hog1], [hog2])[0][0]
    return float(similarity)

def filter_similar_images(image_path1, image_path2, similar_inputs, similarity_threshold=0.7):
    """
    Filter images based on visual similarity to a reference image.
    
    This function analyzes a list of candidate images and returns only those
    that meet a minimum similarity threshold compared to the reference image,
    useful for quality control and relevance filtering.
    
    Args:
        image_path1 (str): Path to the reference image
        image_path2 (str): Path to the second image (currently not used)
        similar_inputs (list): List of candidate image paths to filter
        similarity_threshold (float): Minimum similarity score (0.0 to 1.0)
    
    Returns:
        list: Filtered list of image paths that meet similarity criteria
    """
    if not similar_inputs:
        return []
    
    # Calculate similarity scores with the reference image
    similarities = []
    for img_path in similar_inputs:
        sim_score = get_image_similarity(image_path1, img_path)
        similarities.append((img_path, sim_score))
    
    # Sort by similarity score (highest first) and filter by threshold
    similarities.sort(key=lambda x: x[1], reverse=True)
    filtered_images = [img for img, sim in similarities if sim >= similarity_threshold]
    
    # Return at most the 3 most similar images
    return filtered_images[:3]

def filter_similar_explanations(explanations, similarity_threshold=0.7):
    """
    Filter explanations based on semantic similarity to reduce redundancy.
    
    This function uses sentence embeddings to identify and filter out highly
    similar explanations, keeping only diverse and representative ones.
    
    Args:
        explanations (list): List of text explanations to filter
        similarity_threshold (float): Minimum similarity threshold for grouping
    
    Returns:
        list: Filtered list of semantically diverse explanations
    """
    if not explanations or len(explanations) <= 1:
        return explanations
    
    # Load sentence transformer model for text embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calculate embeddings for all explanations
    embeddings = model.encode(explanations)
    
    # Compute pairwise similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find groups of similar explanations
    similar_groups = []
    used_indices = set()
    
    for i in range(len(explanations)):
        if i in used_indices:
            continue
            
        # Find explanations similar to the current one
        similar_indices = [j for j in range(len(explanations)) 
                         if similarity_matrix[i][j] >= similarity_threshold 
                         and j not in used_indices]
        
        if similar_indices:
            # Add to group and mark as used
            group = [explanations[j] for j in similar_indices]
            similar_groups.append(group)
            used_indices.update(similar_indices)
    
    # Select the most representative explanation from each group
    filtered_explanations = []
    for group in similar_groups:
        # Use the first explanation of the group as representative
        filtered_explanations.append(group[0])
    
    return filtered_explanations

def find_cluster_centroids(explanations, n_components=2, n_clusters=3):
    """
    Find cluster centroids of explanations using PCA and K-means clustering.
    
    This function applies dimensionality reduction followed by clustering to
    identify the most representative explanations from different semantic clusters.
    
    Args:
        explanations (list): List of text explanations to cluster
        n_components (int): Number of PCA components for dimensionality reduction
        n_clusters (int): Number of clusters for K-means
    
    Returns:
        list: List of explanation centroids (one per cluster)
    """
    if not explanations or len(explanations) <= 1:
        return explanations
    
    # Load sentence transformer model for text embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calculate embeddings for all explanations
    embeddings = model.encode(explanations)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(n_components, len(explanations)-1))
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Apply K-means clustering to find groups
    n_clusters = min(n_clusters, len(explanations))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    
    # Find the closest explanation to each cluster center
    centroids = []
    for cluster_id in range(n_clusters):
        # Get indices of explanations in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
            
        # Calculate distance from each point to cluster center
        cluster_embeddings = reduced_embeddings[cluster_indices]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
        
        # Get explanation closest to center
        closest_idx = cluster_indices[np.argmin(distances)]
        centroids.append(explanations[closest_idx])
    
    return centroids

def find_best_explanation(pca_coords, explanations, n_clusters=3):
    """
    Find the best explanation based on the center of the densest cluster.
    
    This function identifies the most representative explanation by finding
    the densest cluster and selecting the explanation closest to its center.
    
    Args:
        pca_coords (np.ndarray): PCA-transformed coordinates of explanations
        explanations (list): Original text explanations
        n_clusters (int): Number of clusters for analysis
    
    Returns:
        tuple: (best_explanation, cluster_size) - the most representative 
               explanation and size of its cluster
    """
    from sklearn.cluster import KMeans
    import numpy as np
    
    # Apply K-means clustering to PCA coordinates
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_coords)
    
    # Find the densest (largest) cluster
    cluster_sizes = np.bincount(cluster_labels)
    densest_cluster = np.argmax(cluster_sizes)
    
    # Get points and explanations from the densest cluster
    cluster_center = kmeans.cluster_centers_[densest_cluster]
    cluster_points = pca_coords[cluster_labels == densest_cluster]
    cluster_explanations = [exp for i, exp in enumerate(explanations) if cluster_labels[i] == densest_cluster]
    
    # Calculate distances to cluster center
    distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
    best_idx = np.argmin(distances)
    
    return cluster_explanations[best_idx], cluster_sizes[densest_cluster]
    
    # Encontrar o ponto mais próximo do centro do cluster mais denso
    cluster_center = kmeans.cluster_centers_[densest_cluster]
    cluster_points = pca_coords[cluster_labels == densest_cluster]
    cluster_explanations = [exp for i, exp in enumerate(explanations) if cluster_labels[i] == densest_cluster]
    
    # Calcular distâncias ao centro
    distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
    best_idx = np.argmin(distances)
    
    return cluster_explanations[best_idx], cluster_sizes[densest_cluster]

def explain_prediction_with_multiple_prompts(image_path1, image_path2, decision, score, custom_prompts=None, num_prompts=5):
    """
    Generate comprehensive explanation using multiple specialized analysis prompts.
    
    This function creates diverse explanations by employing multiple specialized
    analysis perspectives (forensic, security, technical, etc.) and then uses
    clustering to identify the most representative insights.
    
    Args:
        image_path1 (str): Path to the first biometric image
        image_path2 (str): Path to the second biometric image  
        decision (str): Model's classification decision ("Genuine" or "Impostor")
        score (float): Confidence score from the model
        custom_prompts (list, optional): Custom analysis prompts to use
        num_prompts (int): Number of prompts to generate if custom_prompts not provided
    
    Returns:
        dict: Comprehensive analysis results including:
            - all_explanations: All generated explanations
            - centroid_explanations: Representative explanations from each cluster
            - best_explanation: Single best representative explanation
            - prompts_used: List of prompts that were used
            - num_generated: Total number of explanations generated
            - num_clusters: Number of semantic clusters found
    """
    try:
        explanations = []
        prompts_used = []
        
        if custom_prompts:
            # Use provided custom prompts
            for i, prompt in enumerate(custom_prompts):
                print(f"Generating explanation {i+1}/{len(custom_prompts)} with custom prompt...")
                explanation = generate_explanation_with_gpt4(image_path1, image_path2, decision, score, prompt)
                explanations.append(explanation)
                prompts_used.append(prompt)
        else:
            # Generate varied prompts automatically with different specialist perspectives
            prompt_variations = [
                """You are a forensic biometrics expert. Analyze this image pair classified as {decision} (score: {score:.4f}).
                Focus on: unique facial characteristics, image quality, potential security vulnerabilities.""",
                
                """As a biometric security analyst, examine these images classified as {decision} (score: {score:.4f}).
                Prioritize: technical capture aspects, lighting conditions, artifacts that may affect classification.""",
                
                """Facial recognition specialist, evaluate this pair classified as {decision} (score: {score:.4f}).
                Concentrate on: facial geometry, skin texture, distinctive features, analysis reliability.""",
                
                """Biometric systems auditor, analyze this classification {decision} (score: {score:.4f}).
                Examine: algorithm accuracy, possible false positives/negatives, improvement recommendations.""",
                
                """Digital forensic investigator, evaluate this result {decision} (score: {score:.4f}).
                Analyze: image authenticity, manipulation signs, evidence supporting the classification."""
            ]
            
            # Use as many prompts as requested (repeat if necessary)
            for i in range(num_prompts):
                prompt = prompt_variations[i % len(prompt_variations)]
                print(f"Generating explanation {i+1}/{num_prompts}...")
                explanation = generate_explanation_with_gpt4(image_path1, image_path2, decision, score, prompt)
                explanations.append(explanation)
                prompts_used.append(prompt)
        
        if not explanations:
            raise ValueError("Could not generate valid explanations")
        
        print(f"\nTotal explanations generated: {len(explanations)}")
        
        # Find cluster centroids using PCA
        centroid_explanations = find_cluster_centroids(explanations)
        print(f"Clusters found: {len(centroid_explanations)}")
        
        # Select the best explanation
        best_explanation = centroid_explanations[0] if centroid_explanations else explanations[0]
        
        return {
            "all_explanations": explanations,
            "centroid_explanations": centroid_explanations,
            "best_explanation": best_explanation,
            "prompts_used": prompts_used,
            "num_generated": len(explanations),
            "num_clusters": len(centroid_explanations)
        }
    
    except Exception as e:
        print(f"Error generating multiple explanations: {str(e)}")
        return {
            "all_explanations": [],
            "centroid_explanations": [],
            "best_explanation": f"""
## Basic Image Pair Analysis

### System Decision
- Classification: {decision.upper()}
- Confidence Score: {score:.4f}

### Note
Could not generate detailed explanations.
Error: {str(e)}
""",
            "prompts_used": [],
            "num_generated": 0,
            "num_clusters": 0
        }

def generate_and_visualize_explanations(image_path1, image_path2, decision, score, num_explanations=20):
    """
    Generate multiple explanations using GPT-4.1, apply PCA analysis, and create visualizations.
    
    This function performs comprehensive explanation analysis by generating multiple
    diverse explanations, computing their semantic embeddings, applying dimensionality
    reduction, and identifying the most representative explanations through clustering.
    
    Args:
        image_path1 (str): Path to the first biometric image
        image_path2 (str): Path to the second biometric image
        decision (str): Model's classification decision 
        score (float): Confidence score from the model
        num_explanations (int): Number of explanations to generate for analysis
    
    Returns:
        dict: Comprehensive analysis results including explanations, embeddings,
              PCA coordinates, cluster analysis, and visualization metadata
    """
    try:
        explanations = []
        
        # Generate multiple diverse explanations
        for i in range(num_explanations):
            logging.info(f"Generating explanation {i+1}/{num_explanations}")
            explanation = generate_explanation_with_gpt4(image_path1, image_path2, decision, score)
            explanations.append(explanation)
        
        if not explanations:
            raise ValueError("Could not generate valid explanations")
        
        # Compute semantic embeddings using SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(explanations)
        
        # Apply Principal Component Analysis for dimensionality reduction
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(embeddings)
        explained_variance = pca.explained_variance_ratio_
        
        # Find the best explanation using cluster analysis
        best_explanation, cluster_size = find_best_explanation(pca_coords, explanations)
        
        # Prepare comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "explanations": explanations,
            "best_explanation": best_explanation,
            "cluster_size": int(cluster_size),
            "embeddings": embeddings.tolist(),
            "pca_coordinates": pca_coords.tolist(),
            "explained_variance": explained_variance.tolist(),
            "metadata": {
                "num_explanations": len(explanations),
                "timestamp": timestamp,
                "decision": decision,
                "score": score,
                "model": "gpt-4.1"
            }
        }
        
        # Create results directory
        results_dir = os.path.join("results", "pca_analysis")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results to JSON file
        json_path = os.path.join(results_dir, f"pca_analysis_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Create PCA visualization
        plt.figure(figsize=(12, 10))
        
        # Plot all explanation points
        plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c='blue', alpha=0.6, label='Explanations')
        
        # Find and plot the densest cluster
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_coords)
        densest_cluster = np.argmax(np.bincount(cluster_labels))
        cluster_points = pca_coords[cluster_labels == densest_cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c='red', alpha=0.8, label='Densest Cluster')
        
        # Plot the center of the densest cluster
        cluster_center = kmeans.cluster_centers_[densest_cluster]
        plt.scatter(cluster_center[0], cluster_center[1], c='green', s=200, marker='*', label='Cluster Center')
        
        # Add labels for each point
        for i, (x, y) in enumerate(pca_coords):
            plt.annotate(f"Exp {i+1}", (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        # Add title and axis labels
        plt.title(f"PCA Analysis of GPT-4.1 Explanations\nDensest Cluster: {cluster_size} explanations")
        plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(results_dir, f"pca_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"PCA analysis completed. Results saved to {json_path}")
        logging.info(f"PCA visualization saved to {plot_path}")
        logging.info(f"Best explanation found in densest cluster ({cluster_size} explanations)")
        
        return results
        
    except Exception as e:
        logging.error(f"Error in PCA analysis: {str(e)}")
        raise

def demo_multiple_prompts(image_path1, image_path2, decision, score):
    """
    Demonstrate the use of multiple custom prompts for specialized analysis.
    
    This function showcases how different expert perspectives (forensic, medical,
    technical) can be applied to the same biometric verification task to generate
    diverse and comprehensive insights.
    
    Args:
        image_path1 (str): Path to the first biometric image
        image_path2 (str): Path to the second biometric image
        decision (str): Model's classification decision
        score (float): Confidence score from the model
    
    Returns:
        dict: Results from the multi-prompt analysis demonstration
    """
    # Example custom prompts for different expert perspectives
    custom_prompts = [
        """You are a forensic security expert. Analyze these biometric images classified as {decision} (score: {score:.4f}).
        Focus exclusively on: falsification signs, digital manipulation, image authenticity, forensic evidence.""",
        
        """As a medical specialist in facial anatomy, examine this pair classified as {decision} (score: {score:.4f}).
        Concentrate on: bone structures, facial proportions, unique anatomical features, biological variations.""",
        
        """Computer vision expert, evaluate this classification {decision} (score: {score:.4f}).
        Analyze: capture quality, resolution, compression artifacts, lighting conditions, image noise."""
    ]
    
    print("\n" + "="*60)
    print("DEMONSTRATION: ANALYSIS WITH MULTIPLE CUSTOM PROMPTS")
    print("="*60)
    
    # Generate explanations with custom prompts
    results = explain_prediction_with_multiple_prompts(
        image_path1, image_path2, decision, score, 
        custom_prompts=custom_prompts
    )
    
    print(f"\nRESULTS:")
    print(f"- Prompts used: {results['num_generated']}")
    print(f"- Clusters found: {results['num_clusters']}")
    
    print("\n" + "-"*50)
    print("INDIVIDUAL EXPLANATIONS:")
    print("-"*50)
    
    for i, explanation in enumerate(results['all_explanations']):
        print(f"\n### EXPLANATION {i+1} ###")
        print(explanation[:200] + "..." if len(explanation) > 200 else explanation)
        print()
    
    print("\n" + "-"*50)
    print("BEST EXPLANATION (BASED ON CLUSTERING):")
    print("-"*50)
    print(results['best_explanation'])
    
    return results

def main():
    """
    Main function for biometric pair prediction with AI-powered explanation system.
    
    This function provides a comprehensive command-line interface for performing
    biometric verification between image pairs, generating detailed explanations
    using GPT-4 Vision, and creating visualizations of the analysis results.
    
    Command-line Arguments:
        --image1: Path to the first biometric image
        --image2: Path to the second biometric image  
        --model: Path to the trained ResNet model weights
        --resnet: ResNet architecture type (resnet18, resnet34, resnet50)
        --threshold: Decision threshold for classification (default: 0.5)
        --output: Output directory for results and visualizations
        --api_key: OpenAI API key for GPT-4 Vision access
        --prompts: Number of analysis prompts to generate (default: 40)
        --demo_multiple: Enable demonstration of multiple custom prompts
    
    The system performs:
    1. Biometric verification using trained deep learning models
    2. AI-powered explanation generation with multiple expert perspectives
    3. Semantic clustering analysis of explanations
    4. Comprehensive visualization and reporting
    """
    # Configure command-line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='Biometric pair prediction and explanation using GPT-4 Vision')
    parser.add_argument('--image1', type=str, default=None, help='Path to the first image')
    parser.add_argument('--image2', type=str, default=None, help='Path to the second image')
    parser.add_argument('--model', type=str, default=None, help='Path to the trained model')
    parser.add_argument('--resnet', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], 
                        help='ResNet type used in the model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold (default: 0.5)')
    parser.add_argument('--output', type=str, default='./results', help='Output directory for results')
    parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY, 
                        help='OpenAI API key')
    parser.add_argument('--prompts', type=int, default=40, 
                        help='Number of unique prompts to generate (default: 5)')
    parser.add_argument('--demo_multiple', action='store_true', 
                        help='Demonstrate multiple custom prompts functionality')
    args = parser.parse_args()
    
    # Set default paths for demonstration (replace with your actual paths)
    image_path1 = args.image1 or "C:/Users/sanch/Computer Vision/Roi/0058/0058_23_10_2024_10_16_30_0_90_0_0_4_1_1_0_-1_0_2_0_6_0_192_1/frame_000004.jpg"
    image_path2 = args.image2 or "C:/Users/sanch/Computer Vision/Roi/0155/0155_19_09_2024_14_24_15_10_60_4_0_1_-1_-1_-1_-1_6_0_0_6_0_256_1/frame_000006.jpg"
    model_path = args.model or "C:/Users/sanch/Computer Vision/output/final_model_resnet50.pth"
    
    try:
        # Record start time for performance monitoring
        import time
        start_time = time.time()
        
        # Perform biometric verification
        decision, score = predict_image_pair(
            image_path1, image_path2, model_path, 
            args.resnet, args.threshold
        )
        
        # Create and save image comparison visualization
        os.makedirs(args.output, exist_ok=True)
        vis_path = os.path.join(args.output, "comparison.jpg")
        create_visualization(image_path1, image_path2, decision, score, vis_path)
        
        if args.demo_multiple:
            # Demonstrate multiple custom prompts functionality
            demo_results = demo_multiple_prompts(image_path1, image_path2, decision, score)
        else:
            # Use standard analysis with varied prompts
            print("\n" + "="*50)
            print("ANALYSIS WITH MULTIPLE VARIED PROMPTS")
            print("="*50)
            
            results = explain_prediction_with_multiple_prompts(
                image_path1, image_path2, decision, score, 
                num_prompts=args.prompts
            )
            
            print(f"\nClassification: {decision.upper()}")
            print(f"Confidence Score: {score:.4f}")
            print(f"Image Visualization: {vis_path}")
            print(f"Number of Generated Explanations: {results['num_generated']}")
            print(f"Clusters Found: {results['num_clusters']}")
            
            print("\n" + "-"*50)
            print("BEST EXPLANATION:")
            print("-"*50)
            print(results['best_explanation'])
        
        print("\n" + "-"*50)
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")
        print("="*50 + "\n")
        
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        print(f"\nERROR: {str(e)}")
        print("\nError details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()