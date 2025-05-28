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

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


DEFAULT_API_KEY = "yourkey"

def encode_image_to_base64(image_path):
    """
    Codifica uma imagem para base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_explanation_with_gpt4(image_path1, image_path2, decision, score, custom_prompt=None):
    """
    Gera uma explicação usando o GPT-4.1 com roleplay de especialista.
    """
    try:
        # Codificar imagens em base64
        base64_image1 = encode_image_to_base64(image_path1)
        base64_image2 = encode_image_to_base64(image_path2)
        
        # Usar prompt customizado se fornecido, senão usar o padrão
        if custom_prompt:
            prompt = custom_prompt.format(decision=decision, score=score)
        else:
            # Preparar o prompt padrão com roleplay
            prompt = f"""Você é um especialista em análise biométrica e segurança, com anos de experiência em análise forense de imagens. 
            Sua tarefa é analisar este par de imagens biométricas que foram classificadas como {decision} com um score de {score:.4f}.
            
            Como especialista, forneça uma análise detalhada e técnica considerando:
            
            antes de comecares estes pontso quero uma descricao do genero da pessoa , do ambiente das cores da roupa, estilo de cabelo etc 
            1. Análise Forense:
               - Características biométricas similares ou diferentes
               - Qualidade e resolução das imagens
               - Possíveis artefatos ou distorções
            
            2. Análise de Segurança:
               - Nível de confiança na classificação
               - Possíveis pontos de vulnerabilidade
               - Recomendações para verificação adicional
            
            3. Análise Técnica:
               - Detalhes específicos que levaram à classificação
               - Fatores que influenciaram o score
               - Limitações da análise atual
            
            Mantenha um tom profissional e técnico, mas acessível. Use terminologia especializada quando apropriado."""
        
        # Configurar a chamada da API
        client = openai.OpenAI(api_key=DEFAULT_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "Você é um especialista em análise biométrica e segurança, com profundo conhecimento em análise forense de imagens e sistemas de autenticação biométrica."
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
        logging.error(f"Erro ao gerar explicação com GPT-4.1: {str(e)}")
        raise

def predict_image_pair(image_path1, image_path2, model_path, resnet_type='resnet50', threshold=0.5):
    """
    Faz a predição entre um par de imagens e retorna se é genuíno ou impostor.
    
    Args:
        image_path1: Caminho da primeira imagem.
        image_path2: Caminho da segunda imagem.
        model_path: Caminho para o modelo treinado.
        resnet_type: Tipo de ResNet usado no modelo (ex.: resnet18, resnet50).
        threshold: Limiar para decisão (default: 0.5).
    
    Returns:
        (str, float): Tupla com a decisão ("Genuine" ou "Impostor") e o score.
    """
    # Verificar se os arquivos existem
    if not os.path.exists(image_path1) or not os.path.exists(image_path2):
        raise FileNotFoundError("Um ou ambos os caminhos das imagens não existem.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("O modelo especificado não foi encontrado.")
    
    # Configurar o dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carregar o modelo
    model = CombinedChannelModel(resnet_type=resnet_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Carregar e processar as imagens
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    if img1 is None or img2 is None:
        raise ValueError("Não foi possível carregar uma ou ambas as imagens.")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Criar tensor fundido
    fused_tensor = create_fused_instance(img1, img2, output_shape=(256, 128))
    
    # Normalizar o tensor
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
    )
    fused_tensor = normalize(fused_tensor)
    
    # Fazer a predição
    with torch.no_grad():
        fused_input = fused_tensor.unsqueeze(0).to(device)
        prediction = model(fused_input).item()
    
    # Decisão baseada no limiar
    decision = "Genuine" if prediction > threshold else "Impostor"
    return decision, prediction

def create_visualization(image_path1, image_path2, decision, score, output_path):
    """
    Cria uma visualização do par de imagens com o resultado da predição.
    
    Args:
        image_path1: Caminho da primeira imagem.
        image_path2: Caminho da segunda imagem.
        decision: Decisão do modelo ("Genuine" ou "Impostor").
        score: Pontuação de confiança.
        output_path: Caminho para salvar a visualização.
    """
    # Carregar imagens
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    if img1 is None or img2 is None:
        raise ValueError("Não foi possível carregar uma ou ambas as imagens.")
    
    # Redimensionar mantendo proporção
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    target_height = 300
    img1 = cv2.resize(img1, (int(w1 * target_height / h1), target_height))
    img2 = cv2.resize(img2, (int(w2 * target_height / h2), target_height))
    
    # Criar espaço em branco entre as imagens (50 pixels)
    h_max = max(img1.shape[0], img2.shape[0])
    w_total = img1.shape[1] + img2.shape[1] + 50
    
    # Criar tela para visualização com espaço para texto
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
    Filtra apenas as imagens que são mais similares à imagem de referência.
    
    Args:
        image_path1: Caminho da primeira imagem (referência)
        image_path2: Caminho da segunda imagem
        similar_inputs: Lista de caminhos de imagens similares
        similarity_threshold: Limiar de similaridade (default: 0.7)
    
    Returns:
        list: Lista filtrada de caminhos de imagens similares
    """
    if not similar_inputs:
        return []
    
    # Calcular similaridade com a imagem de referência
    similarities = []
    for img_path in similar_inputs:
        sim_score = get_image_similarity(image_path1, img_path)
        similarities.append((img_path, sim_score))
    
    # Ordenar por similaridade e filtrar pelo threshold
    similarities.sort(key=lambda x: x[1], reverse=True)
    filtered_images = [img for img, sim in similarities if sim >= similarity_threshold]
    
    # Retornar no máximo as 3 imagens mais similares
    return filtered_images[:3]

def filter_similar_explanations(explanations, similarity_threshold=0.7):
    """
    Filtra apenas as explicações que são mais similares entre si.
    
    Args:
        explanations: Lista de explicações geradas pelo LVLM
        similarity_threshold: Limiar de similaridade (default: 0.7)
    
    Returns:
        list: Lista filtrada de explicações mais similares
    """
    if not explanations or len(explanations) <= 1:
        return explanations
    
    # Carregar modelo de embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calcular embeddings para todas as explicações
    embeddings = model.encode(explanations)
    
    # Calcular matriz de similaridade
    similarity_matrix = cosine_similarity(embeddings)
    
    # Encontrar grupos de explicações similares
    similar_groups = []
    used_indices = set()
    
    for i in range(len(explanations)):
        if i in used_indices:
            continue
            
        # Encontrar explicações similares à atual
        similar_indices = [j for j in range(len(explanations)) 
                         if similarity_matrix[i][j] >= similarity_threshold 
                         and j not in used_indices]
        
        if similar_indices:
            # Adicionar ao grupo
            group = [explanations[j] for j in similar_indices]
            similar_groups.append(group)
            used_indices.update(similar_indices)
    
    # Selecionar a explicação mais representativa de cada grupo
    filtered_explanations = []
    for group in similar_groups:
        # Usar a primeira explicação do grupo como representante
        filtered_explanations.append(group[0])
    
    return filtered_explanations

def find_cluster_centroids(explanations, n_components=2, n_clusters=3):
    """
    Encontra os centroides dos clusters de explicações usando PCA e K-means.
    
    Args:
        explanations: Lista de explicações geradas pelo LVLM
        n_components: Número de componentes para PCA
        n_clusters: Número de clusters para K-means
    
    Returns:
        list: Lista de explicações centroides (uma por cluster)
    """
    if not explanations or len(explanations) <= 1:
        return explanations
    
    # Carregar modelo de embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calcular embeddings para todas as explicações
    embeddings = model.encode(explanations)
    
    # Aplicar PCA para reduzir dimensionalidade
    pca = PCA(n_components=min(n_components, len(explanations)-1))
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Aplicar K-means para encontrar clusters
    n_clusters = min(n_clusters, len(explanations))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    
    # Encontrar o centroide mais próximo de cada cluster
    centroids = []
    for cluster_id in range(n_clusters):
        # Pegar índices das explicações neste cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
            
        # Calcular distância de cada ponto ao centro do cluster
        cluster_embeddings = reduced_embeddings[cluster_indices]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
        
        # Pegar a explicação mais próxima do centro
        closest_idx = cluster_indices[np.argmin(distances)]
        centroids.append(explanations[closest_idx])
    
    return centroids

def find_best_explanation(pca_coords, explanations, n_clusters=3):
    """
    Encontra a melhor explicação baseada no centro do cluster mais denso.
    """
    from sklearn.cluster import KMeans
    import numpy as np
    
    # Aplicar K-means para encontrar clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_coords)
    
    # Encontrar o cluster mais denso
    cluster_sizes = np.bincount(cluster_labels)
    densest_cluster = np.argmax(cluster_sizes)
    
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
    Explica a predição usando múltiplos prompts personalizados ou gerados automaticamente.
    
    Args:
        image_path1: Caminho da primeira imagem
        image_path2: Caminho da segunda imagem  
        decision: Decisão do modelo ("Genuine" ou "Impostor")
        score: Score de confiança
        custom_prompts: Lista de prompts personalizados (opcional)
        num_prompts: Número de prompts a gerar se custom_prompts não fornecido
    
    Returns:
        dict: Dicionário com todas as explicações e análise
    """
    try:
        explanations = []
        prompts_used = []
        
        if custom_prompts:
            # Usar prompts personalizados fornecidos
            for i, prompt in enumerate(custom_prompts):
                print(f"Gerando explicação {i+1}/{len(custom_prompts)} com prompt personalizado...")
                explanation = generate_explanation_with_gpt4(image_path1, image_path2, decision, score, prompt)
                explanations.append(explanation)
                prompts_used.append(prompt)
        else:
            # Gerar prompts variados automaticamente
            prompt_variations = [
                """Você é um especialista forense em biometria. Analise este par de imagens classificado como {decision} (score: {score:.4f}).
                Foque em: características faciais únicas, qualidade das imagens, possíveis vulnerabilidades de segurança.""",
                
                """Como analista de segurança biométrica, examine estas imagens classificadas como {decision} (score: {score:.4f}).
                Priorize: aspectos técnicos da captura, condições de iluminação, artefatos que possam afetar a classificação.""",
                
                """Especialista em reconhecimento facial, avalie este par classificado como {decision} (score: {score:.4f}).
                Concentre-se em: geometria facial, textura da pele, características distintivas, confiabilidade da análise.""",
                
                """Auditor de sistemas biométricos, analise esta classificação {decision} (score: {score:.4f}).
                Examine: precisão do algoritmo, possíveis falsos positivos/negativos, recomendações de melhoria.""",
                
                """Investigador forense digital, avalie este resultado {decision} (score: {score:.4f}).
                Analise: autenticidade das imagens, sinais de manipulação, evidências que suportam a classificação."""
            ]
            
            # Usar tantos prompts quantos solicitados (repetir se necessário)
            for i in range(num_prompts):
                prompt = prompt_variations[i % len(prompt_variations)]
                print(f"Gerando explicação {i+1}/{num_prompts}...")
                explanation = generate_explanation_with_gpt4(image_path1, image_path2, decision, score, prompt)
                explanations.append(explanation)
                prompts_used.append(prompt)
        
        if not explanations:
            raise ValueError("Não foi possível gerar explicações válidas")
        
        print(f"\nTotal de explicações geradas: {len(explanations)}")
        
        # Encontrar centroides dos clusters usando PCA
        centroid_explanations = find_cluster_centroids(explanations)
        print(f"Clusters encontrados: {len(centroid_explanations)}")
        
        # Selecionar a melhor explicação
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
        print(f"Erro ao gerar explicações múltiplas: {str(e)}")
        return {
            "all_explanations": [],
            "centroid_explanations": [],
            "best_explanation": f"""
## Análise Básica do Par de Imagens

### Decisão do Sistema
- Classificação: {decision.upper()}
- Pontuação de confiança: {score:.4f}

### Observação
Não foi possível gerar explicações detalhadas.
Erro: {str(e)}
""",
            "prompts_used": [],
            "num_generated": 0,
            "num_clusters": 0
        }

def generate_and_visualize_explanations(image_path1, image_path2, decision, score, num_explanations=20):
    """
    Gera múltiplas explicações usando GPT-4.1, aplica PCA e cria visualização.
    """
    try:
        explanations = []
        
        # Gerar múltiplas explicações
        for i in range(num_explanations):
            logging.info(f"Gerando explicação {i+1}/{num_explanations}")
            explanation = generate_explanation_with_gpt4(image_path1, image_path2, decision, score)
            explanations.append(explanation)
        
        if not explanations:
            raise ValueError("Não foi possível gerar explicações válidas")
        
        # Computar embeddings usando SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(explanations)
        
        # Aplicar PCA
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(embeddings)
        explained_variance = pca.explained_variance_ratio_
        
        # Encontrar a melhor explicação
        best_explanation, cluster_size = find_best_explanation(pca_coords, explanations)
        
        # Preparar resultados
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
        
        # Criar diretório para resultados
        results_dir = os.path.join("results", "pca_analysis")
        os.makedirs(results_dir, exist_ok=True)
        
        # Salvar resultados em JSON
        json_path = os.path.join(results_dir, f"pca_analysis_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Criar visualização PCA
        plt.figure(figsize=(12, 10))
        
        # Plotar todos os pontos
        plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c='blue', alpha=0.6, label='Explicações')
        
        # Encontrar e plotar o cluster mais denso
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_coords)
        densest_cluster = np.argmax(np.bincount(cluster_labels))
        cluster_points = pca_coords[cluster_labels == densest_cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c='red', alpha=0.8, label='Cluster Mais Denso')
        
        # Plotar o centro do cluster mais denso
        cluster_center = kmeans.cluster_centers_[densest_cluster]
        plt.scatter(cluster_center[0], cluster_center[1], c='green', s=200, marker='*', label='Centro do Cluster')
        
        # Adicionar rótulos para cada ponto
        for i, (x, y) in enumerate(pca_coords):
            plt.annotate(f"Exp {i+1}", (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        # Adicionar título e labels
        plt.title(f"PCA das Explicações GPT-4.1\nCluster Mais Denso: {cluster_size} explicações")
        plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")
        plt.legend()
        
        # Salvar plot
        plot_path = os.path.join(results_dir, f"pca_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Análise PCA concluída. Resultados salvos em {json_path}")
        logging.info(f"Visualização PCA salva em {plot_path}")
        logging.info(f"Melhor explicação encontrada no cluster mais denso ({cluster_size} explicações)")
        
        return results
        
    except Exception as e:
        logging.error(f"Erro na análise PCA: {str(e)}")
        raise

def demo_multiple_prompts(image_path1, image_path2, decision, score):
    """
    Demonstra o uso de múltiplos prompts personalizados para análise.
    """
    # Exemplo de prompts personalizados
    custom_prompts = [
        """Você é um especialista em segurança forense. Analise estas imagens biométricas classificadas como {decision} (score: {score:.4f}).
        Foque exclusivamente em: sinais de falsificação, manipulação digital, autenticidade das imagens, evidências forenses.""",
        
        """Como médico especialista em anatomia facial, examine este par classificado como {decision} (score: {score:.4f}).
        Concentre-se em: estruturas ósseas, proporções faciais, características anatômicas únicas, variações biológicas.""",
        
        """Especialista em visão computacional, avalie esta classificação {decision} (score: {score:.4f}).
        Analise: qualidade da captura, resolução, artefatos de compressão, condições de iluminação, ruído nas imagens."""
    ]
    
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO: ANÁLISE COM MÚLTIPLOS PROMPTS PERSONALIZADOS")
    print("="*60)
    
    # Gerar explicações com prompts personalizados
    results = explain_prediction_with_multiple_prompts(
        image_path1, image_path2, decision, score, 
        custom_prompts=custom_prompts
    )
    
    print(f"\nRESULTADOS:")
    print(f"- Prompts utilizados: {results['num_generated']}")
    print(f"- Clusters encontrados: {results['num_clusters']}")
    
    print("\n" + "-"*50)
    print("EXPLICAÇÕES INDIVIDUAIS:")
    print("-"*50)
    
    for i, explanation in enumerate(results['all_explanations']):
        print(f"\n### EXPLICAÇÃO {i+1} ###")
        print(explanation[:200] + "..." if len(explanation) > 200 else explanation)
        print()
    
    print("\n" + "-"*50)
    print("MELHOR EXPLICAÇÃO (BASEADA EM CLUSTERING):")
    print("-"*50)
    print(results['best_explanation'])
    
    return results

def main():
    # Argumentos via linha de comando
    import argparse
    parser = argparse.ArgumentParser(description='Predição e explicação biométrica de pares de imagens usando GPT-4 Vision')
    parser.add_argument('--image1', type=str, default=None, help='Caminho para a primeira imagem')
    parser.add_argument('--image2', type=str, default=None, help='Caminho para a segunda imagem')
    parser.add_argument('--model', type=str, default=None, help='Caminho para o modelo treinado')
    parser.add_argument('--resnet', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], 
                        help='Tipo de ResNet usado no modelo')
    parser.add_argument('--threshold', type=float, default=0.5, help='Limiar para decisão (default: 0.5)')
    parser.add_argument('--output', type=str, default='./results', help='Diretório de saída para resultados')
    parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY, 
                        help='Chave da API OpenAI')
    parser.add_argument('--prompts', type=int, default=40, 
                        help='Número de prompts únicos a gerar (default: 5)')
    parser.add_argument('--demo_multiple', action='store_true', 
                        help='Demonstrar funcionalidade de múltiplos prompts personalizados')
    args = parser.parse_args()
    
    # Caminhos padrão
    image_path1 = args.image1 or "C:/Users/sanch/Computer Vision/Roi/0058/0058_23_10_2024_10_16_30_0_90_0_0_4_1_1_0_-1_0_2_0_6_0_192_1/frame_000004.jpg"
    image_path2 = args.image2 or "C:/Users/sanch/Computer Vision/Roi/0155/0155_19_09_2024_14_24_15_10_60_4_0_1_-1_-1_-1_-1_6_0_0_6_0_256_1/frame_000006.jpg"
    model_path = args.model or "C:/Users/sanch/Computer Vision/output/final_model_resnet50.pth"
    
    try:
        # Registrar hora de início
        import time
        start_time = time.time()
        
        # Obter decisão e score
        decision, score = predict_image_pair(
            image_path1, image_path2, model_path, 
            args.resnet, args.threshold
        )
        
        # Criar e salvar visualização das imagens
        os.makedirs(args.output, exist_ok=True)
        vis_path = os.path.join(args.output, "comparison.jpg")
        create_visualization(image_path1, image_path2, decision, score, vis_path)
        
        if args.demo_multiple:
            # Demonstrar funcionalidade de múltiplos prompts
            demo_results = demo_multiple_prompts(image_path1, image_path2, decision, score)
        else:
            # Usar análise padrão com prompts variados
            print("\n" + "="*50)
            print("ANÁLISE COM MÚLTIPLOS PROMPTS VARIADOS")
            print("="*50)
            
            results = explain_prediction_with_multiple_prompts(
                image_path1, image_path2, decision, score, 
                num_prompts=args.prompts
            )
            
            print(f"\nClassificação: {decision.upper()}")
            print(f"Score de Confiança: {score:.4f}")
            print(f"Visualização das Imagens: {vis_path}")
            print(f"Número de Explicações Geradas: {results['num_generated']}")
            print(f"Clusters Encontrados: {results['num_clusters']}")
            
            print("\n" + "-"*50)
            print("MELHOR EXPLICAÇÃO:")
            print("-"*50)
            print(results['best_explanation'])
        
        print("\n" + "-"*50)
        print(f"Tempo total de processamento: {time.time() - start_time:.2f} segundos")
        print("="*50 + "\n")
        
    except Exception as e:
        logging.error(f"Erro durante a execução: {str(e)}")
        print(f"\nERRO: {str(e)}")
        print("\nDetalhes do erro:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()