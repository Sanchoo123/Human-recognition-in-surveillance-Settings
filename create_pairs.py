import os
import numpy as np
import random
import cv2
from glob import glob

def create_frame_pairs(roi_folder, num_pairs=1000, genuine_ratio=0.5, cross_video_ratio=1.0):
    """
    Creates genuine and impostor pairs from individual frames for biometric recognition
    
    Args:
        roi_folder: Folder containing the ROIs organized by subject/video
        num_pairs: Total number of pairs to generate
        genuine_ratio: Ratio of genuine pairs (same person) to generate
        cross_video_ratio: Ratio of genuine pairs to be from different videos (always 1.0 now)
        
    Returns:
        List of tuples (frame1_path, frame2_path, label)
        where label is 1 for genuine and 0 for impostor
    """
    pairs = []
    
    # Forçar cross_video_ratio para 1.0 para garantir que todos os pares genuínos sejam de vídeos diferentes
    cross_video_ratio = 1.0
    
    # Calculate number of genuine and impostor pairs
    num_genuine = int(num_pairs * genuine_ratio)
    num_impostor = num_pairs - num_genuine
    
    print(f"Gerando {num_genuine} pares genuínos (todos de vídeos diferentes da mesma pessoa)")
    print(f"Gerando {num_impostor} pares impostores (pessoas diferentes)")
    
    # Get all subject directories
    subject_dirs = [d for d in os.listdir(roi_folder) if os.path.isdir(os.path.join(roi_folder, d))]
    
    if len(subject_dirs) < 2:
        raise ValueError(f"Necessário pelo menos 2 pessoas diferentes em {roi_folder} para criar pares impostores")
    
    # Dictionary to store frame paths by subject and video
    frames_by_subject = {}
    
    # Collect all frame paths organized by subject and video
    print("Coletando caminhos dos frames...")
    for subject_dir in subject_dirs:
        subject_path = os.path.join(roi_folder, subject_dir)
        frames_by_subject[subject_dir] = {}
        
        # Get all video directories for this subject
        video_dirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
        
        for video_dir in video_dirs:
            video_path = os.path.join(subject_path, video_dir)
            
            # Get all frames in this video
            frame_paths = sorted(glob(os.path.join(video_path, "*.jpg")))
            
            if frame_paths:
                frames_by_subject[subject_dir][video_dir] = frame_paths
    
    # Filtrar sujeitos que têm pelo menos dois vídeos diferentes para pares genuínos
    subjects_with_multiple_videos = [s for s in frames_by_subject if len(frames_by_subject[s]) > 1]
    print(f"Encontrados {len(subjects_with_multiple_videos)} sujeitos com múltiplos vídeos para pares genuínos")
    
    if len(subjects_with_multiple_videos) == 0:
        raise ValueError("Nenhum sujeito com múltiplos vídeos encontrado. Não é possível criar pares genuínos de vídeos diferentes.")
    
    # Generate genuine pairs (same person, different videos) - Agora só consideramos pares genuínos de vídeos diferentes
    print(f"Gerando {num_genuine} pares genuínos de vídeos diferentes...")
    genuine_count = 0
    
    while genuine_count < num_genuine:
        # Verificar se atingimos o máximo possível de pares genuínos
        if len(subjects_with_multiple_videos) == 0:
            print(f"Aviso: Apenas foi possível gerar {genuine_count} pares genuínos de vídeos diferentes.")
            break
        
        # Select a random subject that has multiple videos
        subject = random.choice(subjects_with_multiple_videos)
        
        # Get available videos for this subject
        available_videos = list(frames_by_subject[subject].keys())
        
        # Se não tiver pelo menos 2 vídeos com frames, remover este sujeito e continuar
        if len(available_videos) < 2:
            subjects_with_multiple_videos.remove(subject)
            continue
        
        # Select two different videos from this subject
        try:
            video1, video2 = random.sample(available_videos, 2)
        except ValueError:
            # Não conseguiu amostrar 2 vídeos diferentes, remover este sujeito
            subjects_with_multiple_videos.remove(subject)
            continue
        
        # Check if both videos have frames
        if not frames_by_subject[subject][video1] or not frames_by_subject[subject][video2]:
            continue
        
        # Select a random frame from each video
        frame1 = random.choice(frames_by_subject[subject][video1])
        frame2 = random.choice(frames_by_subject[subject][video2])
        
        # Add this cross-video genuine pair
        pairs.append((frame1, frame2, 1))
        genuine_count += 1
        
        if genuine_count % 100 == 0:
            print(f"Gerados {genuine_count}/{num_genuine} pares genuínos de vídeos diferentes")
    
    # Generate impostor pairs (different people)
    print(f"Gerando {num_impostor} pares impostores (pessoas diferentes)...")
    impostor_count = 0
    
    while impostor_count < num_impostor:
        # Select two different subjects
        subject1, subject2 = random.sample(list(frames_by_subject.keys()), 2)
        
        # Check if both subjects have videos with frames
        if not frames_by_subject[subject1] or not frames_by_subject[subject2]:
            continue
        
        # Select a random video from each subject
        video1 = random.choice(list(frames_by_subject[subject1].keys()))
        video2 = random.choice(list(frames_by_subject[subject2].keys()))
        
        # Check if both videos have frames
        if not frames_by_subject[subject1][video1] or not frames_by_subject[subject2][video2]:
            continue
        
        # Select a random frame from each video
        frame1 = random.choice(frames_by_subject[subject1][video1])
        frame2 = random.choice(frames_by_subject[subject2][video2])
        
        # Add this impostor pair
        pairs.append((frame1, frame2, 0))
        impostor_count += 1
        
        if impostor_count % 100 == 0:
            print(f"Gerados {impostor_count}/{num_impostor} pares impostores")
    
    # Shuffle the pairs
    random.shuffle(pairs)
    
    # Print summary statistics
    genuine_pairs = sum(1 for _, _, label in pairs if label == 1)
    impostor_pairs = sum(1 for _, _, label in pairs if label == 0)
    print(f"Total de pares gerados: {len(pairs)}")
    print(f"Pares genuínos: {genuine_pairs}, Pares impostores: {impostor_pairs}")
    
    return pairs