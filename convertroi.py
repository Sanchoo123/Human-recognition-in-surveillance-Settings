import os
import cv2

def extract_frames(video_path, output_folder, frame_rate=1):
    """ Extrai frames de um vídeo e salva-os na pasta de saída."""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // frame_rate)  # Intervalo entre frames
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_id:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_id += 1
        count += 1
    
    cap.release()
    print(f"Extração concluída para {video_path}: {frame_id} frames extraídos.")

def process_videos(input_folder, output_folder, frame_rate=1):
    """ Processa todos os vídeos da pasta de entrada, mantendo a estrutura de ID. """
    os.makedirs(output_folder, exist_ok=True)
    
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if os.path.isdir(folder_path):  # Confirma que é uma pasta
            id_folder = os.path.join(output_folder, folder_name)
            os.makedirs(id_folder, exist_ok=True)
            
            for filename in os.listdir(folder_path):
                if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_path = os.path.join(folder_path, filename)
                    video_name = os.path.splitext(filename)[0]
                    video_subfolder = os.path.join(id_folder, video_name)
                    os.makedirs(video_subfolder, exist_ok=True)
                    extract_frames(video_path, video_subfolder, frame_rate)
            
if __name__ == "__main__":
    input_videos_folder = "C:/Users/Sancho/Computer Vision/data/CV_24_25_data/ROIs"  # Pasta com os vídeos organizados por ID
    output_frames_folder = "Roi"  # Pasta onde os frames serão salvos
    frame_rate = 1  # Número de frames por segundo
    
    process_videos(input_videos_folder, output_frames_folder, frame_rate)