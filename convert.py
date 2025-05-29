# Video Frame Extraction Module
# This module provides functionality to extract frames from video files
# for surveillance and biometric recognition applications

import os
import cv2

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extracts frames from a video file and saves them to the output folder.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Directory where extracted frames will be saved
        frame_rate (int): Number of frames to extract per second (default: 1)
    
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize video capture object
    cap = cv2.VideoCapture(video_path)
    count = 0  # Total frame counter
    frame_id = 0  # Extracted frame counter
    
    # Get video FPS to calculate frame extraction interval
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // frame_rate)  # Calculate interval between extracted frames
    
    # Process video frame by frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  # End of video reached
        
        # Extract frame only at specified intervals
        if count % frame_interval == 0:
            # Generate filename with zero-padded frame ID
            frame_filename = os.path.join(output_folder, f"frame_{frame_id:06d}.jpg")
            cv2.imwrite(frame_filename, frame)  # Save frame as JPEG
            frame_id += 1
        count += 1
    
    # Release video capture resources
    cap.release()
    print(f"Frame extraction completed for {video_path}: {frame_id} frames extracted.")

def process_videos(input_folder, output_folder, frame_rate=1):
    """
    Processes all video files in the input folder and extracts frames.
    
    Args:
        input_folder (str): Directory containing video files to process
        output_folder (str): Root directory where extracted frames will be organized
        frame_rate (int): Number of frames to extract per second (default: 1)
    
    Returns:
        None
    """
    # Create main output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each video file in the input folder
    for filename in os.listdir(input_folder):
        # Check if file has a supported video extension
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(input_folder, filename)
            
            # Create subfolder for this video's frames
            video_name = os.path.splitext(filename)[0]  # Remove file extension
            video_subfolder = os.path.join(output_folder, video_name)
            os.makedirs(video_subfolder, exist_ok=True)
            
            # Extract frames from current video
            extract_frames(video_path, video_subfolder, frame_rate)            
if __name__ == "__main__":
    # Configuration for frame extraction
    input_videos_folder = "C:/Users/Sancho/Computer Vision/data/CV_24_25_data/Video"  # Directory containing video files
    output_frames_folder = "frames"  # Directory where extracted frames will be saved
    frame_rate = 1  # Number of frames to extract per second
    
    # Start processing all videos in the input folder
    process_videos(input_videos_folder, output_frames_folder, frame_rate)
