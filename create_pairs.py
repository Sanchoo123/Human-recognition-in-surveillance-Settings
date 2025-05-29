# Biometric Pair Creation Module
# This module creates genuine and impostor pairs from individual frames
# for training and testing biometric recognition systems

import os
import numpy as np
import random
import cv2
from glob import glob

def create_frame_pairs(roi_folder, num_pairs=1000, genuine_ratio=0.5, cross_video_ratio=1.0):
    """
    Creates genuine and impostor pairs from individual frames for biometric recognition.
    
    This function generates pairs of images where:
    - Genuine pairs: same person from different videos (cross-video verification)
    - Impostor pairs: different people from any videos
    
    Args:
        roi_folder (str): Folder containing the ROIs organized by subject/video
        num_pairs (int): Total number of pairs to generate (default: 1000)
        genuine_ratio (float): Ratio of genuine pairs (same person) to generate (default: 0.5)
        cross_video_ratio (float): Ratio of genuine pairs to be from different videos (forced to 1.0)
        
    Returns:
        list: List of tuples (frame1_path, frame2_path, label)
              where label is 1 for genuine and 0 for impostor
    """
    pairs = []
    
    # Force cross_video_ratio to 1.0 to ensure all genuine pairs are from different videos
    # This creates a more challenging verification scenario
    cross_video_ratio = 1.0
    
    # Calculate number of genuine and impostor pairs
    num_genuine = int(num_pairs * genuine_ratio)
    num_impostor = num_pairs - num_genuine
    
    print(f"Generating {num_genuine} genuine pairs (all from different videos of same person)")
    print(f"Generating {num_impostor} impostor pairs (different people)")
    
    # Get all subject directories from the ROI folder
    subject_dirs = [d for d in os.listdir(roi_folder) if os.path.isdir(os.path.join(roi_folder, d))]
    
    # Validate that we have enough subjects for creating impostor pairs
    if len(subject_dirs) < 2:
        raise ValueError(f"Need at least 2 different people in {roi_folder} to create impostor pairs")
    
    # Dictionary to store frame paths organized by subject and video
    frames_by_subject = {}
    
    # Collect all frame paths organized by subject and video
    print("Collecting frame paths...")
    for subject_dir in subject_dirs:
        subject_path = os.path.join(roi_folder, subject_dir)
        frames_by_subject[subject_dir] = {}
        
        # Get all video directories for this subject
        video_dirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
        
        for video_dir in video_dirs:
            video_path = os.path.join(subject_path, video_dir)
            
            # Get all frames in this video directory
            frame_paths = sorted(glob(os.path.join(video_path, "*.jpg")))
            
            # Only store if frames are found
            if frame_paths:
                frames_by_subject[subject_dir][video_dir] = frame_paths
    
    # Filter subjects that have at least two different videos for genuine pairs
    subjects_with_multiple_videos = [s for s in frames_by_subject if len(frames_by_subject[s]) > 1]
    print(f"Found {len(subjects_with_multiple_videos)} subjects with multiple videos for genuine pairs")
    
    # Validate that we can create cross-video genuine pairs
    if len(subjects_with_multiple_videos) == 0:
        raise ValueError("No subjects with multiple videos found. Cannot create cross-video genuine pairs.")
    
    # Generate genuine pairs (same person, different videos)
    print(f"Generating {num_genuine} genuine pairs from different videos...")
    genuine_count = 0
    
    while genuine_count < num_genuine:
        # Check if we've exhausted all possible genuine pairs
        if len(subjects_with_multiple_videos) == 0:
            print(f"Warning: Could only generate {genuine_count} cross-video genuine pairs.")
            break
        
        # Select a random subject that has multiple videos
        subject = random.choice(subjects_with_multiple_videos)
        
        # Get available videos for this subject
        available_videos = list(frames_by_subject[subject].keys())
        
        # If subject doesn't have at least 2 videos with frames, remove and continue
        if len(available_videos) < 2:
            subjects_with_multiple_videos.remove(subject)
            continue
        
        # Select two different videos from this subject
        try:
            video1, video2 = random.sample(available_videos, 2)
        except ValueError:
            # Couldn't sample 2 different videos, remove this subject
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
        
        # Progress tracking
        if genuine_count % 100 == 0:
            print(f"Generated {genuine_count}/{num_genuine} cross-video genuine pairs")
    
    # Generate impostor pairs (different people)
    print(f"Generating {num_impostor} impostor pairs (different people)...")
    impostor_count = 0
    
    while impostor_count < num_impostor:
        # Select two different subjects randomly
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
        
        # Progress tracking
        if impostor_count % 100 == 0:
            print(f"Generated {impostor_count}/{num_impostor} impostor pairs")
    
    # Shuffle the pairs to randomize the order
    random.shuffle(pairs)
    
    # Print summary statistics
    genuine_pairs = sum(1 for _, _, label in pairs if label == 1)
    impostor_pairs = sum(1 for _, _, label in pairs if label == 0)
    print(f"Total pairs generated: {len(pairs)}")
    print(f"Genuine pairs: {genuine_pairs}, Impostor pairs: {impostor_pairs}")
    
    return pairs