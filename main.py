# Main Training and Testing Module for Biometric Recognition System
# This module provides the main entry point for training, testing, and evaluating
# the biometric recognition system using ResNet-based architectures

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import time
from tqdm import tqdm
import cv2

# Import custom modules for the biometric recognition pipeline
from create_pairs import create_frame_pairs
from dataset_fusion import FusedFramePairsDataset
from single_model import CombinedChannelModel

def create_simple_visualization(img1_np, img2_np, prediction):
    """
    Creates a simple visualization for comparing two images with prediction result.
    
    This function generates a composite image showing both input images side by side
    with the model's prediction and confidence score.
    
    Args:
        img1_np (numpy.ndarray): First image as numpy array (CHW format)
        img2_np (numpy.ndarray): Second image as numpy array (CHW format)
        prediction (float): Model prediction probability (0-1)
        
    Returns:
        numpy.ndarray: Composite image for visualization
    """
    # Denormalize images from [0,1] range to [0,255] for visualization
    img1_vis = (img1_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    img2_vis = (img2_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    
    # Convert from RGB to BGR format for OpenCV operations
    img1_vis = cv2.cvtColor(img1_vis, cv2.COLOR_RGB2BGR)
    img2_vis = cv2.cvtColor(img2_vis, cv2.COLOR_RGB2BGR)
    
    # Ensure consistent dimensions (256x128 with 2:1 aspect ratio)
    img1_vis = cv2.resize(img1_vis, (256, 128))
    img2_vis = cv2.resize(img2_vis, (256, 128))
        
    # Create composite canvas for side-by-side comparison
    width = 562  # 256 + 256 + 50 (margin between images)
    height = 200  # 128 + margin for text
    composite = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Add prediction decision and confidence score
    decision = "GENUINE" if prediction > 0.5 else "IMPOSTOR"
    label = f"{decision} (score: {prediction:.4f})"
    cv2.putText(composite, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 0, 0), 2)
    
    # Place images side by side with margin
    composite[50:50+128, 10:10+256] = img1_vis      # Left image
    composite[50:50+128, 296:296+256] = img2_vis    # Right image
    
    # Add image labels
    cv2.putText(composite, "Image 1", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 0, 0), 1)
    cv2.putText(composite, "Image 2", (296, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 0, 0), 1)
    
    # Add dimension information
    cv2.putText(composite, f"256x128 (2:1)", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 0, 0), 1)
    
    return composite

def plot_training_metrics(val_losses, val_accuracies, output_dir):
    """
    Plots and saves validation loss and accuracy metrics during training.
    
    This function creates a visualization of the training progress by plotting
    validation loss and accuracy curves over epochs.

    Args:
        val_losses (list): List of validation losses per epoch
        val_accuracies (list): List of validation accuracies per epoch
        output_dir (str): Directory where the plot will be saved
    """
    epochs = range(1, len(val_losses) + 1)
    
    # Create figure with appropriate size
    plt.figure(figsize=(10, 6))
    
    # Plot validation loss and accuracy on the same graph
    plt.plot(epochs, val_losses, label='Val Loss', color='red', linewidth=2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy', color='blue', linewidth=2)
    
    # Configure plot appearance
    plt.title('Validation Loss and Accuracy During Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save the plot to output directory
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'val_loss_accuracy.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to: {plot_path}")
    plt.close()

def main():
    """
    Main function that orchestrates the training, testing, or explanation process.
    
    This function handles command-line arguments, data preparation, model initialization,
    and coordinates the chosen operation mode (train/test/explain).
    """
    # Configure command line arguments for flexible operation
    parser = argparse.ArgumentParser(description='Biometric Recognition System with PyTorch ResNet')
    parser.add_argument('--roi_dir', required=True, help='Directory containing the ROIs organized by subject/video')
    parser.add_argument('--mode', choices=['train', 'test', 'explain'], required=True, help='Operation mode')
    parser.add_argument('--model_path', default=None, help='Path to pre-trained model (required for test/explain modes)')
    parser.add_argument('--num_pairs', type=int, default=1000, help='Number of image pairs to generate for training/testing')
    parser.add_argument('--genuine_ratio', type=float, default=0.5, help='Ratio of genuine pairs (0.0-1.0)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--resnet_type', default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'], 
                       help='ResNet architecture variant to use')
    parser.add_argument('--output_dir', default='./output', help='Directory to save results and model checkpoints')
    parser.add_argument('--continue_training', action='store_true', 
                        help='Continue training from a previous checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None, 
                        help='Path to checkpoint for continuing training')
    args = parser.parse_args()
    
    # Define hyperparameters
    learning_rate = 0.0001  # Conservative learning rate for stable training
    
    # Create output directory for saving results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set computing device (prefer GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Define normalization transforms for 6-channel input (2 RGB images concatenated)
    # Use ImageNet statistics repeated for both images
    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],  # RGB means for both images
        std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225]    # RGB stds for both images
    )
    
    # Generate image pairs for training/testing
    print(f"Generating {args.num_pairs} pairs with {args.genuine_ratio*100}% genuine ratio...")
    pairs = create_frame_pairs(args.roi_dir, num_pairs=args.num_pairs, genuine_ratio=args.genuine_ratio)
    
    # Create dataset with fused image pairs (6-channel tensors)
    dataset = FusedFramePairsDataset(pairs, transform=transform, output_shape=(256, 128))
    
    # Split dataset based on operation mode
    if args.mode == 'train':
        # Training mode: split into train/validation/test sets
        train_size = int(0.7 * len(dataset))    # 70% for training
        val_size = int(0.15 * len(dataset))     # 15% for validation
        test_size = len(dataset) - train_size - val_size  # 15% for testing
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        
        # Create data loaders with appropriate settings
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        print(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    else:
        # Test or explain mode: use entire dataset for evaluation
        test_dataset = dataset
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create or load model
    model = CombinedChannelModel(resnet_type=args.resnet_type)
    
    # Check if continuing training from a checkpoint
    if args.continue_training and args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        print("Checkpoint loaded successfully.")
    elif args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    model = model.to(device)
    
    val_losses = []
    val_accuracies = []
    
    # Training mode
    if args.mode == 'train':
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            
            # Training loop modifications - since we now receive a fused tensor directly
            for batch_idx, (fused_inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")):
                fused_inputs, targets = fused_inputs.to(device), targets.to(device)
                
                # Forward pass - now directly with fused inputs
                outputs = model(fused_inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track loss and accuracy
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == targets.unsqueeze(1)).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / len(train_dataset)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for fused_inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validate]"):
                    fused_inputs, targets = fused_inputs.to(device), targets.to(device)
                    
                    # Forward pass - now directly with fused inputs
                    outputs = model(fused_inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    
                    # Track loss and accuracy
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == targets.unsqueeze(1)).sum().item()
                    
                    # Store predictions and targets for metrics
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
                    
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / len(val_dataset)
            
            # Salvar métricas de validação
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Print metrics
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.2f}s")
            print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Save model if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(args.output_dir, f"best_model_{args.resnet_type}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
        
        # Plotar e salvar o gráfico de métricas de validação
        plot_training_metrics(val_losses, val_accuracies, args.output_dir)
        
        # Final testing phase
        model.eval()
        test_correct = 0
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for fused_inputs, targets in tqdm(test_loader, desc="Testing"):
                fused_inputs, targets = fused_inputs.to(device), targets.to(device)
                
                # Forward pass - now directly with fused inputs
                outputs = model(fused_inputs)
                
                # Track accuracy
                predicted = (outputs > 0.5).float()
                test_correct += (predicted == targets.unsqueeze(1)).sum().item()
                
                # Store predictions and targets for metrics
                test_predictions.extend(outputs.cpu().numpy())
                test_targets.extend(targets.cpu().numpy())
                
        test_accuracy = test_correct / len(test_dataset)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(test_targets, test_predictions)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(args.output_dir, 'roc_curve.png'))
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, f"final_model_{args.resnet_type}.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        
    # Test mode
    elif args.mode == 'test':
        if not args.model_path:
            print("Error: --model_path is required for test mode")
            return
            
        model.eval()
        test_correct = 0
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for fused_inputs, targets in tqdm(test_loader, desc="Testing"):
                fused_inputs, targets = fused_inputs.to(device), targets.to(device)
                
                # Forward pass - now directly with fused inputs
                outputs = model(fused_inputs)
                
                # Track accuracy
                predicted = (outputs > 0.5).float()
                test_correct += (predicted == targets.unsqueeze(1)).sum().item()
                
                # Store predictions and targets for metrics
                test_predictions.extend(outputs.cpu().numpy())
                test_targets.extend(targets.cpu().numpy())
                
        test_accuracy = test_correct / len(test_dataset)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(test_targets, test_predictions)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(args.output_dir, 'roc_curve.png'))
        
        # Save confusion matrix
        y_pred = (np.array(test_predictions) > 0.5).astype(int)
        cm = confusion_matrix(test_targets, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks([0, 1], ['Impostor', 'Genuine'])
        plt.yticks([0, 1], ['Impostor', 'Genuine'])
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
        
    # Explain mode
    elif args.mode == 'explain':
        if not args.model_path:
            print("Error: --model_path is required for explain mode")
            return
            
        model.eval()
        
        # Select a few samples to explain
        num_samples = min(10, len(test_dataset))
        sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            # Get sample
            img1, img2, target = test_dataset[idx]
            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)
            
            # Fuse images along channel dimension
            fused_input = torch.cat((img1, img2), dim=1)
            
            # Get prediction
            with torch.no_grad():
                output = model(fused_input)
                prediction = output.item()
                
            # Convert tensors to numpy arrays for visualization
            # For 6-channel images, use only the first 3 (RGB) for visualization
            img1_np = img1.cpu().numpy().squeeze()
            img2_np = img2.cpu().numpy().squeeze()
            
            if args.input_channels == 6:
                img1_np = img1_np[:3]  # Keep only RGB channels
                img2_np = img2_np[:3]  # Keep only RGB channels
            
            # Create visualization
            visualization = create_simple_visualization(img1_np, img2_np, prediction)
            
            # Save visualization
            vis_path = os.path.join(args.output_dir, f"sample_{i}_vis.jpg")
            cv2.imwrite(vis_path, visualization)
            print(f"Visualization saved to {vis_path}")
            
            # Ground truth info
            gt_label = "GENUINE" if target.item() == 1 else "IMPOSTOR"
            
            # Save basic information about the prediction
            info_path = os.path.join(args.output_dir, f"sample_{i}_info.txt")
            with open(info_path, 'w') as f:
                f.write(f"Ground Truth: {gt_label}\n")
                f.write(f"Prediction: {'GENUINE' if prediction > 0.5 else 'IMPOSTOR'}\n")
                f.write(f"Confidence Score: {prediction:.4f}\n")
                f.write(f"Correct: {'Yes' if (prediction > 0.5 and target.item() == 1) or (prediction <= 0.5 and target.item() == 0) else 'No'}\n")
            print(f"Basic information saved to {info_path}")

if __name__ == "__main__":
    main()