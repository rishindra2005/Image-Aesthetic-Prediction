#!/usr/bin/env python3
"""
Image Aesthetic Prediction Demo
===============================

This script demonstrates how to use the fine-tuned CLIP model
to predict the aesthetic score of images.
"""

import os
import torch
import clip
import argparse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

def load_model(model_path):
    """Load the fine-tuned CLIP model."""
    try:
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using base CLIP model instead...")
        model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        return model, preprocess

def predict_aesthetic_score(model, processor, image_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Predict the aesthetic score for an image."""
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare text prompts for aesthetic scoring
        text_prompts = [f"This image has an aesthetic score of {i}" for i in range(1, 11)]
        
        # Process inputs
        if isinstance(processor, CLIPProcessor):
            # Using Hugging Face transformers model
            inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
        else:
            # Using OpenAI CLIP model
            image_input = processor(image).unsqueeze(0).to(device)
            text_inputs = torch.cat([clip.tokenize(prompt) for prompt in text_prompts]).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                logits_per_image = (100.0 * image_features @ text_features.T)
        
        # Calculate the weighted average score
        scores = torch.arange(1, 11).float().to(device)
        probabilities = logits_per_image.softmax(dim=-1)
        weighted_score = (probabilities * scores).sum().item()
        
        return weighted_score, probabilities.cpu().numpy()
    
    except Exception as e:
        print(f"Error predicting score: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Predict aesthetic scores for images using CLIP")
    parser.add_argument("--model_path", type=str, default="./fine_tuned_clip", 
                        help="Path to the fine-tuned model")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the image for prediction")
    args = parser.parse_args()
    
    # Check if the image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} not found.")
        return
    
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Try to load the fine-tuned model, fall back to base CLIP model if needed
    if os.path.exists(args.model_path):
        model, processor = load_model(args.model_path)
    else:
        print(f"Model path {args.model_path} not found, using base CLIP model")
        model, processor = clip.load("ViT-B/32", device=device)
    
    # Predict the aesthetic score
    score, probabilities = predict_aesthetic_score(model, processor, args.image_path, device)
    
    if score is not None:
        print(f"\nImage: {args.image_path}")
        print(f"Predicted Aesthetic Score: {score:.2f}/10.0")
        
        # Print score distribution
        if probabilities is not None:
            print("\nScore Distribution:")
            for i, prob in enumerate(probabilities[0], 1):
                print(f"  Score {i}: {prob*100:.2f}%")
    
if __name__ == "__main__":
    main() 