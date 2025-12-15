"""
Feedback Viewer Module

This module provides functions for viewing and displaying feedback entries
from Hugging Face Datasets.
"""

import logging
import math
import os
from typing import List, Optional
from io import BytesIO
from datetime import datetime
import tempfile
import uuid
import base64
from datasets import Dataset

import gradio as gr
from PIL import Image

from ipadapter_model import create_image_grid

# Hugging Face Datasets imports
try:
    from datasets import load_dataset  # type: ignore
    from huggingface_hub import login  # type: ignore
    HF_DATASETS_AVAILABLE = True
except ImportError:
    load_dataset = None  # type: ignore
    login = None  # type: ignore
    HF_DATASETS_AVAILABLE = False
    logging.warning("Hugging Face datasets not available. Feedback viewer will not work.")

# Configuration - can be overridden
HF_FEEDBACK_DATASET_REPO = os.getenv("HF_FEEDBACK_DATASET_REPO", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)

def store_feedback_to_hf_dataset(
    rating: str,
    feedback_text: str,
    alpha_start: float,
    alpha_end: float,
    n_steps: int,
    input1_image: Optional[Image.Image],
    input2_image: Optional[Image.Image],
    extra_images: Optional[List[Image.Image]],
    negative_images: Optional[List[Image.Image]],
    blending_result_images: Optional[List[Image.Image]],
    is_public: bool = True,
    dataset_repo: Optional[str] = None,
    token: Optional[str] = None
) -> bool:
    """
    Store feedback and images to a Hugging Face Dataset.
    
    Uses concatenate_datasets to append new entries without re-processing existing ones.
    This preserves existing images correctly.
    
    Args:
        rating: User rating (1-5)
        feedback_text: User feedback text
        alpha_start: Start alpha value used
        alpha_end: End alpha value used
        n_steps: Number of output images
        input1_image: First input image (PIL Image)
        input2_image: Second input image (PIL Image)
        extra_images: List of extra images (PIL Images)
        negative_images: List of negative images (PIL Images)
        blending_result_images: List of blending result images (PIL Images)
        is_public: Whether the feedback should be publicly visible (default True)
        dataset_repo: Hugging Face dataset repository (username/dataset-name)
        token: Hugging Face token (if None, will try to use HF_TOKEN env var)
    
    Returns:
        True if feedback was stored successfully, False otherwise
    """
    if not HF_DATASETS_AVAILABLE:
        logging.warning("Hugging Face datasets library not available")
        return False
    
    if dataset_repo is None:
        dataset_repo = HF_FEEDBACK_DATASET_REPO
    
    if dataset_repo is None:
        logging.warning("HF_FEEDBACK_DATASET_REPO not set. Set it to your Hugging Face username/dataset-name")
        return False
    
    # Validate that input1 and input2 images are not empty
    if input1_image is None:
        error_msg = "Input 1 image cannot be empty. Please provide a valid image."
        logging.error(error_msg)
        raise ValueError(error_msg)
    if input2_image is None:
        error_msg = "Input 2 image cannot be empty. Please provide a valid image."
        logging.error(error_msg)
        raise ValueError(error_msg)
    # Check if image is actually empty (size 0x0)
    if hasattr(input1_image, 'size') and input1_image.size == (0, 0):
        error_msg = "Input 1 image is empty (0x0 size). Please provide a valid image."
        logging.error(error_msg)
        raise ValueError(error_msg)
    if hasattr(input2_image, 'size') and input2_image.size == (0, 0):
        error_msg = "Input 2 image is empty (0x0 size). Please provide a valid image."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Type guards - these should not be None if HF_DATASETS_AVAILABLE is True
        if Dataset is None or load_dataset is None or login is None:
            logging.error("Hugging Face datasets libraries not properly imported")
            return False
        
        from datasets import Features, Image as ImageFeature, Value, Sequence, concatenate_datasets
        
        # Use token from parameter, environment variable, or try to login
        if token is None:
            token = HF_TOKEN
        
        if token:
            login(token=token, add_to_git_credential=True)
        
        # Define features schema
        features = Features({
            "uuid": Value("string"),
            "timestamp": Value("string"),
            "rating": Value("int64"),
            "feedback": Value("string"),
            "alpha_start": Value("float64"),
            "alpha_end": Value("float64"),
            "n_steps": Value("int64"),
            "is_public": Value("bool"),  # Whether feedback is publicly visible
            "input1": ImageFeature(),
            "input2": ImageFeature(),
            "extra_images": Sequence(ImageFeature()),
            "negative_images": Sequence(ImageFeature()),
            "blending_results": Sequence(ImageFeature()),  # List of blending result images
        })
        
        # Ensure images are RGB PIL Images
        def prepare_image(img):
            if img is None:
                return None
            if isinstance(img, Image.Image):
                return img.convert("RGB") if img.mode != "RGB" else img
            return None
        
        # Generate a new UUID for this feedback entry
        new_uuid = str(uuid.uuid4())
        logging.info(f"Generated UUID for new feedback entry: {new_uuid}")
        
        # Create feedback entry with PIL Images directly
        # ImageFeature handles PIL Images and uploads them properly
        feedback_entry = {
            "uuid": new_uuid,
            "timestamp": datetime.now().isoformat(),
            "rating": int(rating) if rating else 0,
            "feedback": feedback_text or "",
            "alpha_start": float(alpha_start),
            "alpha_end": float(alpha_end),
            "n_steps": int(n_steps),
            "is_public": bool(is_public),  # Store public visibility flag
            "input1": prepare_image(input1_image),
            "input2": prepare_image(input2_image),
            "extra_images": [prepare_image(img) for img in (extra_images or []) if prepare_image(img) is not None],
            "negative_images": [prepare_image(img) for img in (negative_images or []) if prepare_image(img) is not None],
            "blending_results": [prepare_image(img) for img in (blending_result_images or []) if prepare_image(img) is not None],
        }
        
        # Create a new dataset with just the new entry
        new_entry_dataset = Dataset.from_list([feedback_entry], features=features)
        
        # Try to load existing dataset and concatenate
        try:
            existing_dataset = load_dataset(dataset_repo, split="train")
            logging.info(f"Loaded existing dataset with {len(existing_dataset)} entries")
            
            # Check if existing dataset has UUID field, if not add it
            if "uuid" not in existing_dataset.column_names:
                logging.info("Existing dataset missing UUID field, adding UUIDs to existing entries")
                def add_uuid(example):
                    if "uuid" not in example or not example.get("uuid"):
                        example["uuid"] = str(uuid.uuid4())
                    return example
                existing_dataset = existing_dataset.map(add_uuid)
            
            # Check if existing dataset has is_public field, if not add it (default to True for old entries)
            if "is_public" not in existing_dataset.column_names:
                logging.info("Existing dataset missing is_public field, adding is_public=True to existing entries")
                def add_is_public(example):
                    if "is_public" not in example:
                        example["is_public"] = True
                    return example
                existing_dataset = existing_dataset.map(add_is_public)
            
            # Ensure the existing dataset has the UUID and is_public fields in its schema
            # Add fields to schema if missing, then cast to match new schema
            try:
                existing_features = existing_dataset.features
                from datasets import Features as DatasetFeatures
                updated_features_dict = dict(existing_features)
                schema_updated = False
                
                if "uuid" not in existing_features:
                    # Create new features dict with UUID added
                    updated_features_dict["uuid"] = Value("string")
                    schema_updated = True
                
                if "is_public" not in existing_features:
                    # Add is_public field to schema
                    updated_features_dict["is_public"] = Value("bool")
                    schema_updated = True
                
                if schema_updated:
                    updated_features = DatasetFeatures(updated_features_dict)
                    existing_dataset = existing_dataset.cast(updated_features)
                
                # Cast to match new schema (this ensures all fields match)
                existing_dataset = existing_dataset.cast(features)
            except Exception as cast_error:
                logging.warning(f"Could not cast existing dataset schema: {cast_error}. Attempting to proceed anyway.")
                # Try to add UUID field manually if cast failed
                if "uuid" not in existing_dataset.column_names:
                    def ensure_uuid(example):
                        if "uuid" not in example or not example.get("uuid"):
                            example["uuid"] = str(uuid.uuid4())
                        return example
                    existing_dataset = existing_dataset.map(ensure_uuid)
            
            # Concatenate: existing entries stay untouched, new entry is appended
            combined_dataset = concatenate_datasets([existing_dataset, new_entry_dataset])
            logging.info(f"Combined dataset has {len(combined_dataset)} entries")
        except Exception as e:
            logging.info(f"Dataset not found or empty, creating new one: {e}")
            combined_dataset = new_entry_dataset
        
        # Verify UUID is present in all entries before pushing
        if "uuid" not in combined_dataset.column_names:
            logging.warning("UUID column missing from combined dataset, adding UUIDs to all entries")
            def ensure_all_have_uuid(example):
                if "uuid" not in example or not example.get("uuid"):
                    example["uuid"] = str(uuid.uuid4())
                return example
            combined_dataset = combined_dataset.map(ensure_all_have_uuid)
            # Ensure schema includes UUID
            try:
                combined_dataset = combined_dataset.cast(features)
            except Exception as cast_err:
                logging.warning(f"Could not cast combined dataset to features schema: {cast_err}")
        
        # Verify the new entry has UUID
        if len(combined_dataset) > 0:
            last_entry = combined_dataset[-1]
            if last_entry.get("uuid") == new_uuid:
                logging.info(f"Verified UUID {new_uuid} is present in the new entry")
            else:
                logging.warning(f"UUID mismatch! Expected {new_uuid}, got {last_entry.get('uuid')}")
        
        # Push to hub
        combined_dataset.push_to_hub(
            dataset_repo,
            private=True,
            token=token
        )
        
        logging.info(f"Feedback stored successfully to {dataset_repo} with UUID: {new_uuid}")
        return True
        
    except Exception as e:
        logging.error(f"Error storing feedback to Hugging Face Dataset: {e}")
        import traceback
        traceback.print_exc()
        return False



def convert_dataset_image_to_pil(image_data):
    """
    Convert image data from Hugging Face Dataset format to PIL Image.
    
    Dataset images can be:
    - PIL Image (already correct)
    - datasets.Image object (from Hugging Face datasets library)
    - dict with 'path' key (file path)
    - dict with 'bytes' key (image bytes)
    - dict with 'image' key (nested PIL Image)
    - str (file path, filename, or hub reference)
    - None
    """
    if image_data is None or image_data == "":
        return None
    
    # Already a PIL Image
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGB") if image_data.mode != "RGB" else image_data
    
    # Handle datasets.Image type (from Hugging Face datasets library)
    # This is the primary format when images are loaded from hub with ImageFeature
    try:
        from datasets import Image as DatasetImage
        
        # Check if it's a DatasetImage (using isinstance or checking class name)
        is_dataset_image = isinstance(image_data, DatasetImage) if DatasetImage else False
        if not is_dataset_image and hasattr(image_data, '__class__'):
            class_name = str(type(image_data))
            is_dataset_image = 'Image' in class_name and 'datasets' in class_name
        
        if is_dataset_image:
            # Convert datasets.Image to PIL Image
            # DatasetImage objects decode lazily when accessed
            try:
                # Method 1: Try accessing .image attribute (most common)
                if hasattr(image_data, 'image'):
                    pil_img = image_data.image
                    if isinstance(pil_img, Image.Image):
                        return pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img
                
                # Method 2: Try calling as function (some versions)
                if callable(image_data):
                    try:
                        pil_img = image_data()
                        if isinstance(pil_img, Image.Image):
                            return pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img
                    except Exception:
                        pass
                
                # Method 3: Try convert method
                if hasattr(image_data, 'convert'):
                    try:
                        pil_img = image_data.convert("RGB")
                        if isinstance(pil_img, Image.Image):
                            return pil_img
                    except Exception:
                        pass
                
                # Method 4: Try direct access (some versions store PIL Image directly)
                if isinstance(image_data, Image.Image):
                    return image_data.convert("RGB") if image_data.mode != "RGB" else image_data
                
                # Method 5: Try to get bytes and decode
                if hasattr(image_data, 'bytes'):
                    try:
                        bytes_data = image_data.bytes
                        if bytes_data:
                            return Image.open(BytesIO(bytes_data)).convert("RGB")
                    except Exception:
                        pass
                
                # Method 6: Try path attribute and load from hub cache
                if hasattr(image_data, 'path'):
                    try:
                        path = image_data.path
                        if path:
                            # Path might be relative (hub reference) or absolute (local cache)
                            if os.path.exists(path):
                                return Image.open(path).convert("RGB")
                            # If path doesn't exist, it might be in HF cache
                            # Try to find it using huggingface_hub
                            try:
                                from huggingface_hub import hf_hub_download
                                # We'd need the repo name, but for now try the path
                                # The dataset library should handle this automatically
                                pass
                            except:
                                pass
                    except Exception:
                        pass
                
                # Method 7: Try accessing via __getitem__ or direct attribute access
                try:
                    if hasattr(image_data, '__getitem__'):
                        pil_img = image_data[0] if len(image_data) > 0 else None
                        if isinstance(pil_img, Image.Image):
                            return pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img
                except Exception:
                    pass
                
                logging.debug(f"Could not convert DatasetImage object (type: {type(image_data)}, methods tried)")
                return None
            except Exception as e:
                logging.debug(f"Error converting DatasetImage: {e}")
                return None
    except (ImportError, AttributeError) as e:
        pass
    
    # Handle dictionary formats
    if isinstance(image_data, dict):
        # Check for nested image
        if 'image' in image_data:
            return convert_dataset_image_to_pil(image_data['image'])
        
        # Check for path
        if 'path' in image_data:
            try:
                path = image_data['path']
                # If it's a relative path (just filename), try to find it in dataset cache
                if not os.path.isabs(path) and not os.path.exists(path):
                    # Try to load from huggingface cache
                    try:
                        from huggingface_hub import hf_hub_download
                        # This is a fallback - we'd need the dataset repo to download
                        # For now, just try the path as-is
                        pass
                    except:
                        pass
                return Image.open(path).convert("RGB")
            except Exception as e:
                logging.warning(f"Could not load image from path {image_data.get('path')}: {e}")
                return None
        
        # Check for bytes
        if 'bytes' in image_data:
            try:
                return Image.open(BytesIO(image_data['bytes'])).convert("RGB")
            except Exception as e:
                logging.warning(f"Could not load image from bytes: {e}")
                return None
        
        # If dict has no recognized keys, try to extract image directly
        # Some datasets store PIL Images wrapped in dicts
        for key in ['pil', 'PIL', 'img', 'image_data']:
            if key in image_data:
                return convert_dataset_image_to_pil(image_data[key])
        
        # Last resort: if dict has a single value that might be an image path
        if len(image_data) == 1:
            value = list(image_data.values())[0]
            if isinstance(value, str):
                return convert_dataset_image_to_pil(value)
    
    # Try to open if it's a string path
    if isinstance(image_data, str):
        # First try as direct file path
        try:
            if os.path.exists(image_data):
                return Image.open(image_data).convert("RGB")
        except:
            pass
        
        # If it's just a filename (not a full path), it might be in the dataset cache
        # When images are loaded from hub, they're cached locally
        # Try to find it in the Hugging Face cache
        if not os.path.isabs(image_data) and not os.path.exists(image_data):
            try:
                # Try to get the image from the currently loaded dataset
                # This is a workaround - ideally the dataset should return Image objects
                # For now, log a warning
                logging.debug(f"Image filename '{image_data}' not found locally. It should be loaded as Image object from dataset.")
                return None
            except:
                pass
        
        # Last attempt: try opening it anyway
        try:
            return Image.open(image_data).convert("RGB")
        except Exception as e:
            logging.warning(f"Could not load image from string '{image_data}': {e}")
            return None
    
    # If we get here, we don't know how to convert it
    logging.warning(f"Unknown image format: {type(image_data)}")
    return None


def create_image_grid_for_entry(input1, input2, result):
    """Create a grid showing input1, input2, and result side by side."""
    if input1 is None and input2 is None and result is None:
        return None
    
    # Create a horizontal grid: input1 | input2 | result
    images_to_combine = []
    if input1:
        images_to_combine.append(input1)
    if input2:
        images_to_combine.append(input2)
    if result:
        images_to_combine.append(result)
    
    if not images_to_combine:
        return None
    
    # Calculate grid dimensions
    max_width = max(img.width for img in images_to_combine if img)
    max_height = max(img.height for img in images_to_combine if img)
    
    # Resize all images to same height, maintain aspect ratio
    resized_images = []
    for img in images_to_combine:
        if img:
            aspect_ratio = img.width / img.height
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
            resized_images.append(img.resize((new_width, new_height), Image.Resampling.LANCZOS))
        else:
            # Create placeholder
            placeholder = Image.new("RGB", (max_width, max_height), color="gray")
            resized_images.append(placeholder)
    
    # Combine images horizontally
    total_width = sum(img.width for img in resized_images)
    combined = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in resized_images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return combined


def pil_image_to_base64(img: Optional[Image.Image], max_size: int = 400) -> str:
    """Convert PIL Image to base64 data URI for HTML embedding."""
    if img is None:
        return ""
    
    # Resize if too large
    if max(img.width, img.height) > max_size:
        aspect_ratio = img.width / img.height
        if img.width > img.height:
            new_width = max_size
            new_height = int(max_size / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * aspect_ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def load_feedback_from_hf_dataset(
    dataset_repo: Optional[str] = None,
    token: Optional[str] = None,
    limit: Optional[int] = None,
    reverse: bool = False,
    public_only: bool = True
) -> List[dict]:
    """
    Load feedback entries from a Hugging Face Dataset.
    
    Args:
        dataset_repo: Hugging Face dataset repository (username/dataset-name)
        token: Hugging Face token (if None, will try to use HF_TOKEN env var)
        limit: Maximum number of entries to return (None for all)
        reverse: If True, reverse the order (newest first). Default False (oldest first).
        public_only: If True, only return public feedback entries. Default True.
                    Old entries without is_public field are treated as public.
    
    Returns:
        List of feedback entries as dictionaries
    """
    if not HF_DATASETS_AVAILABLE:
        logging.warning("Hugging Face datasets library not available")
        return []
    
    if dataset_repo is None:
        dataset_repo = HF_FEEDBACK_DATASET_REPO
    
    if dataset_repo is None:
        logging.warning("HF_FEEDBACK_DATASET_REPO not set")
        return []
    
    try:
        # Type guards
        if load_dataset is None or login is None:
            logging.error("Hugging Face datasets libraries not properly imported")
            return []
        
        # Use token from parameter, environment variable, or try to login
        if token is None:
            token = HF_TOKEN
        
        if token:
            login(token=token, add_to_git_credential=True)
        
        # Load dataset and cast image columns to ImageFeature to ensure proper decoding
        try:
            from datasets import Features, Image as ImageFeature, Value, Sequence
            
            # Load dataset first
            dataset = load_dataset(dataset_repo, split="train")
            
            # Cast image columns to ImageFeature to ensure they're properly decoded from hub storage
            # This is crucial - without casting, images stored as file paths won't be accessible
            # When images are uploaded with ImageFeature, they're stored in hub storage
            # Casting ensures they're decoded as Image objects when loaded
            try:
                # Check current feature types
                current_features = dataset.features
                logging.debug(f"Dataset features before casting: {list(current_features.keys())}")
                
                # Cast each image column - this will decode file paths from hub storage
                if "input1" in dataset.column_names:
                    dataset = dataset.cast_column("input1", ImageFeature())
                if "input2" in dataset.column_names:
                    dataset = dataset.cast_column("input2", ImageFeature())
                if "blending_results" in dataset.column_names:
                    dataset = dataset.cast_column("blending_results", Sequence(ImageFeature()))
                elif "blending_result" in dataset.column_names:
                    # Backward compatibility for old single-image format
                    dataset = dataset.cast_column("blending_result", ImageFeature())
                if "extra_images" in dataset.column_names:
                    dataset = dataset.cast_column("extra_images", Sequence(ImageFeature()))
                if "negative_images" in dataset.column_names:
                    dataset = dataset.cast_column("negative_images", Sequence(ImageFeature()))
                
                logging.debug(f"Successfully cast image columns to ImageFeature")
            except Exception as e:
                logging.warning(f"Could not cast image columns (may already be ImageFeature or incompatible format): {e}")
                # Continue anyway - images might already be ImageFeature objects or need different handling
            
            # Convert dataset to list, ensuring images are properly loaded
            # When ImageFeature is used, images come as Image objects that need to be accessed
            # Iterate through dataset and explicitly access image fields to trigger decoding
            data = []
            for idx, entry in enumerate(dataset):
                # Access image fields directly from entry to trigger lazy loading/decoding
                # This is critical for DatasetImage objects which decode on access
                try:
                    # Force access to image fields - this triggers decoding from hub storage
                    input1_raw = entry["input1"] if "input1" in entry else None
                    input2_raw = entry["input2"] if "input2" in entry else None
                    # Support both new format (blending_results - list) and old format (blending_result - single)
                    blending_results_raw = entry.get("blending_results", [])
                    blending_result_raw = entry.get("blending_result", None)  # Backward compatibility
                    extra_images_raw = entry.get("extra_images", [])
                    negative_images_raw = entry.get("negative_images", [])
                except Exception as e:
                    logging.debug(f"Error accessing image fields in entry {idx}: {e}")
                    input1_raw = None
                    input2_raw = None
                    blending_results_raw = []
                    blending_result_raw = None
                    extra_images_raw = []
                    negative_images_raw = []
                
                # Create a copy of the entry with decoded images
                entry_dict = dict(entry)
                
                # Add UUID if missing (backward compatibility for old entries)
                if "uuid" not in entry_dict or not entry_dict.get("uuid"):
                    entry_dict["uuid"] = str(uuid.uuid4())
                
                # Add is_public if missing (backward compatibility - old entries are public by default)
                if "is_public" not in entry_dict:
                    entry_dict["is_public"] = True
                
                # Filter by public_only if requested
                if public_only and not entry_dict.get("is_public", True):
                    continue
                
                # Convert Image objects to PIL Images using the raw accessed values
                # ImageFeature objects need to be converted to PIL Images for display
                try:
                    # Handle single image fields using the raw accessed values
                    entry_dict["input1"] = convert_dataset_image_to_pil(input1_raw)
                    entry_dict["input2"] = convert_dataset_image_to_pil(input2_raw)
                    
                    # Handle blending results - support both new list format and old single image format
                    if blending_results_raw:
                        converted_results = []
                        for img_item in blending_results_raw:
                            converted = convert_dataset_image_to_pil(img_item)
                            if converted is not None:
                                converted_results.append(converted)
                        entry_dict["blending_results"] = converted_results
                    elif blending_result_raw:
                        # Backward compatibility: convert single image to list
                        single_img = convert_dataset_image_to_pil(blending_result_raw)
                        entry_dict["blending_results"] = [single_img] if single_img else []
                    else:
                        entry_dict["blending_results"] = []
                    
                    # Handle list image fields
                    if extra_images_raw:
                        converted_extra = []
                        for img_item in extra_images_raw:
                            converted = convert_dataset_image_to_pil(img_item)
                            if converted is not None:
                                converted_extra.append(converted)
                        entry_dict["extra_images"] = converted_extra
                    else:
                        entry_dict["extra_images"] = []
                    
                    if negative_images_raw:
                        converted_neg = []
                        for img_item in negative_images_raw:
                            converted = convert_dataset_image_to_pil(img_item)
                            if converted is not None:
                                converted_neg.append(converted)
                        entry_dict["negative_images"] = converted_neg
                    else:
                        entry_dict["negative_images"] = []
                except Exception as e:
                    logging.warning(f"Error converting images in entry {idx}: {e}")
                    # Set to None if conversion fails
                    entry_dict["input1"] = None
                    entry_dict["input2"] = None
                    entry_dict["blending_results"] = []
                    entry_dict["extra_images"] = []
                    entry_dict["negative_images"] = []
                
                data.append(entry_dict)
            
            # Sort by timestamp to ensure proper chronological order
            # Parse timestamp and sort (oldest first by default)
            def get_timestamp(entry):
                timestamp_str = entry.get("timestamp", "")
                if not timestamp_str:
                    return 0
                try:
                    # Parse ISO format timestamp (e.g., "2024-01-01T12:00:00" or "2024-01-01T12:00:00.123456")
                    # Handle Z suffix for UTC
                    if timestamp_str.endswith("Z"):
                        timestamp_str = timestamp_str[:-1] + "+00:00"
                    # Try parsing with timezone info first
                    if "+" in timestamp_str or timestamp_str.count("-") > 2:
                        # Has timezone info
                        dt = datetime.fromisoformat(timestamp_str)
                    else:
                        # No timezone, assume naive datetime
                        dt = datetime.fromisoformat(timestamp_str)
                    return dt.timestamp()
                except (ValueError, AttributeError) as e:
                    # If timestamp parsing fails, log and use 0 (will appear first in old-to-new)
                    logging.debug(f"Could not parse timestamp '{timestamp_str}': {e}")
                    return 0
            
            # Sort by timestamp (oldest first)
            data.sort(key=get_timestamp)
            
            # Reverse if requested (newest first)
            if reverse:
                data.reverse()
            
            # Apply limit if specified
            if limit is not None:
                data = data[:limit]
            
            return data
        except Exception as e:
            logging.warning(f"Could not load dataset: {e}")
            return []
        
    except Exception as e:
        logging.error(f"Error loading feedback from Hugging Face Dataset: {e}")
        return []


def create_feedback_viewer_tab():
    """Create the feedback viewer tab interface."""
    with gr.Tab("Feedback Viewer"):
        gr.Markdown("""
        ## Feedback Viewer
        
        View submitted feedback and images from users.
        """)
        
        # Top controls group
        with gr.Group():
            refresh_button = gr.Button("🔄 Refresh Feedback", variant="primary")
        
        # Main feedback display
        feedback_html = gr.HTML(
            label="Feedback Entries",
            value="<p>Click 'Refresh Feedback' to load entries.</p>"
        )
        
        # Pagination controls group
        with gr.Group():
            gr.Markdown("### Pagination")
            with gr.Row():
                items_per_page_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=10,
                    label="Items per page",
                    scale=2
                )
                with gr.Column(scale=3):
                    with gr.Row():
                        prev_page_button = gr.Button("◀ Previous", variant="secondary", scale=1)
                        page_number = gr.Number(
                            value=1,
                            minimum=1,
                            step=1,
                            label="Page",
                            precision=0,
                            scale=2
                        )
                        next_page_button = gr.Button("Next ▶", variant="secondary", scale=1)
                total_pages_display = gr.Markdown("**Total Pages:** -")
        
        # Search controls group
        with gr.Group():
            gr.Markdown("### Search & Filter")
            with gr.Row():
                uuid_search_input = gr.Textbox(
                    label="Search by UUID",
                    placeholder="Enter UUID to search (leave empty to show all)",
                    value="",
                    scale=3
                )
                search_button = gr.Button("🔍 Search", variant="secondary", scale=1)
            
            with gr.Row():
                timestamp_start_input = gr.Textbox(
                    label="Start Timestamp",
                    placeholder="YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS",
                    value="",
                    scale=1
                )
                timestamp_end_input = gr.Textbox(
                    label="End Timestamp",
                    placeholder="YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS",
                    value="",
                    scale=1
                )
                rating_filter = gr.Dropdown(
                    choices=["All", "1", "2", "3", "4", "5"],
                    value="All",
                    label="Rating Filter",
                    scale=1
                )
            
            with gr.Row():
                filter_extra_images = gr.Checkbox(
                    label="Only entries with extra images",
                    value=False
                )
                filter_negative_images = gr.Checkbox(
                    label="Only entries with negative images",
                    value=False
                )
                sort_order_radio = gr.Radio(
                    choices=["Old to New", "New to Old"],
                    value="Old to New",
                    label="Sort Order",
                    scale=2
                )
        
        selected_details = gr.JSON(label="Full Feedback Details", visible=False)
        
        # Admin section - hidden behind accordion
        with gr.Accordion("⋮", open=False):
            gr.Markdown("### Admin Options")
            with gr.Row():
                admin_password_input = gr.Textbox(
                    label="Admin Password",
                    placeholder="Enter admin password",
                    type="password",
                    value="",
                    scale=2
                )
                include_private_checkbox = gr.Checkbox(
                    label="Include private feedbacks",
                    value=False,
                    interactive=False,
                    scale=1
                )
                verify_password_button = gr.Button("🔓 Verify", variant="secondary", scale=1)
            admin_status = gr.Markdown("")
            
            gr.Markdown("### Delete Entry")
            with gr.Row():
                delete_uuid_input = gr.Textbox(
                    label="UUID to Delete",
                    placeholder="Enter UUID or prefix (e.g. e7132a33)",
                    value="",
                    scale=3
                )
                delete_button = gr.Button("🗑️ Delete", variant="stop", scale=1)
            delete_status = gr.Markdown("")
        
        def verify_admin_password(password: str, current_include_private: bool):
            """Verify admin password and enable/disable private feedback checkbox."""
            if password == "admin":
                return (
                    gr.update(value=True, interactive=True),  # Enable and check the checkbox
                    "✅ Admin access granted. You can now view private feedbacks."
                )
            else:
                return (
                    gr.update(value=False, interactive=False),  # Disable and uncheck the checkbox
                    "❌ Invalid password. Private feedbacks hidden."
                )
        
        verify_password_button.click(
            verify_admin_password,
            inputs=[admin_password_input, include_private_checkbox],
            outputs=[include_private_checkbox, admin_status]
        )
        
        def delete_entry_by_uuid(uuid_to_delete: str, admin_password: str):
            """Delete a feedback entry by UUID after password verification."""
            # Check password
            if admin_password != "admin":
                return "❌ Invalid admin password. Enter password and click Verify first."
            
            if not uuid_to_delete or not uuid_to_delete.strip():
                return "❌ Please enter a UUID to delete."
            
            uuid_to_delete = uuid_to_delete.strip()
            
            try:
                if not HF_DATASETS_AVAILABLE or load_dataset is None or login is None:
                    return "❌ Hugging Face datasets library not available."
                
                if HF_TOKEN:
                    login(token=HF_TOKEN, add_to_git_credential=True)
                
                if not HF_FEEDBACK_DATASET_REPO:
                    return "❌ HF_FEEDBACK_DATASET_REPO not configured."
                
                # Load dataset
                dataset = load_dataset(HF_FEEDBACK_DATASET_REPO, split="train")
                original_count = len(dataset)
                
                if original_count == 0:
                    return "❌ Dataset is empty. Nothing to delete."
                
                # First, find matching entries (exact or prefix match)
                search_term = uuid_to_delete.lower()
                matching_entries = []
                
                for entry in dataset:
                    entry_uuid = entry.get("uuid", "")
                    if entry_uuid.lower() == search_term or entry_uuid.lower().startswith(search_term):
                        matching_entries.append(entry_uuid)
                
                if len(matching_entries) == 0:
                    return f"❌ No UUID matching '{uuid_to_delete}' found in dataset."
                elif len(matching_entries) > 1:
                    matches_display = ", ".join([u[:12] + "..." for u in matching_entries[:5]])
                    if len(matching_entries) > 5:
                        matches_display += f" (+{len(matching_entries) - 5} more)"
                    return f"❌ Multiple UUIDs match '{uuid_to_delete}': {matches_display}. Please be more specific."
                
                # Exactly one match - use the full UUID
                uuid_to_delete_full = matching_entries[0]
                
                # Filter entries to keep (exclude the matched UUID)
                entries_to_keep = []
                
                for entry in dataset:
                    entry_uuid = entry.get("uuid", "")
                    if entry_uuid == uuid_to_delete_full:
                        continue
                    else:
                        entries_to_keep.append(dict(entry))
                
                # Handle empty dataset case
                if len(entries_to_keep) == 0:
                    # Create placeholder entry to avoid empty dataset issues
                    import uuid as uuid_module
                    placeholder = {}
                    deleted_entry = next(entry for entry in dataset if entry.get("uuid", "") == uuid_to_delete_full)
                    for key in dataset.features.keys():
                        if key == "timestamp":
                            placeholder[key] = datetime.now().isoformat()
                        elif key == "rating":
                            placeholder[key] = 0
                        elif key == "feedback":
                            placeholder[key] = "[PLACEHOLDER - This entry can be deleted]"
                        elif key in ["alpha_start", "alpha_end"]:
                            placeholder[key] = 0.0
                        elif key == "n_steps":
                            placeholder[key] = 0
                        elif key in ["input1", "input2"]:
                            placeholder[key] = deleted_entry.get(key)
                        elif key in ["extra_images", "negative_images", "blending_results", "blending_result"]:
                            placeholder[key] = []
                        elif key == "uuid":
                            placeholder[key] = str(uuid_module.uuid4())
                        else:
                            placeholder[key] = deleted_entry.get(key, "")
                    
                    new_dataset = Dataset.from_list([placeholder], features=dataset.features)
                    new_dataset.push_to_hub(
                        HF_FEEDBACK_DATASET_REPO,
                        private=True,
                        token=HF_TOKEN
                    )
                    return f"✅ Deleted entry with UUID '{uuid_to_delete_full}'. Dataset now has 1 placeholder entry."
                
                # Create new dataset with remaining entries
                new_dataset = Dataset.from_list(entries_to_keep, features=dataset.features)
                
                # Push to hub
                new_dataset.push_to_hub(
                    HF_FEEDBACK_DATASET_REPO,
                    private=True,
                    token=HF_TOKEN
                )
                
                return f"✅ Successfully deleted entry with UUID '{uuid_to_delete_full}'. {len(entries_to_keep)} entries remaining."
                
            except Exception as e:
                logging.error(f"Error deleting entry: {e}")
                import traceback
                traceback.print_exc()
                return f"❌ Error deleting entry: {str(e)}"
        
        delete_button.click(
            delete_entry_by_uuid,
            inputs=[delete_uuid_input, admin_password_input],
            outputs=[delete_status]
        )
        
        def parse_timestamp(timestamp_str):
            """Parse timestamp string to datetime object."""
            if not timestamp_str or not timestamp_str.strip():
                return None
            
            timestamp_str = timestamp_str.strip()
            
            # Try different formats
            formats = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d",
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            # Try ISO format parsing
            try:
                if timestamp_str.endswith("Z"):
                    timestamp_str = timestamp_str[:-1] + "+00:00"
                return datetime.fromisoformat(timestamp_str)
            except (ValueError, AttributeError):
                pass
            
            return None
        
        def load_and_display_feedback(items_per_page, page, sort_order, uuid_search="", timestamp_start="", timestamp_end="", rating_filter="All", filter_extra=False, filter_negative=False, include_private=False):
            """Load feedback from dataset and format as HTML table with pagination."""
            # Convert radio selection to reverse boolean
            sort_reverse = (sort_order == "New to Old")
            # Load all feedback entries (no limit, we'll paginate ourselves)
            # If include_private is True, set public_only to False
            all_feedbacks = load_feedback_from_hf_dataset(reverse=sort_reverse, public_only=not include_private)
            
            # Filter by UUID if search term provided
            if uuid_search and uuid_search.strip():
                search_term = uuid_search.strip().lower()
                all_feedbacks = [
                    fb for fb in all_feedbacks 
                    if fb.get("uuid", "").lower().startswith(search_term)
                ]
            
            # Filter by timestamp range if provided
            start_dt = parse_timestamp(timestamp_start)
            end_dt = parse_timestamp(timestamp_end)
            
            if start_dt or end_dt:
                filtered_feedbacks = []
                for fb in all_feedbacks:
                    fb_timestamp_str = fb.get("timestamp", "")
                    if not fb_timestamp_str:
                        continue
                    
                    fb_dt = parse_timestamp(fb_timestamp_str)
                    if not fb_dt:
                        continue
                    
                    # Check if timestamp is within range
                    if start_dt and fb_dt < start_dt:
                        continue
                    if end_dt and fb_dt > end_dt:
                        continue
                    
                    filtered_feedbacks.append(fb)
                
                all_feedbacks = filtered_feedbacks
            
            # Filter by rating if specified
            if rating_filter and rating_filter != "All":
                try:
                    rating_value = int(rating_filter)
                    all_feedbacks = [
                        fb for fb in all_feedbacks 
                        if fb.get("rating", 0) == rating_value
                    ]
                except (ValueError, TypeError):
                    pass  # Invalid rating filter, ignore it
            
            # Filter by extra images if checkbox is checked
            if filter_extra:
                all_feedbacks = [
                    fb for fb in all_feedbacks 
                    if fb.get("extra_images") and len(fb.get("extra_images", [])) > 0
                ]
            
            # Filter by negative images if checkbox is checked
            if filter_negative:
                all_feedbacks = [
                    fb for fb in all_feedbacks 
                    if fb.get("negative_images") and len(fb.get("negative_images", [])) > 0
                ]
            
            if not all_feedbacks:
                if uuid_search and uuid_search.strip():
                    gr.Info(f"No feedback entries found matching UUID: {uuid_search}")
                elif start_dt or end_dt:
                    gr.Info("No feedback entries found in the specified timestamp range.")
                elif rating_filter and rating_filter != "All":
                    gr.Info(f"No feedback entries found with rating: {rating_filter}")
                else:
                    gr.Info("No feedback entries found. Make sure HF_FEEDBACK_DATASET_REPO is configured.")
                return "<p>No feedback entries found.</p>", 1, "**Total Pages:** 0"
            
            # Calculate pagination
            total_items = len(all_feedbacks)
            items_per_page = max(1, int(items_per_page))
            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
            page = max(1, min(int(page), total_pages))
            
            # Get the slice of feedbacks for current page
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            feedbacks = all_feedbacks[start_idx:end_idx]
            
            # Calculate global index offset for display
            global_start_idx = start_idx
            
            # Start building HTML table
            html_parts = ["""
            <style>
                .feedback-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-family: Arial, sans-serif;
                    table-layout: fixed;
                }
                .feedback-table th {
                    background-color: #4a5568;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    border: 1px solid #2d3748;
                    font-weight: bold;
                }
                .feedback-table th:nth-child(1) {
                    width: 20%;
                }
                .feedback-table th:nth-child(2) {
                    width: 35%;
                }
                .feedback-table th:nth-child(3) {
                    width: 15%;
                }
                .feedback-table th:nth-child(4) {
                    width: 15%;
                }
                .feedback-table th:nth-child(5) {
                    width: 15%;
                }
                .feedback-table td {
                    padding: 10px;
                    border: 1px solid #e2e8f0;
                    vertical-align: top;
                }
                .feedback-table tr:nth-child(even) {
                    background-color: #f7fafc;
                }
                .feedback-table tr:hover {
                    background-color: #edf2f7;
                }
                .input-images {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                    align-items: center;
                }
                .input-images img {
                    max-width: 200px;
                    max-height: 200px;
                    border: 2px solid #cbd5e0;
                    border-radius: 4px;
                }
                .result-image img {
                    max-width: 100%;
                    max-height: 500px;
                    border: 2px solid #48bb78;
                    border-radius: 4px;
                }
                .options-cell {
                    font-size: 0.85em;
                    line-height: 1.5;
                }
                .options-cell strong {
                    color: #2d3748;
                }
                .image-gallery {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                    justify-content: center;
                }
                .image-gallery img {
                    max-width: 100px;
                    max-height: 100px;
                    border: 1px solid #cbd5e0;
                    border-radius: 4px;
                }
                .no-image {
                    color: #a0aec0;
                    font-style: italic;
                    text-align: center;
                    padding: 20px;
                }
            </style>
            <table class="feedback-table">
                <thead>
                    <tr>
                        <th>Input Images</th>
                        <th>Result Image</th>
                        <th>Info</th>
                        <th>Extra Images</th>
                        <th>Negative Images</th>
                    </tr>
                </thead>
                <tbody>
            """]
            
            for local_idx, feedback in enumerate(feedbacks):
                # Calculate global index for display
                global_idx = global_start_idx + local_idx
                
                uuid_value = feedback.get("uuid", "N/A")
                timestamp = feedback.get("timestamp", "Unknown")
                # Format timestamp for display
                if len(timestamp) > 19:
                    timestamp = timestamp[:19].replace("T", " ")
                
                rating = str(feedback.get("rating", "N/A"))
                feedback_text = feedback.get("feedback", "") or ""
                alpha_start = str(feedback.get("alpha_start", "N/A"))
                alpha_end = str(feedback.get("alpha_end", "N/A"))
                n_steps = str(feedback.get("n_steps", "N/A"))
                
                # Convert images from dataset format to PIL Images
                input1_img = convert_dataset_image_to_pil(feedback.get("input1"))
                input2_img = convert_dataset_image_to_pil(feedback.get("input2"))
                
                # Get blending results list (new format) or single result (old format)
                blending_results_raw = feedback.get("blending_results", [])
                blending_results_list = []
                if blending_results_raw:
                    for img in blending_results_raw:
                        converted = convert_dataset_image_to_pil(img)
                        if converted:
                            blending_results_list.append(converted)
                
                if not blending_results_list:
                    # Backward compatibility: check for old single image format
                    old_result = convert_dataset_image_to_pil(feedback.get("blending_result"))
                    if old_result:
                        blending_results_list = [old_result]
                
                # Convert list of images
                extra_imgs = []
                if feedback.get("extra_images"):
                    for img in feedback.get("extra_images", []):
                        converted = convert_dataset_image_to_pil(img)
                        if converted:
                            extra_imgs.append(converted)
                
                negative_imgs = []
                if feedback.get("negative_images"):
                    for img in feedback.get("negative_images", []):
                        converted = convert_dataset_image_to_pil(img)
                        if converted:
                            negative_imgs.append(converted)
                
                # Column 1: Input Images
                input_images_html = '<div class="input-images">'
                if input1_img:
                    input1_base64 = pil_image_to_base64(input1_img, max_size=512)
                    input_images_html += f'<img src="{input1_base64}" alt="Input 1" title="Input 1" />'
                else:
                    input_images_html += '<div class="no-image">No Input 1</div>'
                
                if input2_img:
                    input2_base64 = pil_image_to_base64(input2_img, max_size=512)
                    input_images_html += f'<img src="{input2_base64}" alt="Input 2" title="Input 2" />'
                else:
                    input_images_html += '<div class="no-image">No Input 2</div>'
                input_images_html += '</div>'
                
                # Column 2: Result Image - create grid from list of images
                if blending_results_list and len(blending_results_list) > 0:
                    # Create grid from list of blending result images
                    n_images = len(blending_results_list)
                    cols = min(4, n_images)
                    rows = math.ceil(n_images / cols)
                    blending_result_grid = create_image_grid(blending_results_list, rows=rows, cols=cols)
                    result_base64 = pil_image_to_base64(blending_result_grid, max_size=99999)
                    result_html = f'<div class="result-image"><img src="{result_base64}" alt="Result" title="Blending Result ({n_images} images)" /></div>'
                else:
                    result_html = '<div class="no-image">No Result</div>'
                
                # Column 3: Info
                uuid_display = uuid_value[:8] + "..." if len(uuid_value) > 12 else uuid_value
                options_html = f'''
                <div class="options-cell">
                    <strong>Timestamp:</strong> {timestamp}<br/>
                    <strong>UUID:</strong> <code onclick="navigator.clipboard.writeText('{uuid_value}').then(() => this.style.backgroundColor='#90EE90').catch(() => alert('Copy failed')); setTimeout(() => this.style.backgroundColor='#f0f0f0', 500);" style="font-size: 0.8em; background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; cursor: pointer;" title="Click to copy: {uuid_value}">{uuid_display}</code><br/>
                    <strong>Rating:</strong> {rating}/5<br/>
                    <strong>Alpha:</strong> {alpha_start} → {alpha_end}<br/>
                    <strong>Steps:</strong> {n_steps}<br/>
                    <strong>Feedback:</strong> {feedback_text if feedback_text else "None"}
                </div>
                '''
                
                # Column 4: Extra Images
                if extra_imgs:
                    extra_html = '<div class="image-gallery">'
                    for img in extra_imgs:
                        img_base64 = pil_image_to_base64(img, max_size=512)
                        extra_html += f'<img src="{img_base64}" alt="Extra Image" />'
                    extra_html += '</div>'
                else:
                    extra_html = '<div class="no-image">None</div>'
                
                # Column 5: Negative Images
                if negative_imgs:
                    negative_html = '<div class="image-gallery">'
                    for img in negative_imgs:
                        img_base64 = pil_image_to_base64(img, max_size=512)
                        negative_html += f'<img src="{img_base64}" alt="Negative Image" />'
                    negative_html += '</div>'
                else:
                    negative_html = '<div class="no-image">None</div>'
                
                # Add row
                html_parts.append(f"""
                <tr>
                    <td>{input_images_html}</td>
                    <td>{result_html}</td>
                    <td>{options_html}</td>
                    <td>{extra_html}</td>
                    <td>{negative_html}</td>
                </tr>
                """)
            
            html_parts.append("</tbody></table>")
            
            # Add pagination info
            pagination_info = f"""
            <div style="margin-top: 20px; padding: 15px; background-color: #f7fafc; border-radius: 5px; text-align: center;">
                <strong>Showing entries {start_idx + 1}-{min(end_idx, total_items)} of {total_items}</strong><br/>
                <span style="color: #718096;">Page {page} of {total_pages}</span>
            </div>
            """
            html_parts.append(pagination_info)
            
            html_content = "".join(html_parts)
            
            gr.Info(f"Loaded {len(feedbacks)} feedback entries (page {page}/{total_pages})")
            return html_content, page, f"**Total Pages:** {total_pages}"
        
        # Refresh button - loads first page
        def refresh_feedback(items_per_page, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private):
            return load_and_display_feedback(items_per_page, 1, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private)
        
        refresh_button.click(
            refresh_feedback,
            inputs=[items_per_page_slider, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )
        
        # Sort order radio - reset to page 1 when sort order changes
        def on_sort_order_change(items_per_page, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private):
            # Reset to page 1 when sort order changes
            return load_and_display_feedback(items_per_page, 1, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private)
        
        sort_order_radio.change(
            on_sort_order_change,
            inputs=[items_per_page_slider, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )
        
        # Search button
        def on_search(items_per_page, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private):
            return load_and_display_feedback(items_per_page, 1, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private)
        
        search_button.click(
            on_search,
            inputs=[items_per_page_slider, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )
        
        # Rating filter change
        rating_filter.change(
            on_search,
            inputs=[items_per_page_slider, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )
        
        # Checkbox changes
        filter_extra_images.change(
            on_search,
            inputs=[items_per_page_slider, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )
        
        filter_negative_images.change(
            on_search,
            inputs=[items_per_page_slider, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )
        
        # Include private checkbox change - refresh when toggled
        include_private_checkbox.change(
            on_search,
            inputs=[items_per_page_slider, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )
        
        
        # Pagination controls
        def on_page_change(items_per_page, page, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private):
            return load_and_display_feedback(items_per_page, page, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private)
        
        def on_items_per_page_change(items_per_page, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private):
            return load_and_display_feedback(items_per_page, 1, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private)
        
        page_number.change(
            on_page_change,
            inputs=[items_per_page_slider, page_number, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )
        
        items_per_page_slider.change(
            on_items_per_page_change,
            inputs=[items_per_page_slider, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )
        
        # Previous/Next page buttons
        def go_to_previous_page(items_per_page, current_page, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private):
            new_page = max(1, int(current_page) - 1)
            return load_and_display_feedback(items_per_page, new_page, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private)
        
        def go_to_next_page(items_per_page, current_page, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private):
            # Convert radio selection to reverse boolean
            sort_reverse = (sort_order == "New to Old")
            # Load all feedbacks to calculate total pages (with filters if applicable)
            all_feedbacks = load_feedback_from_hf_dataset(reverse=sort_reverse, public_only=not include_private)
            
            # Apply UUID filter if search term provided
            if uuid_search and uuid_search.strip():
                search_term = uuid_search.strip().lower()
                all_feedbacks = [
                    fb for fb in all_feedbacks 
                    if fb.get("uuid", "").lower().startswith(search_term)
                ]
            
            # Apply timestamp range filter
            start_dt = parse_timestamp(timestamp_start)
            end_dt = parse_timestamp(timestamp_end)
            
            if start_dt or end_dt:
                filtered_feedbacks = []
                for fb in all_feedbacks:
                    fb_timestamp_str = fb.get("timestamp", "")
                    if not fb_timestamp_str:
                        continue
                    
                    fb_dt = parse_timestamp(fb_timestamp_str)
                    if not fb_dt:
                        continue
                    
                    if start_dt and fb_dt < start_dt:
                        continue
                    if end_dt and fb_dt > end_dt:
                        continue
                    
                    filtered_feedbacks.append(fb)
                
                all_feedbacks = filtered_feedbacks
            
            # Apply rating filter if specified
            if rating_filter and rating_filter != "All":
                try:
                    rating_value = int(rating_filter)
                    all_feedbacks = [
                        fb for fb in all_feedbacks 
                        if fb.get("rating", 0) == rating_value
                    ]
                except (ValueError, TypeError):
                    pass  # Invalid rating filter, ignore it
            
            # Apply extra images filter if checkbox is checked
            if filter_extra:
                all_feedbacks = [
                    fb for fb in all_feedbacks 
                    if fb.get("extra_images") and len(fb.get("extra_images", [])) > 0
                ]
            
            # Apply negative images filter if checkbox is checked
            if filter_negative:
                all_feedbacks = [
                    fb for fb in all_feedbacks 
                    if fb.get("negative_images") and len(fb.get("negative_images", [])) > 0
                ]
            
            if not all_feedbacks:
                return load_and_display_feedback(items_per_page, 1, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private)
            
            total_items = len(all_feedbacks)
            items_per_page = max(1, int(items_per_page))
            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
            new_page = min(total_pages, int(current_page) + 1)
            return load_and_display_feedback(items_per_page, new_page, sort_order, uuid_search, timestamp_start, timestamp_end, rating_filter, filter_extra, filter_negative, include_private)
        
        prev_page_button.click(
            go_to_previous_page,
            inputs=[items_per_page_slider, page_number, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )
        
        next_page_button.click(
            go_to_next_page,
            inputs=[items_per_page_slider, page_number, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )
        
        # Allow Enter key to trigger search
        uuid_search_input.submit(
            on_search,
            inputs=[items_per_page_slider, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )
        
        timestamp_start_input.submit(
            on_search,
            inputs=[items_per_page_slider, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )
        
        timestamp_end_input.submit(
            on_search,
            inputs=[items_per_page_slider, sort_order_radio, uuid_search_input, timestamp_start_input, timestamp_end_input, rating_filter, filter_extra_images, filter_negative_images, include_private_checkbox],
            outputs=[feedback_html, page_number, total_pages_display]
        )


if __name__ == "__main__":
    """Run the feedback viewer as a standalone application."""
    logging.basicConfig(level=logging.INFO)
    
    # Create a Gradio interface with just the feedback viewer tab
    demo = gr.Blocks()
    with demo:
        create_feedback_viewer_tab()
    
    # Launch the demo
    demo.launch(
        share=True,
        show_error=True
    )
