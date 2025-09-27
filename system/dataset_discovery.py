"""Dataset discovery module for Hootsight.

Scans the models directory to find projects and analyze dataset types.
Supports multiple dataset formats: folder-based labels, txt annotations, JSON annotations.
"""
import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS, lang


class DatasetType:
    """Dataset type constants."""
    FOLDER_LABELS = "folder_labels"  # Subfolders are labels
    TXT_ANNOTATIONS = "txt_annotations"  # Each image has corresponding .txt file
    JSON_ANNOTATIONS = "json_annotations"  # JSON file with annotations
    CSV_ANNOTATIONS = "csv_annotations"  # CSV file with image,label pairs
    UNKNOWN = "unknown"


class ProjectInfo:
    """Information about a discovered project."""
    
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.dataset_path = os.path.join(path, "dataset")
        self.model_path = os.path.join(path, "model")
        self.dataset_type = DatasetType.UNKNOWN
        self.image_count = 0
        self.labels = []
        self.label_distribution = {}  # Label -> count mapping
        self.detailed_distribution = {}  # Hierarchical distribution with sub-categories
        self.balance_score = 0.0  # 0-1, where 1 = perfectly balanced
        self.balance_analysis = {}  # Detailed balance analysis
        self.recommendations = []  # List of recommendations to improve dataset
        self.has_dataset = False
        self.has_model_dir = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "dataset_path": self.dataset_path,
            "model_path": self.model_path,
            "dataset_type": self.dataset_type,
            "image_count": self.image_count,
            "labels": self.labels,
            "label_distribution": self.label_distribution,
            "detailed_distribution": self.detailed_distribution,
            "balance_score": round(self.balance_score, 3),
            "balance_status": self.get_balance_status(),
            "balance_analysis": self.balance_analysis,
            "recommendations": self.recommendations,
            "has_dataset": self.has_dataset,
            "has_model_dir": self.has_model_dir
        }
    
    def get_balance_status(self) -> str:
        """Get human-readable balance status."""
        if self.balance_score >= 0.9:
            return "Excellent"
        elif self.balance_score >= 0.7:
            return "Good"
        elif self.balance_score >= 0.5:
            return "Fair"
        elif self.balance_score >= 0.3:
            return "Poor"
        else:
            return "Critical"


def get_image_files(directory: str) -> List[str]:
    """Get all image files in directory recursively."""
    image_extensions = SETTINGS.get('dataset', {}).get('image_extensions', 
        ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'])
    image_files = []
    
    if not os.path.exists(directory):
        return image_files
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files


def analyze_balance_and_generate_recommendations(label_distribution: Dict[str, int], 
                                               detailed_distribution: Dict[str, int],
                                               dataset_type: str) -> tuple[float, Dict[str, Any], List[str]]:
    """Comprehensive balance analysis with actionable recommendations."""
    
    if not label_distribution:
        return 1.0, {}, []
    
    balance_config = SETTINGS.get('dataset', {}).get('discovery', {}).get('balance_analysis', {})
    min_images_per_class = balance_config.get('min_images_per_class', 5)
    critical_shortage_threshold = balance_config.get('critical_shortage_threshold', 50)
    over_representation_ratio = balance_config.get('over_representation_ratio', 2.0)
    under_representation_ratio = balance_config.get('under_representation_ratio', 0.5)
    severe_over_representation_ratio = balance_config.get('severe_over_representation_ratio', 3.0)
    hierarchical_balance_threshold = balance_config.get('hierarchical_balance_threshold', 0.3)
    dataset_size_warnings = balance_config.get('dataset_size_warnings', {})
    tiny_dataset_threshold = dataset_size_warnings.get('tiny_dataset', 100)
    small_dataset_threshold = dataset_size_warnings.get('small_dataset', 1000)
    
    total_images = sum(label_distribution.values())
    num_labels = len(label_distribution)
    
    # Calculate ideal distribution
    ideal_per_label = total_images / num_labels
    
    # Calculate balance score
    balance_score = calculate_balance_score(label_distribution)
    
    # Detailed analysis
    analysis = {
        "total_images": total_images,
        "num_labels": num_labels,
        "ideal_per_label": int(ideal_per_label),
        "min_images": min(label_distribution.values()),
        "max_images": max(label_distribution.values()),
        "ratio_max_to_min": max(label_distribution.values()) / max(min(label_distribution.values()), 1),
        "over_represented": [],
        "under_represented": [],
        "critical_shortage": []
    }
    
    # Identify problematic categories
    for label, count in label_distribution.items():
        ratio = count / ideal_per_label
        if ratio > over_representation_ratio:  # More than 2x ideal
            analysis["over_represented"].append({"label": label, "count": count, "ratio": round(ratio, 2)})
        elif ratio < under_representation_ratio:  # Less than half ideal
            analysis["under_represented"].append({"label": label, "count": count, "ratio": round(ratio, 2)})
        if count < critical_shortage_threshold:  # Critical shortage
            analysis["critical_shortage"].append({"label": label, "count": count})
    
    # Generate recommendations
    recommendations = []
    
    # Critical shortage recommendations
    if analysis["critical_shortage"]:
        critical_labels = [item["label"] for item in analysis["critical_shortage"]]
        recommendations.append(lang("recommendations.critical_shortage", labels=critical_labels))
    
    # Over-representation recommendations
    if analysis["over_represented"]:
        over_labels = [item["label"] for item in analysis["over_represented"] if item["ratio"] > severe_over_representation_ratio]
        if over_labels:
            recommendations.append(lang("recommendations.reduce_oversampled", labels=over_labels))
    
    # Under-representation recommendations
    if analysis["under_represented"]:
        under_labels = [item["label"] for item in analysis["under_represented"]]
        recommendations.append(lang("recommendations.augment_undersampled", labels=under_labels))
    
    # Training strategy recommendations
    if balance_score < balance_config.get('balance_score_thresholds', {}).get('poor', 0.3):
        recommendations.append(lang("recommendations.weighted_loss"))
        recommendations.append(lang("recommendations.stratified_sampling"))
    
    # Dataset-specific recommendations
    if dataset_type == DatasetType.FOLDER_LABELS and detailed_distribution:
        # Analyze hierarchical imbalance
        hierarchical_issues = []
        parent_categories = {}
        for key, count in detailed_distribution.items():
            if '/' in key:
                parent = key.split('/')[0]
                parent_categories[parent] = parent_categories.get(parent, 0) + count
        
        for parent, total in parent_categories.items():
            sub_cats = {k: v for k, v in detailed_distribution.items() if k.startswith(parent + '/')}
            if sub_cats:
                sub_balance = calculate_balance_score(sub_cats)
                if sub_balance < hierarchical_balance_threshold:
                    hierarchical_issues.append(f"{parent} (balance: {sub_balance:.2f})")
        
        if hierarchical_issues:
            recommendations.append(lang("recommendations.hierarchical_imbalance", categories=hierarchical_issues))
    
    # Minimum dataset size recommendations
    if total_images < small_dataset_threshold:
        recommendations.append(lang("recommendations.small_dataset"))
    elif total_images < tiny_dataset_threshold:
        recommendations.append(lang("recommendations.tiny_dataset"))
    
    return balance_score, analysis, recommendations


def calculate_balance_score(label_distribution: Dict[str, int]) -> float:
    """Calculate dataset balance score (0-1, where 1 = perfectly balanced).
    
    Uses coefficient of variation to measure balance.
    Lower variation = higher balance score.
    """
    if not label_distribution or len(label_distribution) < 2:
        return 1.0  # Single class or empty is technically "balanced"
    
    counts = list(label_distribution.values())
    if min(counts) == 0:
        return 0.0  # Any class with 0 samples = completely unbalanced
    
    mean_count = sum(counts) / len(counts)
    variance = sum((count - mean_count) ** 2 for count in counts) / len(counts)
    std_dev = variance ** 0.5
    
    # Coefficient of variation (CV)
    cv = std_dev / mean_count if mean_count > 0 else float('inf')
    
    # Convert CV to balance score (0-1)
    # CV of 0 = perfect balance (score 1.0)
    # CV of 1+ = poor balance (score approaches 0)
    balance_score = max(0.0, 1.0 - cv)
    
    return balance_score


def analyze_dataset_type(dataset_path: str) -> tuple[str, List[str], int, Dict[str, int], float, Dict[str, int], Dict[str, Any], List[str]]:
    """Analyze dataset structure to determine type.
    
    Priority: If >=90% of images have txt files, use txt annotations (multi-label).
    Otherwise, use folder-based labels.
    
    Returns: (dataset_type, labels, image_count, label_distribution, balance_score, detailed_distribution, balance_analysis, recommendations)
    """
    if not os.path.exists(dataset_path):
        return DatasetType.UNKNOWN, [], 0, {}, 0.0, {}, {}, []
    
    image_files = get_image_files(dataset_path)
    image_count = len(image_files)
    
    if image_count == 0:
        return DatasetType.UNKNOWN, [], 0, {}, 0.0, {}, {}, []
    
    # First, check txt file coverage
    txt_count = 0
    for img_file in image_files:
        txt_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(txt_file):
            txt_count += 1
    
    txt_percentage = (txt_count / image_count) * 100 if image_count > 0 else 0
    min_coverage = SETTINGS.get('dataset', {}).get('discovery', {}).get('annotation_formats', {}).get('txt_annotations', {}).get('min_coverage_percent', 90)
    
    if txt_percentage >= min_coverage:
        info(f"Found txt annotation dataset ({txt_percentage:.1f}% coverage >= {min_coverage}%) - using txt-based multi-label")
        # Extract labels and count distribution from txt files
        labels, label_distribution = extract_labels_from_txt_files_with_counts(dataset_path, image_files)
        
        # Also get recursive folder structure for additional insight
        folder_structure = get_recursive_folder_structure(dataset_path)
        if folder_structure:
            info(f"Recursive folder structure: {folder_structure}")
            # Merge folder structure into label distribution for complete picture
            combined_distribution = {**label_distribution, **{f"folder:{k}": v for k, v in folder_structure.items()}}
            # Add folder info to labels
            folder_labels = [f"folder:{k}" for k in folder_structure.keys()]
            combined_labels = labels + folder_labels
        else:
            combined_distribution = label_distribution
            combined_labels = labels
        
        # Comprehensive balance analysis
        balance_score, balance_analysis, recommendations = analyze_balance_and_generate_recommendations(
            label_distribution, combined_distribution, DatasetType.TXT_ANNOTATIONS
        )
        
        info(f"Balance score: {balance_score:.3f}, Recommendations: {len(recommendations)}")
        return (DatasetType.TXT_ANNOTATIONS, combined_labels, image_count, 
               label_distribution, balance_score, combined_distribution, balance_analysis, recommendations)
    
    # Fallback to folder-based labels if txt coverage is insufficient
    info(f"Insufficient txt coverage ({txt_percentage:.1f}% < {min_coverage}%) - using folder-based labels")
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    if subdirs:
        # Check if subdirectories contain images and count them
        folder_with_images = []
        label_distribution = {}
        transformed_label_distribution = {}
        
        for subdir in subdirs:
            subdir_path = os.path.join(dataset_path, subdir)
            subdir_images = get_image_files(subdir_path)
            if subdir_images:
                folder_with_images.append(subdir)
                label_distribution[subdir] = len(subdir_images)
                # Create transformed version for consistency
                transformed_key = subdir.replace(' ', '_').replace('/', '.')
                transformed_label_distribution[transformed_key] = len(subdir_images)
        
        if folder_with_images:
            # Get detailed recursive structure for hierarchical analysis
            detailed_structure = {}
            for folder in folder_with_images:
                folder_path = os.path.join(dataset_path, folder)
                sub_structure = get_recursive_folder_structure(folder_path)
                if sub_structure:
                    # Add sub-folder details
                    for sub_path, count in sub_structure.items():
                        if sub_path != 'root':
                            # Transform the folder name and create hierarchical key
                            transformed_folder = folder.replace(' ', '_').replace('/', '.')
                            detailed_key = f"{transformed_folder}.{sub_path}"
                        else:
                            # Root level folder - just transform the name
                            detailed_key = folder.replace(' ', '_').replace('/', '.')
                        detailed_structure[detailed_key] = count
                else:
                    # No sub-structure, just the main folder
                    transformed_folder = folder.replace(' ', '_').replace('/', '.')
                    detailed_structure[transformed_folder] = transformed_label_distribution[transformed_folder]
            
            # Comprehensive balance analysis
            balance_score, balance_analysis, recommendations = analyze_balance_and_generate_recommendations(
                transformed_label_distribution, detailed_structure, DatasetType.FOLDER_LABELS
            )
            
            info(f"Found folder-based dataset with labels: {folder_with_images}")
            info(f"Detailed structure: {detailed_structure}")
            info(f"Balance score: {balance_score:.3f}, Recommendations: {len(recommendations)}")
            
            # Return comprehensive information
            # Transform folder names for labels list
            transformed_labels = [folder.replace(' ', '_').replace('/', '.') for folder in folder_with_images]
            all_labels = transformed_labels + [k for k in detailed_structure.keys() if '.' in k and k not in transformed_labels]
            
            return (DatasetType.FOLDER_LABELS, all_labels, image_count, 
                   transformed_label_distribution, balance_score, detailed_structure, balance_analysis, recommendations)
    
    # Check for JSON annotations
    json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
    if json_files:
        info(f"Found JSON annotation files: {json_files}")
        labels, label_distribution = extract_labels_from_json_with_counts(os.path.join(dataset_path, json_files[0]))
        
        # Comprehensive balance analysis
        balance_score, balance_analysis, recommendations = analyze_balance_and_generate_recommendations(
            label_distribution, label_distribution, DatasetType.JSON_ANNOTATIONS
        )
        
        info(f"Balance score: {balance_score:.3f}, Recommendations: {len(recommendations)}")
        return (DatasetType.JSON_ANNOTATIONS, labels, image_count, 
               label_distribution, balance_score, label_distribution, balance_analysis, recommendations)
    
    # Check for CSV annotations
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    if csv_files:
        info(f"Found CSV annotation files: {csv_files}")
        labels, label_distribution = extract_labels_from_csv_with_counts(os.path.join(dataset_path, csv_files[0]))
        
        # Also get recursive folder structure for additional insight
        folder_structure = get_recursive_folder_structure(dataset_path)
        if folder_structure:
            info(f"Recursive folder structure: {folder_structure}")
            # Merge folder structure into label distribution
            combined_distribution = {**label_distribution, **{f"folder:{k}": v for k, v in folder_structure.items()}}
            # Add folder info to labels
            folder_labels = [f"folder:{k}" for k in folder_structure.keys()]
            combined_labels = labels + folder_labels
        else:
            combined_distribution = label_distribution
            combined_labels = labels
        
        # Comprehensive balance analysis
        balance_score, balance_analysis, recommendations = analyze_balance_and_generate_recommendations(
            label_distribution, combined_distribution, DatasetType.CSV_ANNOTATIONS
        )
        
        info(f"Balance score: {balance_score:.3f}, Recommendations: {len(recommendations)}")
        return (DatasetType.CSV_ANNOTATIONS, combined_labels, image_count, 
               label_distribution, balance_score, combined_distribution, balance_analysis, recommendations)
    
    warning(f"Unknown dataset type in {dataset_path}")
    return DatasetType.UNKNOWN, [], image_count, {}, 0.0, {}, {}, []


def get_recursive_folder_structure(directory: str) -> Dict[str, int]:
    """Get recursive folder structure with image counts for deepest folders.
    
    This helps understand the sub-structure in txt/csv annotation datasets.
    Returns mapping of folder_path -> image_count for leaf folders containing images.
    
    For folder_labels datasets, transforms folder names:
    - Spaces become underscores: " " -> "_"
    - Slashes become dots: "/" -> "."
    Example: "awesome fursuits/tania" -> "awesome_fursuits.tania"
    """
    folder_counts = {}
    
    if not os.path.exists(directory):
        return folder_counts
    
    for root, dirs, files in os.walk(directory):
        # Check if this is a leaf directory (contains images but no subdirs with images)
        has_images = any(file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')) 
                        for file in files)
        
        if has_images:
            # Check if any subdirectory also has images
            has_subdir_with_images = False
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                subdir_files = os.listdir(subdir_path) if os.path.exists(subdir_path) else []
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')) 
                       for f in subdir_files):
                    has_subdir_with_images = True
                    break
            
            # If this folder has images and no subfolders with images, count it
            if not has_subdir_with_images:
                rel_path = os.path.relpath(root, directory)
                if rel_path == '.':
                    rel_path = 'root'
                
                # Transform folder path for folder_labels datasets
                # Spaces become underscores, slashes become dots
                transformed_path = rel_path.replace(' ', '_').replace('/', '.')
                
                image_count = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'))])
                if image_count > 0:
                    folder_counts[transformed_path] = image_count
    
    return folder_counts


def extract_labels_from_txt_files_with_counts(dataset_path: str, image_files: List[str]) -> tuple[List[str], Dict[str, int]]:
    """Extract unique labels and their counts from txt files.
    
    Assumes txt files contain comma-separated labels, one line per image.
    Each label occurrence is counted (multi-label support).
    """
    label_counts = {}
    
    for img_file in image_files:
        txt_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # Split by comma and strip whitespace
                            labels_in_file = [label.strip() for label in line.split(',') if label.strip()]
                            for label in labels_in_file:
                                label_counts[label] = label_counts.get(label, 0) + 1
            except Exception as e:
                warning(f"Error reading {txt_file}: {e}")
    
    labels = sorted(list(label_counts.keys()))
    return labels, label_counts


def extract_labels_from_json_with_counts(json_file: str) -> tuple[List[str], Dict[str, int]]:
    """Extract labels and their counts from JSON annotation file."""
    label_counts = {}
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Try different JSON structures
        if 'categories' in data:  # COCO format
            for cat in data['categories']:
                label = cat.get('name', str(cat.get('id', '')))
                # Count annotations for this category
                count = 0
                if 'annotations' in data:
                    count = sum(1 for ann in data['annotations'] 
                              if ann.get('category_id') == cat.get('id'))
                label_counts[label] = count
        elif 'labels' in data:  # Custom format with counts
            if isinstance(data['labels'], dict):
                label_counts.update(data['labels'])
            else:
                for label in data['labels']:
                    label_counts[label] = label_counts.get(label, 0) + 1
        elif isinstance(data, list):  # List of annotations
            for item in data:
                if 'label' in item:
                    label = item['label']
                elif 'category' in item:
                    label = item['category']
                else:
                    continue
                label_counts[label] = label_counts.get(label, 0) + 1
    
    except Exception as e:
        warning(f"Error reading JSON file {json_file}: {e}")
    
    labels = sorted(list(label_counts.keys()))
    return labels, label_counts


def extract_labels_from_csv_with_counts(csv_file: str) -> tuple[List[str], Dict[str, int]]:
    """Extract labels and their counts from CSV annotation file."""
    label_counts = {}
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) >= 2:
                label = parts[1].strip('"')  # Assume label is second column
                label_counts[label] = label_counts.get(label, 0) + 1
    
    except Exception as e:
        warning(f"Error reading CSV file {csv_file}: {e}")
    
    labels = sorted(list(label_counts.keys()))
    return labels, label_counts


def extract_labels_from_txt_files(dataset_path: str, sample_files: List[str]) -> List[str]:
    """Extract unique labels from sample txt files."""
    labels = set()
    
    for img_file in sample_files:
        txt_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            # Assume first part is class ID or label
                            labels.add(parts[0])
            except Exception as e:
                warning(f"Error reading {txt_file}: {e}")
    
    return sorted(list(labels))


def extract_labels_from_json(json_file: str) -> List[str]:
    """Extract labels from JSON annotation file."""
    labels = set()
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Try different JSON structures
        if 'categories' in data:  # COCO format
            labels.update(cat.get('name', str(cat.get('id', ''))) for cat in data['categories'])
        elif 'labels' in data:  # Custom format
            labels.update(data['labels'])
        elif isinstance(data, list):  # List of annotations
            for item in data:
                if 'label' in item:
                    labels.add(item['label'])
                elif 'category' in item:
                    labels.add(item['category'])
    
    except Exception as e:
        warning(f"Error reading JSON file {json_file}: {e}")
    
    return sorted(list(labels))


def extract_labels_from_csv(csv_file: str) -> List[str]:
    """Extract labels from CSV annotation file."""
    labels = set()
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) >= 2:
                labels.add(parts[1].strip('"'))  # Assume label is second column
    
    except Exception as e:
        warning(f"Error reading CSV file {csv_file}: {e}")
    
    return sorted(list(labels))


def discover_projects() -> List[ProjectInfo]:
    """Discover all projects in the configured projects directory."""
    config = SETTINGS.get("paths", {})
    models_dir = config.get("projects_dir") or config.get("models_dir", "projects")
    
    # Get absolute path
    base_path = os.path.dirname(os.path.dirname(__file__))
    models_path = os.path.join(base_path, models_dir)
    
    info(f"Scanning for projects in: {models_path}")
    
    projects = []
    
    if not os.path.exists(models_path):
        warning(f"Projects directory not found: {models_path}")
        return projects
    
    # Scan for project directories
    for item in os.listdir(models_path):
        item_path = os.path.join(models_path, item)
        
        if os.path.isdir(item_path):
            project = ProjectInfo(item, item_path)
            
            # Check for dataset and model directories
            project.has_dataset = os.path.exists(project.dataset_path)
            project.has_model_dir = os.path.exists(project.model_path)
            
            if project.has_dataset:
                # Analyze dataset with comprehensive analysis
                (dataset_type, labels, image_count, label_distribution, balance_score, 
                 detailed_distribution, balance_analysis, recommendations) = analyze_dataset_type(project.dataset_path)
                
                project.dataset_type = dataset_type
                project.labels = labels
                project.image_count = image_count
                project.label_distribution = label_distribution
                project.detailed_distribution = detailed_distribution
                project.balance_score = balance_score
                project.balance_analysis = balance_analysis
                project.recommendations = recommendations
                
                info(f"Project '{item}': {dataset_type}, {image_count} images, balance: {balance_score:.3f}")
                info(f"Recommendations: {len(recommendations)} suggestions")
            else:
                warning(f"Project '{item}': No dataset directory found")
            
            projects.append(project)
    
    return projects


def get_project_info(project_name: str) -> Optional[ProjectInfo]:
    """Get detailed information about a specific project."""
    projects = discover_projects()
    
    for project in projects:
        if project.name == project_name:
            return project
    
    return None


def get_discovery_summary() -> Dict[str, Any]:
    """Get summary of all discovered projects."""
    projects = discover_projects()
    
    summary = {
        "total_projects": len(projects),
        "projects_with_datasets": len([p for p in projects if p.has_dataset]),
        "total_images": sum(p.image_count for p in projects),
        "dataset_types": {},
        "projects": [p.to_dict() for p in projects]
    }
    
    # Count dataset types
    for project in projects:
        if project.dataset_type != DatasetType.UNKNOWN:
            summary["dataset_types"][project.dataset_type] = \
                summary["dataset_types"].get(project.dataset_type, 0) + 1
    
    return summary
