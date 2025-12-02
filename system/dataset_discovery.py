import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS


DEFAULT_BALANCE_SCORE_THRESHOLDS: Dict[str, float] = {
    "legendary": 0.98,
    "excellent": 0.95,
    "very_good": 0.90,
    "good": 0.85,
    "balanced": 0.80,
    "slightly_unbalanced": 0.70,
    "fair": 0.60,
    "poor": 0.45,
    "very_poor": 0.35,
    "critical": 0.25
}


def resolve_balance_score_thresholds(custom_thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    thresholds = DEFAULT_BALANCE_SCORE_THRESHOLDS.copy()
    if isinstance(custom_thresholds, dict):
        for key, value in custom_thresholds.items():
            if isinstance(value, (int, float)):
                thresholds[key] = float(value)
    return thresholds


def get_configured_balance_thresholds() -> Dict[str, float]:
    try:
        custom = SETTINGS['dataset']['discovery']['balance_analysis'].get('balance_score_thresholds', {})
    except Exception:
        custom = {}
    return resolve_balance_score_thresholds(custom)


def format_status_label(key: str) -> str:
    return key.replace('_', ' ').title()


def classify_balance_status(balance_score: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    effective_thresholds = thresholds or get_configured_balance_thresholds()
    if not effective_thresholds:
        return "Critical"
    sorted_thresholds = sorted(effective_thresholds.items(), key=lambda item: item[1], reverse=True)
    for status_key, min_score in sorted_thresholds:
        if balance_score >= min_score:
            return format_status_label(status_key)
    return format_status_label(sorted_thresholds[-1][0])


class DatasetType:
    MULTI_LABEL = "multi_label"           # txt files with short comma-separated tags (80%+ short tags)
    FOLDER_CLASSIFICATION = "folder_classification"  # images organized in folders, minimal txt (<30%)
    ANNOTATION = "annotation"             # txt files with sentences/long descriptions
    MIXED = "mixed"                       # mixture of folders and varied txt content
    CLEAN = "clean"                       # just images, no folders, no txt
    UNKNOWN = "unknown"


class ProjectInfo:
    
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.dataset_path = os.path.join(path, "dataset")
        self.model_path = os.path.join(path, "model")
        self.dataset_type = DatasetType.UNKNOWN
        self.image_count = 0
        self.labels = []
        self.label_distribution = {}
        self.detailed_distribution = {}
        self.balance_score = 0.0
        self.balance_analysis = {}
        self.recommendations = []
        self.has_dataset = False
        self.has_model_dir = False
    
    def to_dict(self) -> Dict[str, Any]:
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
        return classify_balance_status(self.balance_score)


def get_image_files(directory: str) -> List[str]:
    try:
        image_extensions = SETTINGS['dataset']['image_extensions']
    except Exception:
        raise ValueError("Missing 'dataset.image_extensions' in config/config.json")
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
    
    if not label_distribution:
        return 1.0, {}, []
    
    try:
        balance_config = SETTINGS['dataset']['discovery']['balance_analysis']
    except Exception:
        raise ValueError("Missing 'dataset.discovery.balance_analysis' in config/config.json")
    balance_thresholds = resolve_balance_score_thresholds(balance_config.get('balance_score_thresholds'))
    min_images_per_class = balance_config['min_images_per_class']
    critical_shortage_threshold = balance_config['critical_shortage_threshold']
    over_representation_ratio = balance_config['over_representation_ratio']
    under_representation_ratio = balance_config['under_representation_ratio']
    severe_over_representation_ratio = balance_config['severe_over_representation_ratio']
    hierarchical_balance_threshold = balance_config['hierarchical_balance_threshold']
    dataset_size_warnings = balance_config['dataset_size_warnings']
    tiny_dataset_threshold = dataset_size_warnings['tiny_dataset']
    small_dataset_threshold = dataset_size_warnings['small_dataset']
    
    total_images = sum(label_distribution.values())
    num_labels = len(label_distribution)
    
    ideal_per_label = total_images / num_labels
    
    balance_score = calculate_balance_score(label_distribution)
    
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
    
    for label, count in label_distribution.items():
        ratio = count / ideal_per_label
        if ratio > over_representation_ratio:
            analysis["over_represented"].append({"label": label, "count": count, "ratio": round(ratio, 2)})
        elif ratio < under_representation_ratio:
            analysis["under_represented"].append({"label": label, "count": count, "ratio": round(ratio, 2)})
        if count < critical_shortage_threshold:
            analysis["critical_shortage"].append({"label": label, "count": count})
    
    recommendations = []
    
    if analysis["critical_shortage"]:
        critical_labels = [item["label"] for item in analysis["critical_shortage"]]
        recommendations.append(f"Critical data shortage detected in: {critical_labels}")
    
    if analysis["over_represented"]:
        over_labels = [item["label"] for item in analysis["over_represented"] if item["ratio"] > severe_over_representation_ratio]
        if over_labels:
            recommendations.append(f"Consider reducing oversampled labels: {over_labels}")
    
    if analysis["under_represented"]:
        under_labels = [item["label"] for item in analysis["under_represented"]]
        recommendations.append(f"Consider augmenting undersampled labels: {under_labels}")
    
    intervention_threshold = balance_thresholds.get('poor', balance_thresholds.get('very_poor', 0.35))
    if balance_score < intervention_threshold:
        recommendations.append("Consider using weighted loss functions")
        recommendations.append("Consider using stratified sampling")
    
    if dataset_type == DatasetType.FOLDER_CLASSIFICATION and detailed_distribution:
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
            recommendations.append(f"Hierarchical imbalance detected in: {hierarchical_issues}")
    
    if total_images < small_dataset_threshold:
        recommendations.append("Small dataset detected - consider data augmentation")
    elif total_images < tiny_dataset_threshold:
        recommendations.append("Very small dataset - high risk of overfitting")
    
    return balance_score, analysis, recommendations


def calculate_balance_score(label_distribution: Dict[str, int]) -> float:
    if not label_distribution or len(label_distribution) < 2:
        return 1.0
    
    counts = list(label_distribution.values())
    if min(counts) == 0:
        return 0.0
    
    mean_count = sum(counts) / len(counts)
    variance = sum((count - mean_count) ** 2 for count in counts) / len(counts)
    std_dev = variance ** 0.5
    
    cv = std_dev / mean_count if mean_count > 0 else float('inf')
    
    balance_score = max(0.0, 1.0 - cv)
    
    return balance_score


def _analyze_txt_content(txt_files: List[str], sample_size: int = 50) -> dict:
    """
    Analyze txt file content to determine tag vs annotation style.
    Returns stats about the content patterns.
    """
    if not txt_files:
        return {"has_txt": False}
    
    sampled = txt_files[:sample_size] if len(txt_files) > sample_size else txt_files
    
    total_entries = 0
    short_tag_entries = 0  # 1-2 words
    long_entries = 0       # 3+ words or sentence-like
    comma_separated_files = 0
    
    for txt_path in sampled:
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if not content:
                    continue
                
                # Check if comma-separated
                if ',' in content:
                    comma_separated_files += 1
                    tags = [t.strip() for t in content.split(',') if t.strip()]
                else:
                    tags = [content]
                
                for tag in tags:
                    total_entries += 1
                    word_count = len(tag.split())
                    if word_count <= 2:
                        short_tag_entries += 1
                    else:
                        long_entries += 1
        except Exception:
            continue
    
    short_tag_ratio = short_tag_entries / total_entries if total_entries > 0 else 0
    comma_ratio = comma_separated_files / len(sampled) if sampled else 0
    
    return {
        "has_txt": True,
        "total_entries": total_entries,
        "short_tag_ratio": short_tag_ratio,
        "long_entry_ratio": 1 - short_tag_ratio,
        "comma_separated_ratio": comma_ratio
    }


def analyze_dataset_type(dataset_path: str) -> tuple[str, List[str], int, Dict[str, int], float, Dict[str, int], Dict[str, Any], List[str]]:
    """
    Analyze dataset and determine its type (priority order):
    1. MULTI_LABEL: txt files with tags present (highest priority if txt coverage is good)
    2. ANNOTATION: txt files with sentence-like content
    3. FOLDER_CLASSIFICATION: images organized in folders without significant txt
    4. CLEAN: just images, no organization
    5. MIXED: varied content that doesn't fit other categories
    """
    if not os.path.exists(dataset_path):
        return DatasetType.UNKNOWN, [], 0, {}, 0.0, {}, {}, []
    
    image_files = get_image_files(dataset_path)
    image_count = len(image_files)
    
    if image_count == 0:
        return DatasetType.UNKNOWN, [], 0, {}, 0.0, {}, {}, []
    
    txt_count = 0
    txt_files_list = []
    for img_file in image_files:
        txt_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(txt_file):
            txt_count += 1
            txt_files_list.append(txt_file)
    
    txt_percentage = (txt_count / image_count) * 100 if image_count > 0 else 0
    
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')]
    
    has_folder_structure = False
    folder_with_images = []
    folder_label_distribution = {}
    
    if subdirs:
        for subdir in subdirs:
            subdir_path = os.path.join(dataset_path, subdir)
            subdir_images = get_image_files(subdir_path)
            if subdir_images:
                folder_with_images.append(subdir)
                transformed_key = subdir.replace(' ', '_').replace('/', '.')
                folder_label_distribution[transformed_key] = len(subdir_images)
        has_folder_structure = len(folder_with_images) > 0
    
    txt_analysis = _analyze_txt_content(txt_files_list) if txt_files_list else None
    
    # PRIORITY 1: If we have excellent txt coverage (>80%) -> MULTI_LABEL or ANNOTATION
    if txt_analysis and txt_percentage >= 80:
        labels, label_distribution = extract_labels_from_txt_files_with_counts(dataset_path, image_files)
        
        folder_structure = get_recursive_folder_structure(dataset_path)
        if folder_structure:
            combined_distribution = {**label_distribution, **{f"folder:{k}": v for k, v in folder_structure.items()}}
            folder_labels = [f"folder:{k}" for k in folder_structure.keys()]
            combined_labels = labels + folder_labels
        else:
            combined_distribution = label_distribution
            combined_labels = labels
        
        # MULTI_LABEL: >80% short tags, ANNOTATION: <30% short tags
        if txt_analysis['short_tag_ratio'] >= 0.8:
            dataset_type = DatasetType.MULTI_LABEL
        elif txt_analysis['short_tag_ratio'] < 0.3:
            dataset_type = DatasetType.ANNOTATION
        else:
            dataset_type = DatasetType.MIXED
        
        balance_score, balance_analysis, recommendations = analyze_balance_and_generate_recommendations(
            label_distribution, combined_distribution, dataset_type
        )
        
        return (dataset_type, combined_labels, image_count, 
               label_distribution, balance_score, combined_distribution, balance_analysis, recommendations)
    
    # PRIORITY 2: Folder-based classification (no significant txt)
    if has_folder_structure:
        detailed_structure = {}
        for folder in folder_with_images:
            folder_path = os.path.join(dataset_path, folder)
            sub_structure = get_recursive_folder_structure(folder_path)
            transformed_folder = folder.replace(' ', '_').replace('/', '.')
            if sub_structure:
                for sub_path, count in sub_structure.items():
                    if sub_path != 'root':
                        detailed_key = f"{transformed_folder}.{sub_path}"
                    else:
                        detailed_key = transformed_folder
                    detailed_structure[detailed_key] = count
            else:
                detailed_structure[transformed_folder] = folder_label_distribution[transformed_folder]
        
        balance_score, balance_analysis, recommendations = analyze_balance_and_generate_recommendations(
            folder_label_distribution, detailed_structure, DatasetType.FOLDER_CLASSIFICATION
        )
        
        transformed_labels = list(folder_label_distribution.keys())
        all_labels = transformed_labels + [k for k in detailed_structure.keys() if '.' in k and k not in transformed_labels]
        
        return (DatasetType.FOLDER_CLASSIFICATION, all_labels, image_count, 
               folder_label_distribution, balance_score, detailed_structure, balance_analysis, recommendations)
    
    # PRIORITY 3: Clean dataset (just images, no structure)
    if txt_percentage < 5:
        return DatasetType.CLEAN, [], image_count, {}, 0.0, {}, {}, []
    
    # Fallback: MIXED
    warning(f"Mixed/unknown dataset type in {dataset_path}")
    return DatasetType.MIXED, [], image_count, {}, 0.0, {}, {}, []


def get_recursive_folder_structure(directory: str) -> Dict[str, int]:
    folder_counts = {}
    
    if not os.path.exists(directory):
        return folder_counts
    
    for root, dirs, files in os.walk(directory):
        has_images = any(file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')) 
                        for file in files)
        
        if has_images:
            has_subdir_with_images = False
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                subdir_files = os.listdir(subdir_path) if os.path.exists(subdir_path) else []
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')) 
                       for f in subdir_files):
                    has_subdir_with_images = True
                    break
            
            if not has_subdir_with_images:
                rel_path = os.path.relpath(root, directory)
                if rel_path == '.':
                    rel_path = 'root'
                
                transformed_path = rel_path.replace(' ', '_').replace('/', '.')
                
                image_count = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'))])
                if image_count > 0:
                    folder_counts[transformed_path] = image_count
    
    return folder_counts


def extract_labels_from_txt_files_with_counts(dataset_path: str, image_files: List[str]) -> tuple[List[str], Dict[str, int]]:
    label_counts = {}
    
    for img_file in image_files:
        txt_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            labels_in_file = [label.strip() for label in line.split(',') if label.strip()]
                            for label in labels_in_file:
                                label_counts[label] = label_counts.get(label, 0) + 1
            except Exception as e:
                warning(f"Error reading {txt_file}: {e}")
    
    labels = sorted(list(label_counts.keys()))
    return labels, label_counts


def extract_labels_from_json_with_counts(json_file: str) -> tuple[List[str], Dict[str, int]]:
    label_counts = {}
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'categories' in data:  # COCO format
            for cat in data['categories']:
                label = cat.get('name', str(cat.get('id', '')))
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
    labels = set()
    
    for img_file in sample_files:
        txt_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            labels.add(parts[0])
            except Exception as e:
                warning(f"Error reading {txt_file}: {e}")
    
    return sorted(list(labels))


def extract_labels_from_json(json_file: str) -> List[str]:
    labels = set()
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
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


def discover_projects(compute_stats: bool = False) -> List[ProjectInfo]:
    """
    Discover all projects in the projects directory.
    
    Args:
        compute_stats: If False (default), only basic info is gathered (fast).
                      If True, full statistics are computed (expensive).
    """
    try:
        paths_cfg = SETTINGS['paths']
    except Exception:
        raise ValueError("Missing 'paths' configuration in config/config.json")
    models_dir = paths_cfg.get("projects_dir") or paths_cfg.get("models_dir")
    if not models_dir:
        raise ValueError("Missing required 'paths.projects_dir' or 'paths.models_dir' in config/config.json")
    
    base_path = os.path.dirname(os.path.dirname(__file__))
    models_path = os.path.join(base_path, models_dir)
    
    projects = []
    
    if not os.path.exists(models_path):
        warning(f"Projects directory not found: {models_path}")
        return projects
    
    for item in os.listdir(models_path):
        item_path = os.path.join(models_path, item)
        
        if os.path.isdir(item_path):
            project = ProjectInfo(item, item_path)
            
            project.has_dataset = os.path.exists(project.dataset_path)
            project.has_model_dir = os.path.exists(project.model_path)
            
            if project.has_dataset:
                if compute_stats:
                    # Full analysis - expensive (for balance/labels only, not dataset_type)
                    (_, labels, image_count, label_distribution, balance_score, 
                     detailed_distribution, balance_analysis, recommendations) = analyze_dataset_type(project.dataset_path)
                    
                    # dataset_type is set by user, not auto-detected
                    project.dataset_type = DatasetType.UNKNOWN
                    project.labels = labels
                    project.image_count = image_count
                    project.label_distribution = label_distribution
                    project.detailed_distribution = detailed_distribution
                    project.balance_score = balance_score
                    project.balance_analysis = balance_analysis
                    project.recommendations = recommendations
                else:
                    # Lightweight - just count images
                    image_files = get_image_files(project.dataset_path)
                    project.image_count = len(image_files)
                    # dataset_type is set by user via UI, not auto-detected
                    project.dataset_type = DatasetType.UNKNOWN
            
            projects.append(project)
    
    return projects


def _detect_dataset_type_fast(dataset_path: str) -> str:
    """Quick detection of dataset type without full analysis."""
    if not os.path.exists(dataset_path):
        return DatasetType.UNKNOWN
    
    has_folders = False
    has_txt = False
    image_count = 0
    txt_count = 0
    
    try:
        entries = os.listdir(dataset_path)
    except OSError:
        return DatasetType.UNKNOWN
    
    for entry in entries[:50]:
        entry_path = os.path.join(dataset_path, entry)
        if os.path.isdir(entry_path) and not entry.startswith('.'):
            for f in os.listdir(entry_path)[:10]:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')):
                    has_folders = True
                    break
        elif entry.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')):
            image_count += 1
        elif entry.lower().endswith('.txt'):
            txt_count += 1
            has_txt = True
    
    if image_count == 0 and not has_folders:
        return DatasetType.UNKNOWN
    
    txt_ratio = txt_count / max(image_count, 1)
    
    if has_folders and txt_ratio < 0.3:
        return DatasetType.FOLDER_CLASSIFICATION
    
    if has_txt and txt_ratio >= 0.3:
        return DatasetType.MULTI_LABEL
    
    if not has_folders and not has_txt and image_count > 0:
        return DatasetType.CLEAN
    
    return DatasetType.MIXED


def get_project_info(project_name: str, compute_stats: bool = False) -> Optional[ProjectInfo]:
    """
    Get info for a specific project.
    
    Args:
        project_name: Name of the project
        compute_stats: If True, compute full statistics (expensive)
    """
    projects = discover_projects(compute_stats=compute_stats)
    
    for project in projects:
        if project.name == project_name:
            return project
    
    return None


def compute_project_stats(project_name: str) -> Optional[Dict[str, Any]]:
    """
    Compute full statistics for a project and return as dict.
    This is the expensive operation that should only be called explicitly.
    """
    project = get_project_info(project_name, compute_stats=True)
    if project is None:
        return None
    return project.to_dict()


def get_discovery_summary() -> Dict[str, Any]:
    projects = discover_projects()
    
    summary = {
        "total_projects": len(projects),
        "projects_with_datasets": len([p for p in projects if p.has_dataset]),
        "total_images": sum(p.image_count for p in projects),
        "dataset_types": {},
        "projects": [p.to_dict() for p in projects]
    }
    
    for project in projects:
        if project.dataset_type != DatasetType.UNKNOWN:
            summary["dataset_types"][project.dataset_type] = \
                summary["dataset_types"].get(project.dataset_type, 0) + 1
    
    return summary
