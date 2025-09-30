# Projects Guide

This guide explains the Projects system in Hootsight, which organizes datasets, configurations, and training artifacts for machine learning projects.

## What is the Projects System?

The Projects system is Hootsight's way of organizing your work. Each project is a self-contained workspace that includes:

- **Dataset folder**: Your training images and labels
- **Configuration**: Project-specific training settings
- **Model artifacts**: Trained models and checkpoints
- **Heatmaps**: Visualization outputs for model interpretation
- **Validation results**: Performance metrics and test results

Think of a project as a complete AI training workspace where everything related to one specific task lives together.

## Project Structure

When you create a project, Hootsight automatically creates this folder structure:

```
projects/your_project_name/
├── dataset/           # Your training images and labels
├── data_source/       # Original data before preprocessing
├── model/            # Trained models and checkpoints
├── heatmaps/         # Grad-CAM visualizations
├── validation/       # Test results and metrics
└── config.json       # Project-specific configuration
```

## Projects Page Overview

The Projects page is your project management hub. Here's what you'll find:

### Empty State
When you first start Hootsight or have no projects:
- **Empty message**: "No projects yet"
- **Description**: Explains that projects scaffold dataset, data source, and model folders
- **Create New Project button**: Primary action to get started

### Project Cards
Each project appears as a card showing:

**Project Information**:
- **Project name**: The identifier for your project
- **Images count**: Number of training images found
- **Labels count**: Number of distinct labels/classes
- **Balance Score**: Data balance quality (0-1 scale)
- **Balance Status**: Human-readable balance assessment (Excellent, Good, Fair, Poor, Critical)
- **Dataset Type**: How labels are organized (folder_labels, txt_annotations, etc.)

**Visual Indicators**:
- **Current Project badge**: Shows which project is currently loaded
- **Training Active indicator**: Highlights projects currently being trained
- **Training status**: Shows current training phase if active

**Action Buttons**:
- **Load**: Makes this project active for training configuration
- **Start Training**: Begins model training (requires loaded project)
- **Stop Training**: Halts active training for this project

## Core Operations

### Creating a New Project

**How to create**:
1. Click "Create New Project" (in header or empty state)
2. Enter a project name in the dialog
3. Click "Create"

**Project naming rules**:
- **Length**: 3-64 characters
- **Characters**: Letters, numbers, hyphens, underscores only
- **Pattern**: Must start with letter or number
- **Uniqueness**: No duplicate names allowed

**What happens during creation**:
1. **Folder structure**: Creates all necessary directories
2. **Configuration**: Copies global config as starting point
3. **Auto-loading**: Automatically loads the new project
4. **Project refresh**: Updates the projects list
5. **Status feedback**: Shows creation progress and results

### Loading a Project

**Purpose**: Makes a project active for configuration and training.

**What happens when you load**:
1. **Project selection**: Sets this as the current project
2. **Configuration loading**: 
   - Tries to load project-specific `config.json`
   - Falls back to global config if project config missing
   - Merges project labels into configuration
3. **UI updates**:
   - Refreshes Dataset and Training pages
   - Updates project cards to show current selection
   - Enables training-related actions
4. **Status tracking**: Connects to any active training for this project

**System impact**: Only one project can be loaded at a time. Loading a new project replaces the current one.

### Starting Training

**Prerequisites**:
- Project must be loaded first
- No other training can be active for this project
- Valid dataset must exist in project folder

**What happens during start**:
1. **Validation**: Checks project configuration and dataset
2. **Training launch**: Sends request to start background training
3. **Status tracking**: 
   - Creates training ID for monitoring
   - Updates UI to show training progress
   - Enables training controls
4. **UI updates**:
   - Disables "Start Training" button
   - Enables "Stop Training" button
   - Shows training status in project card

**Training process**: Runs in background, allowing continued UI use.

### Stopping Training

**When available**: Only when training is active for the project.

**What happens during stop**:
1. **Stop request**: Sends termination signal to training process
2. **Cleanup**: Safely stops current training epoch
3. **Status update**: Updates training status and UI
4. **Button states**: Re-enables "Start Training", disables "Stop Training"

**Safety**: Training stops safely after completing the current batch.

## Project Discovery and Analysis

### Automatic Discovery
Hootsight automatically scans the projects directory and analyzes:

**Dataset Detection**:
- **Image count**: Finds all supported image formats
- **Label extraction**: Identifies labels based on folder structure or annotation files
- **Dataset type**: Determines organization method (folders, txt files, JSON, CSV)

**Balance Analysis**:
- **Label distribution**: Counts images per label
- **Balance scoring**: Calculates 0-1 balance score
- **Recommendations**: Suggests improvements for imbalanced datasets

**Project Health**:
- **Folder structure**: Verifies required directories exist
- **Configuration**: Checks for valid project config
- **Model artifacts**: Identifies existing trained models

### Dataset Types Supported

**Folder Labels** (`folder_labels`):
- Each subfolder in `dataset/` represents a class
- Images directly in class folders
- Example: `dataset/cats/`, `dataset/dogs/`

**Text Annotations** (`txt_annotations`):
- Each image has corresponding `.txt` file with labels
- Supports multi-label classification
- Example: `image1.jpg` + `image1.txt`

**JSON Annotations** (`json_annotations`):
- Single JSON file with image-label mappings
- Structured annotation format

**CSV Annotations** (`csv_annotations`):
- CSV file with image,label pairs
- Simple tabular format

### Balance Assessment

**Balance Score Calculation**:
- **Perfect balance (1.0)**: All labels have equal representation
- **Good balance (0.7-0.9)**: Minor imbalances, acceptable for training
- **Fair balance (0.5-0.7)**: Noticeable imbalances, may affect performance
- **Poor balance (0.3-0.5)**: Significant imbalances, requires attention
- **Critical balance (0.0-0.3)**: Severe imbalances, will hurt model quality

**Status Indicators**:
- **Excellent**: 90%+ balance, ready for training
- **Good**: 70-90% balance, good for most cases
- **Fair**: 50-70% balance, consider data augmentation
- **Poor**: 30-50% balance, add more data for minority classes
- **Critical**: <30% balance, requires immediate data balancing

## Project Configuration

### Configuration Inheritance
Projects use a hierarchical configuration system:

1. **Global config**: Base settings in `config/config.json`
2. **Project config**: Overrides in `projects/name/config.json`
3. **Runtime merging**: Project settings override global ones

### Project-Specific Settings
You can customize per project:
- **Training parameters**: Learning rates, batch sizes, epochs
- **Model selection**: Different architectures per project
- **Augmentation**: Project-specific data augmentation
- **Optimization**: Custom optimizers and schedulers

### Configuration Persistence
Changes to training settings are saved to the project's `config.json`, not the global config.

## Integration with Other Features

### Training System
- Projects provide the dataset and configuration for training
- Training artifacts are saved to the project's model directory
- Each project can have independent training sessions

### Heatmap Generation
- Heatmaps are generated per project using its trained model
- Outputs saved to project's heatmaps directory
- Visualization tied to specific project datasets

### Dataset Analysis
- Dataset page shows information for the currently loaded project
- Analysis updates when switching projects
- Project-specific dataset recommendations

## Best Practices

### Project Organization
1. **Descriptive names**: Use clear, descriptive project names
2. **Consistent structure**: Keep similar projects organized similarly
3. **Data isolation**: One dataset per project for clarity
4. **Configuration documentation**: Comment significant config changes

### Workflow Recommendations
1. **Create project first**: Always start with project creation
2. **Prepare dataset**: Add and organize images before training
3. **Review balance**: Check dataset balance before starting training
4. **Test configurations**: Experiment with settings on small datasets first
5. **Monitor training**: Watch progress and stop if issues arise

### Data Management
1. **Backup important projects**: Copy project folders for safekeeping
2. **Clean unused projects**: Remove test projects to save space
3. **Document experiments**: Keep notes about what works
4. **Version datasets**: Track changes to training data

## Troubleshooting

### Common Issues

**Project won't load**:
- Check project folder structure exists
- Verify dataset folder has images
- Look for configuration errors in project config

**Training won't start**:
- Ensure project is loaded first
- Check dataset has valid images and labels
- Verify no other training is active

**Missing project data**:
- Projects are stored in the configured projects directory
- Check `paths.projects_dir` in global config
- Ensure directory permissions allow read/write

**Balance scores seem wrong**:
- Balance calculation requires labeled data
- Empty or unlabeled datasets show 1.0 balance
- Check that labels are properly detected

### Recovery Steps
1. **Restart application**: Fixes most UI state issues
2. **Check file permissions**: Ensure Hootsight can read/write project folders
3. **Verify configuration**: Look for JSON syntax errors in config files
4. **Recreate project**: Delete and recreate if corruption suspected

The Projects system is designed to keep your work organized and make it easy to manage multiple AI training tasks. Each project is independent, so you can experiment freely without affecting other work.

_Page created by Roxxy (AI) – 2025-10-01._