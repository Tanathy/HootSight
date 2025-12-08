/**
 * HootSight - API Endpoints
 * Centralized endpoint definitions for frontend-backend communication
 */

const API = {
    /**
     * Base URL for API calls
     */
    baseUrl: '',

    /**
     * Projects endpoints
     */
    projects: {
        /**
         * List all projects
         * @returns {Promise<Object>} - List of projects
         */
        list: async function() {
            const response = await fetch(`${API.baseUrl}/projects`);
            if (!response.ok) throw new Error(`Failed to fetch projects: ${response.statusText}`);
            return response.json();
        },

        /**
         * Get project details
         * @param {string} name - Project name
         * @returns {Promise<Object>} - Project details
         */
        get: async function(name) {
            const response = await fetch(`${API.baseUrl}/projects/${encodeURIComponent(name)}`);
            if (!response.ok) throw new Error(`Failed to fetch project: ${response.statusText}`);
            return response.json();
        },

        /**
         * Compute/refresh project statistics
         * @param {string} name - Project name
         * @returns {Promise<Object>} - Computed statistics
         */
        computeStats: async function(name) {
            const response = await fetch(`${API.baseUrl}/projects/${encodeURIComponent(name)}/stats`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error(`Failed to compute stats: ${response.statusText}`);
            return response.json();
        },

        /**
         * Get project-specific configuration
         * @param {string} name - Project name
         * @returns {Promise<Object>} - Project config
         */
        getConfig: async function(name) {
            const response = await fetch(`${API.baseUrl}/projects/${encodeURIComponent(name)}/config`);
            if (!response.ok) throw new Error(`Failed to fetch project config: ${response.statusText}`);
            return response.json();
        },

        /**
         * Save project-specific configuration
         * @param {string} name - Project name
         * @param {Object} config - Configuration to save
         * @returns {Promise<Object>} - Result
         */
        saveConfig: async function(name, config) {
            const response = await fetch(`${API.baseUrl}/projects/${encodeURIComponent(name)}/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            if (!response.ok) throw new Error(`Failed to save project config: ${response.statusText}`);
            return response.json();
        },

        /**
         * Create a new project
         * @param {string} name - Project name
         * @returns {Promise<Object>} - Result
         */
        create: async function(name) {
            const response = await fetch(`${API.baseUrl}/projects/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });
            if (!response.ok) throw new Error(`Failed to create project: ${response.statusText}`);
            return response.json();
        },

        /**
         * Delete a project
         * @param {string} name - Project name
         * @returns {Promise<Object>} - Result
         */
        delete: async function(name) {
            const response = await fetch(`${API.baseUrl}/projects/${encodeURIComponent(name)}`, {
                method: 'DELETE'
            });
            if (!response.ok) throw new Error(`Failed to delete project: ${response.statusText}`);
            return response.json();
        },

        /**
         * Rename a project
         * @param {string} name - Current project name
         * @param {string} newName - New project name
         * @returns {Promise<Object>} - Result
         */
        rename: async function(name, newName) {
            const response = await fetch(`${API.baseUrl}/projects/${encodeURIComponent(name)}/rename`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ new_name: newName })
            });
            if (!response.ok) throw new Error(`Failed to rename project: ${response.statusText}`);
            return response.json();
        }
    },

    /**
     * Dataset Editor endpoints
     */
    datasetEditor: {
        /**
         * Get folder tree structure
         * @param {string} projectName - Project name
         * @returns {Promise<Object>} - Folder tree
         */
        getFolders: async function(projectName) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/folders`);
            if (!response.ok) throw new Error(`Failed to fetch folders: ${response.statusText}`);
            return response.json();
        },

        /**
         * Get project statistics (includes label distribution for tag suggestions)
         * @param {string} projectName - Project name
         * @returns {Promise<Object>} - Stats summary
         */
        getStats: async function(projectName) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/stats`);
            if (!response.ok) throw new Error(`Failed to fetch stats: ${response.statusText}`);
            return response.json();
        },

        /**
         * Refresh/recompute and save project statistics
         * @param {string} projectName - Project name
         * @returns {Promise<Object>} - Result status
         */
        refreshStats: async function(projectName) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/stats/refresh`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error(`Failed to refresh stats: ${response.statusText}`);
            return response.json();
        },

        /**
         * Set dataset type override
         * @param {string} projectName - Project name
         * @param {string} type - Dataset type (multi_label, folder_classification, annotation)
         * @returns {Promise<Object>} - Result status
         */
        setDatasetType: async function(projectName, type) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/dataset-type`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type })
            });
            if (!response.ok) throw new Error(`Failed to set dataset type: ${response.statusText}`);
            return response.json();
        },

        /**
         * Get dataset type override
         * @param {string} projectName - Project name
         * @returns {Promise<Object>} - {type: string}
         */
        getDatasetType: async function(projectName) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/dataset-type`);
            if (!response.ok) throw new Error(`Failed to get dataset type: ${response.statusText}`);
            return response.json();
        },

        /**
         * Set project config value (saves to project.db)
         * @param {string} projectName - Project name
         * @param {string} key - Config key (e.g., 'training.input_size')
         * @param {any} value - Config value
         * @returns {Promise<Object>} - Result status
         */
        setProjectConfigValue: async function(projectName, key, value) {
            const response = await fetch(`${API.baseUrl}/projects/${encodeURIComponent(projectName)}/config/set`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key, value })
            });
            if (!response.ok) throw new Error(`Failed to set project config: ${response.statusText}`);
            return response.json();
        },

        /**
         * Get paginated images
         * @param {string} projectName - Project name
         * @param {Object} params - Query params (page, page_size, folder, search, category)
         * @returns {Promise<Object>} - Paginated items
         */
        getItems: async function(projectName, params = {}) {
            const query = new URLSearchParams();
            if (params.page) query.set('page', params.page);
            if (params.page_size) query.set('page_size', params.page_size);
            if (params.folder) query.set('folder', params.folder);
            if (params.search) query.set('search', params.search);
            if (params.category) query.set('category', params.category);
            const qs = query.toString();
            const url = `${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/items${qs ? '?' + qs : ''}`;
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Failed to fetch items: ${response.statusText}`);
            return response.json();
        },

        /**
         * Get image thumbnail/full URL
         * @param {string} projectName - Project name
         * @param {string} imagePath - Image path (URL encoded)
         * @param {number} size - Optional crop size
         * @returns {string} - Image URL
         */
        getImageUrl: function(projectName, imagePath, size = null) {
            let url = `${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/image/${encodeURIComponent(imagePath)}`;
            if (size) url += `?size=${size}`;
            return url;
        },

        /**
         * Update image annotation
         * @param {string} projectName - Project name
         * @param {string} imagePath - Image path
         * @param {string} content - New annotation text
         * @returns {Promise<Object>} - Updated item
         */
        updateAnnotation: async function(projectName, imagePath, content) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/items/${encodeURIComponent(imagePath)}/annotation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content })
            });
            if (!response.ok) throw new Error(`Failed to update annotation: ${response.statusText}`);
            return response.json();
        },

        /**
         * Update image crop bounds
         * @param {string} projectName - Project name
         * @param {string} imagePath - Image path
         * @param {Object} bounds - { x, y, zoom }
         * @returns {Promise<Object>} - Updated item
         */
        updateBounds: async function(projectName, imagePath, bounds) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/items/${encodeURIComponent(imagePath)}/bounds`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(bounds)
            });
            if (!response.ok) throw new Error(`Failed to update bounds: ${response.statusText}`);
            return response.json();
        },

        /**
         * Delete images
         * @param {string} projectName - Project name
         * @param {string[]} items - Image paths to delete
         * @returns {Promise<Object>} - Delete result
         */
        deleteItems: async function(projectName, items) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/items/delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ items })
            });
            if (!response.ok) throw new Error(`Failed to delete items: ${response.statusText}`);
            return response.json();
        },

        /**
         * Refresh/discover images
         * @param {string} projectName - Project name
         * @returns {Promise<Object>} - Discovery status
         */
        refresh: async function(projectName) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/refresh`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error(`Failed to refresh: ${response.statusText}`);
            return response.json();
        },

        /**
         * Build dataset
         * @param {string} projectName - Project name
         * @param {number} size - Crop size
         * @returns {Promise<Object>} - Build status
         */
        build: async function(projectName, size = 224) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/build?size=${size}`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error(`Failed to build dataset: ${response.statusText}`);
            return response.json();
        },

        /**
         * Get build status
         * @param {string} projectName - Project name
         * @returns {Promise<Object>} - Build status
         */
        getBuildStatus: async function(projectName) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/build/status`);
            if (!response.ok) throw new Error(`Failed to get build status: ${response.statusText}`);
            return response.json();
        },

        /**
         * Bulk tag operations
         * @param {string} projectName - Project name
         * @param {Object} params - { add: [], remove: [], folder, search }
         * @returns {Promise<Object>} - Bulk result
         */
        bulkTags: async function(projectName, params) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/bulk/tags`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            if (!response.ok) throw new Error(`Failed to bulk update tags: ${response.statusText}`);
            return response.json();
        },

        /**
         * Create a new folder
         * @param {string} projectName - Project name
         * @param {string} path - Relative path within data_source
         * @returns {Promise<Object>} - Operation result
         */
        createFolder: async function(projectName, path) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/folders`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            });
            if (!response.ok) throw new Error(`Failed to create folder: ${response.statusText}`);
            return response.json();
        },

        /**
         * Rename a folder
         * @param {string} projectName - Project name
         * @param {string} oldPath - Current relative path
         * @param {string} newName - New folder name (not path)
         * @returns {Promise<Object>} - Operation result
         */
        renameFolder: async function(projectName, oldPath, newName) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/folders/rename`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ old_path: oldPath, new_name: newName })
            });
            if (!response.ok) throw new Error(`Failed to rename folder: ${response.statusText}`);
            return response.json();
        },

        /**
         * Delete a folder
         * @param {string} projectName - Project name
         * @param {string} path - Relative path
         * @param {boolean} recursive - Delete non-empty folders
         * @returns {Promise<Object>} - Operation result
         */
        deleteFolder: async function(projectName, path, recursive = false) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/folders/delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path, recursive })
            });
            if (!response.ok) throw new Error(`Failed to delete folder: ${response.statusText}`);
            return response.json();
        },

        /**
         * Upload images
         * @param {string} projectName - Project name
         * @param {FileList|File[]} files - Files to upload
         * @param {string} folder - Target folder (relative path within data_source)
         * @returns {Promise<Object>} - Upload result
         */
        uploadImages: async function(projectName, files, folder = '') {
            const formData = new FormData();
            for (const file of files) {
                formData.append('files', file);
            }
            
            let url = `${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/upload`;
            if (folder) {
                url += `?folder=${encodeURIComponent(folder)}`;
            }
            
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });
            if (!response.ok) throw new Error(`Failed to upload images: ${response.statusText}`);
            return response.json();
        },

        /**
         * Scan for duplicate images
         * @param {string} projectName - Project name
         * @returns {Promise<Object>} - Duplicate scan results with groups
         */
        scanDuplicates: async function(projectName) {
            const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(projectName)}/duplicates`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error(`Failed to scan duplicates: ${response.statusText}`);
            return response.json();
        }
    },

    /**
     * System monitoring endpoints
     */
    system: {
        /**
         * Get current system resource usage
         * @returns {Promise<Object>} - { cpu, memory, gpus: [{ index, name, usage, memory }] }
         */
        getStats: async function() {
            const response = await fetch(`${API.baseUrl}/system/stats`);
            if (!response.ok) throw new Error(`Failed to get system stats: ${response.statusText}`);
            return response.json();
        }
    },

    /**
     * Training endpoints
     */
    training: {
        /**
         * Start a new training session
         * @param {string} projectName - Project name
         * @param {string} modelType - Model type (e.g., 'resnet', 'efficientnet')
         * @param {string} modelName - Model name (e.g., 'resnet50')
         * @param {number} [epochs] - Optional epoch count override
         * @param {boolean} [resume] - Resume from last checkpoint if available
         * @returns {Promise<Object>} - Training start result with training_id
         */
        start: async function(projectName, modelType, modelName, epochs = null, resume = false) {
            const body = {
                project_name: projectName,
                model_type: modelType,
                model_name: modelName,
                resume: resume
            };
            if (epochs) body.epochs = epochs;
            
            const response = await fetch(`${API.baseUrl}/training/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            if (!response.ok) throw new Error(`Failed to start training: ${response.statusText}`);
            return response.json();
        },

        /**
         * Stop a running training session
         * @param {string} trainingId - Training ID to stop
         * @returns {Promise<Object>} - Stop result
         */
        stop: async function(trainingId) {
            const response = await fetch(`${API.baseUrl}/training/stop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ training_id: trainingId })
            });
            if (!response.ok) throw new Error(`Failed to stop training: ${response.statusText}`);
            return response.json();
        },

        /**
         * Get training status and updates (incremental)
         * @param {string} [trainingId] - Optional training ID, omit for list of active trainings
         * @returns {Promise<Object>} - Training status with incremental updates
         */
        getStatus: async function(trainingId = null) {
            const url = trainingId 
                ? `${API.baseUrl}/training/status?training_id=${encodeURIComponent(trainingId)}`
                : `${API.baseUrl}/training/status`;
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Failed to get training status: ${response.statusText}`);
            return response.json();
        },

        /**
         * Get full training history (all events)
         * @param {string} [trainingId] - Optional training ID
         * @returns {Promise<Object>} - Full training history with all events
         */
        getHistory: async function(trainingId = null) {
            const url = trainingId
                ? `${API.baseUrl}/training/status/all?training_id=${encodeURIComponent(trainingId)}`
                : `${API.baseUrl}/training/status/all`;
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Failed to get training history: ${response.statusText}`);
            return response.json();
        }
    },

    /**
     * Heatmap endpoints
     */
    heatmap: {
        /**
         * Generate heatmap for an image
         * @param {string} projectName - Project name
         * @param {string|null} imagePath - Optional image path (uses random if not provided)
         * @param {number|null} classIndex - Optional target class index
         * @param {number} alpha - Heatmap overlay alpha (0-1)
         * @returns {Promise<Blob>} - PNG image blob
         */
        generate: async function(projectName, imagePath = null, classIndex = null, alpha = 0.5) {
            const params = new URLSearchParams();
            if (imagePath) params.append('image_path', imagePath);
            if (classIndex !== null) params.append('class_index', classIndex);
            params.append('alpha', alpha);
            
            const url = `${API.baseUrl}/projects/${encodeURIComponent(projectName)}/heatmap?${params}`;
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Failed to generate heatmap: ${response.statusText}`);
            return response.blob();
        },

        /**
         * Evaluate image with heatmap and predictions
         * @param {string} projectName - Project name
         * @param {string|null} imagePath - Optional image path (uses random if not provided)
         * @param {string|null} checkpointPath - Optional specific checkpoint path
         * @param {boolean} useLiveModel - If true, skip cache and load fresh model
         * @param {boolean} multiLabel - If true, show all class activations combined
         * @returns {Promise<Object>} - Evaluation results with heatmap (base64) and predictions
         */
        evaluate: async function(projectName, imagePath = null, checkpointPath = null, useLiveModel = false, multiLabel = true) {
            const params = new URLSearchParams();
            if (imagePath) params.append('image_path', imagePath);
            if (checkpointPath) params.append('checkpoint_path', checkpointPath);
            if (useLiveModel) params.append('use_live_model', 'true');
            if (multiLabel) params.append('multi_label', 'true');
            
            const url = `${API.baseUrl}/projects/${encodeURIComponent(projectName)}/evaluate?${params}`;
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Failed to evaluate: ${response.statusText}`);
            return response.json();
        },

        /**
         * Upload and evaluate a custom image
         * @param {string} projectName - Project name
         * @param {File} file - Image file to upload
         * @param {boolean} multiLabel - If true, show all class activations combined
         * @returns {Promise<Object>} - Evaluation results
         */
        evaluateUpload: async function(projectName, file, multiLabel = true) {
            const formData = new FormData();
            formData.append('file', file);
            
            const params = new URLSearchParams();
            if (multiLabel) params.append('multi_label', 'true');
            
            const response = await fetch(`${API.baseUrl}/projects/${encodeURIComponent(projectName)}/evaluate/upload?${params}`, {
                method: 'POST',
                body: formData
            });
            if (!response.ok) throw new Error(`Failed to evaluate uploaded image: ${response.statusText}`);
            return response.json();
        }
    }
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = API;
}
