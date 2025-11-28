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
    }
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = API;
}
