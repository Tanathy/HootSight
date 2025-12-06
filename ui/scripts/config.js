/**
 * HootSight - Configuration Module
 * Loads and manages application configuration with deep merge support
 */

const Config = {
    /**
     * Global configuration data
     */
    _data: {},

    /**
     * Configuration schema
     */
    _schema: {},

    /**
     * Currently loaded project name
     */
    _activeProject: null,

    /**
     * Whether config has been loaded
     */
    _loaded: false,

    /**
     * Whether schema has been loaded
     */
    _schemaLoaded: false,

    /**
     * Load global configuration from API
     * @returns {Promise<boolean>} - Success status
     */
    load: async function() {
        try {
            const response = await fetch('/config');
            if (!response.ok) {
                console.error('Failed to load config:', response.statusText);
                return false;
            }

            const data = await response.json();
            if (data.config) {
                this._data = data.config;
                this._loaded = true;
                console.log('Config loaded');
                return true;
            }
            return false;
        } catch (err) {
            console.error('Failed to load config:', err);
            return false;
        }
    },

    /**
     * Load configuration schema from API
     * @returns {Promise<boolean>} - Success status
     */
    loadSchema: async function() {
        try {
            const response = await fetch('/config/schema');
            if (!response.ok) {
                console.error('Failed to load config schema:', response.statusText);
                return false;
            }

            const data = await response.json();
            if (data.schema) {
                this._schema = data.schema;
                this._schemaLoaded = true;
                console.log('Config schema loaded');
                return true;
            }
            return false;
        } catch (err) {
            console.error('Failed to load config schema:', err);
            return false;
        }
    },

    /**
     * Get the full schema object
     * @returns {Object}
     */
    getSchema: function() {
        return this._schema;
    },

    /**
     * Get schema for a specific config path
     * @param {string} path - Dot-notation path like "training.batch_size"
     * @returns {Object|undefined} - Schema definition or undefined
     */
    getSchemaFor: function(path) {
        if (!path || !this._schema.properties) return undefined;

        const parts = path.split('.');
        let current = this._schema;

        for (const part of parts) {
            if (!current.properties || !(part in current.properties)) {
                return undefined;
            }
            current = current.properties[part];
        }

        return current;
    },

    /**
     * Check if schema is loaded
     * @returns {boolean}
     */
    isSchemaLoaded: function() {
        return this._schemaLoaded;
    },

    /**
     * Load and merge project-specific configuration
     * Always resets to global config first to avoid cross-project contamination
     * @param {string} projectName - Project name
     * @returns {Promise<boolean>} - Success status
     */
    loadProject: async function(projectName) {
        try {
            // Always reset to global config first to avoid cross-project config contamination
            await this.load();
            
            const response = await fetch(`/projects/${encodeURIComponent(projectName)}/config`);
            if (!response.ok) {
                // No project config is fine - just use global
                console.log(`No project config for ${projectName}, using global`);
                this._activeProject = projectName;
                this._updateHeaderProjectInfo(projectName);
                return true;
            }

            const data = await response.json();
            if (data.config && typeof data.config === 'object') {
                // Deep merge project config onto fresh global config
                this._data = this._deepMerge(this._data, data.config);
                this._activeProject = projectName;
                this._updateHeaderProjectInfo(projectName);
                console.log(`Project config loaded and merged: ${projectName}`);
                return true;
            }
            
            this._activeProject = projectName;
            this._updateHeaderProjectInfo(projectName);
            return true;
        } catch (err) {
            console.error(`Failed to load project config for ${projectName}:`, err);
            this._activeProject = projectName;
            this._updateHeaderProjectInfo(projectName);
            return false;
        }
    },

    /**
     * Deep merge two objects
     * @param {Object} target - Target object
     * @param {Object} source - Source object to merge in
     * @returns {Object} - Merged object
     */
    _deepMerge: function(target, source) {
        const result = { ...target };

        for (const key in source) {
            if (!source.hasOwnProperty(key)) continue;

            const sourceVal = source[key];
            const targetVal = result[key];

            if (sourceVal !== null && typeof sourceVal === 'object' && !Array.isArray(sourceVal)) {
                if (targetVal !== null && typeof targetVal === 'object' && !Array.isArray(targetVal)) {
                    // Both are objects - recurse
                    result[key] = this._deepMerge(targetVal, sourceVal);
                } else {
                    // Target is not an object - replace
                    result[key] = this._deepMerge({}, sourceVal);
                }
            } else {
                // Primitive or array - replace
                result[key] = sourceVal;
            }
        }

        return result;
    },

    /**
     * Get a config value by dot-notation path
     * @param {string} path - Dot-notation path like "training.batch_size"
     * @param {*} defaultValue - Default value if path not found
     * @returns {*} - Config value or default
     */
    get: function(path, defaultValue = undefined) {
        if (!path) return this._data;

        const parts = path.split('.');
        let current = this._data;

        for (const part of parts) {
            if (current === null || typeof current !== 'object') {
                return defaultValue;
            }
            if (!(part in current)) {
                return defaultValue;
            }
            current = current[part];
        }

        return current;
    },

    /**
     * Set a config value by dot-notation path (in-memory only)
     * @param {string} path - Dot-notation path
     * @param {*} value - Value to set
     */
    set: function(path, value) {
        if (!path) return;

        const parts = path.split('.');
        let current = this._data;

        for (let i = 0; i < parts.length - 1; i++) {
            const part = parts[i];
            if (!(part in current) || typeof current[part] !== 'object') {
                current[part] = {};
            }
            current = current[part];
        }

        current[parts[parts.length - 1]] = value;
    },

    /**
     * Get the full config object
     * @returns {Object}
     */
    getAll: function() {
        return this._data;
    },

    /**
     * Get currently active project name
     * @returns {string|null}
     */
    getActiveProject: function() {
        return this._activeProject;
    },

    /**
     * Clear currently active project (without reloading config)
     */
    clearActiveProject: function() {
        this._activeProject = null;
        this._updateHeaderProjectInfo(null);
    },

    /**
     * Check if config is loaded
     * @returns {boolean}
     */
    isLoaded: function() {
        return this._loaded;
    },

    /**
     * Reset to global config (unload project config)
     * Reloads global config from API
     * @returns {Promise<boolean>}
     */
    resetToGlobal: async function() {
        this._activeProject = null;
        this._updateHeaderProjectInfo(null);
        return await this.load();
    },

    /**
     * Update header-info with current project name
     * @param {string|null} projectName
     */
    _updateHeaderProjectInfo: function(projectName) {
        const headerInfo = Q('.header-info').get(0);
        if (!headerInfo) return;

        // Remove existing project info
        const existingInfo = Q(headerInfo).find('.header-project-info').get(0);
        if (existingInfo) {
            existingInfo.remove();
        }

        // Add new project info if we have a project
        if (projectName) {
            const projectInfo = Q('<div>', { class: 'header-project-info' }).get(0);
            
            const label = Q('<span>', {
                class: 'project-label',
                text: typeof lang === 'function' ? lang('common.project_label') : 'Project:'
            }).get(0);
            
            const name = Q('<span>', {
                class: 'project-name',
                text: projectName
            }).get(0);
            
            Q(projectInfo).append(label);
            Q(projectInfo).append(name);
            
            // Insert at beginning of header-info
            headerInfo.insertBefore(projectInfo, headerInfo.firstChild);
        }
    }
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Config;
}
