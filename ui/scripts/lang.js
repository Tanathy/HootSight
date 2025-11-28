/**
 * HootSight - Localization Module
 * Loads language data from API and provides lang() function for text retrieval
 */

const Lang = {
    /**
     * Flattened localization strings
     * Keys are dot-notation paths like "ui.dataset.title"
     */
    _strings: {},

    /**
     * Whether localization data has been loaded
     */
    _loaded: false,

    /**
     * Load localization data from API
     * @returns {Promise<boolean>} - Success status
     */
    load: async function() {
        try {
            const response = await fetch('/localization');
            if (!response.ok) {
                console.error('Failed to load localization:', response.statusText);
                return false;
            }

            const data = await response.json();
            if (data.localization) {
                this._strings = this._flatten(data.localization);
                this._loaded = true;
                console.log('Localization loaded:', Object.keys(this._strings).length, 'keys');
                return true;
            }
            return false;
        } catch (err) {
            console.error('Failed to load localization:', err);
            return false;
        }
    },

    /**
     * Flatten nested object to dot-notation keys
     * @param {Object} obj - Nested object
     * @param {string} prefix - Current key prefix
     * @returns {Object} - Flattened object
     */
    _flatten: function(obj, prefix = '') {
        const result = {};

        for (const key in obj) {
            if (!obj.hasOwnProperty(key)) continue;

            const fullKey = prefix ? `${prefix}.${key}` : key;
            const value = obj[key];

            if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
                // Recurse into nested objects
                Object.assign(result, this._flatten(value, fullKey));
            } else {
                // Store leaf value
                result[fullKey] = value;
            }
        }

        return result;
    },

    /**
     * Get localized string with parameter interpolation
     * @param {string} key - Dot-notation key like "ui.dataset.title"
     * @param {Object} params - Parameters for interpolation {name: "value"}
     * @returns {string} - Localized string or key if not found
     */
    get: function(key, params = {}) {
        let text = this._strings[key];

        // Return key if not found
        if (text === undefined || text === null) {
            return key;
        }

        // Interpolate parameters: {param} -> value
        if (params && typeof params === 'object') {
            for (const param in params) {
                if (params.hasOwnProperty(param)) {
                    const placeholder = new RegExp(`\\{${param}\\}`, 'g');
                    text = text.replace(placeholder, params[param]);
                }
            }
        }

        return text;
    },

    /**
     * Check if a key exists
     * @param {string} key - Dot-notation key
     * @returns {boolean}
     */
    has: function(key) {
        return this._strings.hasOwnProperty(key);
    },

    /**
     * Check if localization is loaded
     * @returns {boolean}
     */
    isLoaded: function() {
        return this._loaded;
    }
};

/**
 * Shorthand function for Lang.get()
 * @param {string} key - Dot-notation key
 * @param {Object} params - Parameters for interpolation
 * @returns {string} - Localized string or key if not found
 */
function lang(key, params = {}) {
    return Lang.get(key, params);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Lang, lang };
}
