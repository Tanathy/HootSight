/**
 * HootSight - Localization Module
 * Loads language data from API and provides lang() function for text retrieval
 * Supports live language switching without page refresh
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
     * Currently active language code
     */
    _activeLanguage: 'en',

    /**
     * Available languages list
     */
    _availableLanguages: [],

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
                this._activeLanguage = data.active || 'en';
                this._availableLanguages = data.languages || [];
                console.log('Localization loaded:', Object.keys(this._strings).length, 'keys, active:', this._activeLanguage);
                return true;
            }
            return false;
        } catch (err) {
            console.error('Failed to load localization:', err);
            return false;
        }
    },

    /**
     * Switch to a different language
     * @param {string} langCode - Language code (e.g., 'en', 'hu')
     * @returns {Promise<boolean>} - Success status
     */
    switchLanguage: async function(langCode) {
        if (langCode === this._activeLanguage) {
            return true; // Already on this language
        }

        try {
            const response = await fetch('/localization/switch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lang_code: langCode })
            });

            if (!response.ok) {
                console.error('Failed to switch language:', response.statusText);
                return false;
            }

            const data = await response.json();
            if (data.localization) {
                this._strings = this._flatten(data.localization);
                this._activeLanguage = data.active || langCode;
                
                // Refresh all translatable elements on the page
                this.refresh();
                
                console.log('Language switched to:', this._activeLanguage);
                return true;
            }
            return false;
        } catch (err) {
            console.error('Failed to switch language:', err);
            return false;
        }
    },

    /**
     * Refresh all elements with data-lang-key attribute
     * Updates their text content with the new language strings
     */
    refresh: function() {
        // Find all elements with data-lang-key
        const elements = document.querySelectorAll('[data-lang-key]');
        
        elements.forEach(el => {
            const key = el.getAttribute('data-lang-key');
            const params = el.getAttribute('data-lang-params');
            
            let parsedParams = {};
            if (params) {
                try {
                    parsedParams = JSON.parse(params);
                } catch (e) {
                    // Ignore parse errors
                }
            }
            
            const text = this.get(key, parsedParams);
            
            // Check if it's an input placeholder
            if (el.hasAttribute('data-lang-placeholder')) {
                el.placeholder = text;
            } else if (el.hasAttribute('data-lang-title')) {
                el.title = text;
            } else {
                el.textContent = text;
            }
        });

        // Dispatch custom event for components that need special refresh handling
        document.dispatchEvent(new CustomEvent('lang:refresh', { 
            detail: { language: this._activeLanguage } 
        }));
    },

    /**
     * Get currently active language code
     * @returns {string}
     */
    getActiveLanguage: function() {
        return this._activeLanguage;
    },

    /**
     * Get list of available languages
     * @returns {Array<{code: string, name: string}>}
     */
    getAvailableLanguages: function() {
        return this._availableLanguages;
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
