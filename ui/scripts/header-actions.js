/**
 * HootSight - Header Actions Utility
 * Global utility for managing action buttons in the page header
 * 
 * Usage:
 *   // Clear all buttons
 *   HeaderActions.clear();
 *   
 *   // Add buttons
 *   HeaderActions.add([
 *       {
 *           id: 'my-button',
 *           label: 'Click Me',
 *           labelLangKey: 'some.lang.key',  // optional
 *           type: 'primary',                 // primary, secondary, success, danger, warning
 *           icon: 'icon-name.svg',           // optional
 *           disabled: false,                 // optional
 *           onClick: () => doSomething()
 *       },
 *       // ... more buttons
 *   ]);
 *   
 *   // Get a button by ID
 *   const btn = HeaderActions.get('my-button');
 *   btn.setDisabled(true);
 *   
 *   // Remove a specific button
 *   HeaderActions.remove('my-button');
 */

const HeaderActions = {
    _buttons: {},
    _container: null,

    /**
     * Get the header actions container element
     * @returns {HTMLElement|null}
     */
    _getContainer: function() {
        if (!this._container) {
            this._container = Q('#header-actions').get(0);
        }
        return this._container;
    },

    /**
     * Clear all action buttons from the header
     * @returns {HeaderActions}
     */
    clear: function() {
        const container = this._getContainer();
        if (container) {
            Q(container).empty();
        }
        this._buttons = {};
        return this;
    },

    /**
     * Add action buttons to the header
     * @param {Array|Object} buttons - Button config or array of button configs
     * @returns {HeaderActions}
     */
    add: function(buttons) {
        const container = this._getContainer();
        if (!container) {
            console.warn('HeaderActions: #header-actions container not found');
            return this;
        }

        // Normalize to array
        const buttonArray = Array.isArray(buttons) ? buttons : [buttons];

            buttonArray.forEach(config => {
                const btn = this._createButton(config);
                if (btn) {
                    this._buttons[config.id] = btn;
                    Q(container).append(btn.getElement());
                }
            });

        return this;
    },

    /**
     * Create an ActionButton from config
     * @param {Object} config - Button configuration
     * @returns {ActionButton|null}
     */
    _createButton: function(config) {
        if (!config || !config.id) {
            console.warn('HeaderActions: Button config must have an id');
            return null;
        }

            // Allow custom elements (e.g., switches) to live in header actions
            if (config.customElement) {
                const el = config.customElement;
                const item = {
                    getElement: () => el,
                    setDisabled: (state) => {
                        if (typeof el.disabled !== 'undefined') {
                            el.disabled = !!state;
                        }
                        // Add/remove disabled class for visual consistency
                        if (state) {
                            Q(el).addClass('disabled');
                        } else {
                            Q(el).removeClass('disabled');
                        }
                    },
                    setLabel: () => {}
                };
                return item;
            }

        // Map type to class
        const typeClassMap = {
            'primary': 'btn btn-primary',
            'secondary': 'btn btn-secondary',
            'icon': 'btn btn-secondary btn-icon'
        };

        const className = config.className || typeClassMap[config.type] || 'btn btn-secondary';

        const btn = new ActionButton(config.id, {
            label: config.label || '',
            labelLangKey: config.labelLangKey,
            className: className,
            icon: config.icon,
            title: config.title || '',
            titleLangKey: config.titleLangKey,
            disabled: !!config.disabled,
            onClick: config.onClick
        });

        return btn;
    },

    /**
     * Get a button by ID
     * @param {string} id - Button ID
     * @returns {ActionButton|null}
     */
    get: function(id) {
        return this._buttons[id] || null;
    },

    /**
     * Remove a specific button
     * @param {string} id - Button ID
     * @returns {HeaderActions}
     */
    remove: function(id) {
        const btn = this._buttons[id];
        if (btn) {
            const el = btn.getElement();
            if (el && el.parentNode) {
                el.parentNode.removeChild(el);
            }
            delete this._buttons[id];
        }
        return this;
    },

    /**
     * Enable a button
     * @param {string} id - Button ID
     * @returns {HeaderActions}
     */
    enable: function(id) {
        const btn = this.get(id);
        if (btn) {
            btn.setDisabled(false);
        }
        return this;
    },

    /**
     * Disable a button
     * @param {string} id - Button ID
     * @returns {HeaderActions}
     */
    disable: function(id) {
        const btn = this.get(id);
        if (btn) {
            btn.setDisabled(true);
        }
        return this;
    },

    /**
     * Update button label
     * @param {string} id - Button ID
     * @param {string} label - New label
     * @param {string} langKey - Optional lang key for localization
     * @returns {HeaderActions}
     */
    setLabel: function(id, label, langKey) {
        const btn = this.get(id);
        if (btn) {
            btn.setLabel(label, langKey);
        }
        return this;
    },

    /**
     * Check if a button exists
     * @param {string} id - Button ID
     * @returns {boolean}
     */
    has: function(id) {
        return !!this._buttons[id];
    },

    /**
     * Get all button IDs
     * @returns {string[]}
     */
    getIds: function() {
        return Object.keys(this._buttons);
    },

    /**
     * Reset container reference (call when page changes)
     */
    reset: function() {
        this._container = null;
        this._buttons = {};
    }
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HeaderActions;
}
