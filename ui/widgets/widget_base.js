/**
 * HootSight - Widget Base
 * Base class and registry for all custom widgets
 * 
 * Supported widget types (from schema):
 * - dropdown: Select dropdown
 * - switch: Toggle boolean
 * - text: Single line text input
 * - textarea: Multi-line text input
 * - slider: Range slider with number input
 * - number: Number input field
 * - auto_or_number: "auto" or number hybrid
 * - array: Tag/list input
 * - path: File/directory path input
 * - readonly: Non-editable display
 */

const WidgetRegistry = {
    _widgets: {},
    
    /**
     * Register a widget class
     * @param {string} type - Widget type name
     * @param {Function} WidgetClass - Widget constructor
     */
    register(type, WidgetClass) {
        this._widgets[type] = WidgetClass;
    },
    
    /**
     * Create a widget instance
     * @param {string} type - Widget type
     * @param {string} id - Widget identifier
     * @param {Object} options - Widget options from schema
     * @returns {Object|null} Widget instance or null
     */
    create(type, id, options = {}) {
        const WidgetClass = this._widgets[type];
        if (!WidgetClass) {
            console.warn('Unknown widget type:', type);
            return null;
        }
        return new WidgetClass(id, options);
    },
    
    /**
     * Get all registered widget types
     * @returns {string[]}
     */
    getTypes() {
        return Object.keys(this._widgets);
    }
};

/**
 * Base Widget Class
 * All widgets extend this
 */
class BaseWidget {
    constructor(id, options = {}) {
        this.id = id;
        this.options = options;
        this.element = null;
        this._value = options.default !== undefined ? options.default : null;
        this._changeCallbacks = [];
    }
    
    /**
     * Get widget value
     * @returns {*}
     */
    get() {
        return this._value;
    }
    
    /**
     * Set widget value
     * @param {*} value
     */
    set(value) {
        this._value = value;
        this._triggerChange();
    }
    
    /**
     * Get the DOM element
     * @returns {HTMLElement}
     */
    getElement() {
        return this.element;
    }
    
    /**
     * Register change callback
     * @param {Function} callback
     */
    onChange(callback) {
        this._changeCallbacks.push(callback);
    }
    
    /**
     * Trigger change event
     */
    _triggerChange() {
        this._changeCallbacks.forEach(cb => cb(this._value));
    }
    
    /**
     * Enable widget
     */
    enable() {
        if (this.element) {
            Q(this.element).removeClass('disabled');
        }
    }
    
    /**
     * Disable widget
     */
    disable() {
        if (this.element) {
            Q(this.element).addClass('disabled');
        }
    }
    
    /**
     * Show widget
     */
    show() {
        if (this.element) {
            this.element.style.display = '';
        }
    }
    
    /**
     * Hide widget
     */
    hide() {
        if (this.element) {
            this.element.style.display = 'none';
        }
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { WidgetRegistry, BaseWidget };
}
