/**
 * HootSight - Text Widget
 * Schema-compatible single-line text input
 * 
 * Schema UI properties used:
 *   - widget: "text"
 *   - group: string (group name)
 *   - order: number (display order)
 *   - visible: boolean (default true)
 *   - disabled: boolean (default false)
 * 
 * Schema properties used:
 *   - type: "string"
 *   - default: string (default value)
 *   - description: string (tooltip/description)
 *   - minLength: number
 *   - maxLength: number
 *   - pattern: string (regex pattern)
 * 
 * Usage:
 *   const input = new TextInput('api_host', {
 *       label: 'API Host',
 *       description: 'Server hostname or IP address',
 *       default: '127.0.0.1',
 *       placeholder: 'Enter hostname...'
 *   });
 *   container.appendChild(input.getElement());
 */

class TextInput {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            description: options.description || '',
            default: options.default || '',
            placeholder: options.placeholder || '',
            disabled: options.disabled || false,
            visible: options.visible !== false,
            minLength: options.minLength || null,
            maxLength: options.maxLength || null,
            pattern: options.pattern || null
        };
        
        this._value = this.options.default;
        this._changeCallbacks = [];
        
        this._build();
    }
    
    _build() {
        // Main wrapper
        this.element = Q('<div>', { class: 'widget widget-text' }).get(0);
        this.element.id = this.id;
        
        if (!this.options.visible) {
            this.element.style.display = 'none';
        }
        
        // Label
        if (this.options.label) {
            this.labelEl = Q('<label>', { class: 'widget-label', text: this.options.label }).get(0);
            this.labelEl.setAttribute('for', this.id + '-input');
            this.element.appendChild(this.labelEl);
        }
        
        // Input
        this.input = Q('<input>', { type: 'text', class: 'text-input', id: this.id + '-input' }).get(0);
        this.input.value = this._value;
        
        if (this.options.placeholder) {
            this.input.placeholder = this.options.placeholder;
        }
        if (this.options.minLength !== null) {
            this.input.minLength = this.options.minLength;
        }
        if (this.options.maxLength !== null) {
            this.input.maxLength = this.options.maxLength;
        }
        if (this.options.pattern) {
            this.input.pattern = this.options.pattern;
        }
        
        this.element.appendChild(this.input);
        
        // Description
        if (this.options.description) {
            this.descEl = Q('<div>', { class: 'widget-description', text: this.options.description }).get(0);
            this.element.appendChild(this.descEl);
        }
        
        // Event listeners
        this._setupEvents();
        
        // Disabled state
        if (this.options.disabled) {
            this.disable();
        }
    }
    
    _setupEvents() {
        Q(this.input).on('input', () => {
            const previousValue = this._value;
            this._value = this.input.value;
            
            if (previousValue !== this._value) {
                this._changeCallbacks.forEach(cb => cb(this._value, previousValue));
            }
        });
    }
    
    /**
     * Get current value
     * @returns {string}
     */
    get() {
        return this._value;
    }
    
    /**
     * Set value
     * @param {string} value
     * @returns {TextInput}
     */
    set(value) {
        const previousValue = this._value;
        this._value = String(value);
        this.input.value = this._value;
        
        if (previousValue !== this._value) {
            this._changeCallbacks.forEach(cb => cb(this._value, previousValue));
        }
        return this;
    }
    
    /**
     * Register change callback
     * @param {Function} callback - Called with (newValue, previousValue)
     * @returns {TextInput}
     */
    onChange(callback) {
        this._changeCallbacks.push(callback);
        return this;
    }
    
    /**
     * Get DOM element
     * @returns {HTMLElement}
     */
    getElement() {
        return this.element;
    }
    
    /**
     * Enable widget
     * @returns {TextInput}
     */
    enable() {
        this.options.disabled = false;
        Q(this.element).removeClass('disabled');
        this.input.disabled = false;
        return this;
    }
    
    /**
     * Disable widget
     * @returns {TextInput}
     */
    disable() {
        this.options.disabled = true;
        Q(this.element).addClass('disabled');
        this.input.disabled = true;
        return this;
    }
    
    /**
     * Show widget
     * @returns {TextInput}
     */
    show() {
        this.options.visible = true;
        this.element.style.display = '';
        return this;
    }
    
    /**
     * Hide widget
     * @returns {TextInput}
     */
    hide() {
        this.options.visible = false;
        this.element.style.display = 'none';
        return this;
    }
    
    /**
     * Focus input
     * @returns {TextInput}
     */
    focus() {
        this.input.focus();
        return this;
    }
    
    /**
     * Create from schema definition
     * @param {string} id - Widget ID
     * @param {Object} schema - Schema property definition
     * @param {Function} lang - Localization function (optional)
     * @returns {TextInput}
     */
    static fromSchema(id, schema, lang = null) {
        const translate = lang || (key => key);
        
        return new TextInput(id, {
            label: translate(schema.description || ''),
            description: schema.ui?.description ? translate(schema.ui.description) : '',
            default: schema.default || '',
            placeholder: schema.ui?.placeholder ? translate(schema.ui.placeholder) : '',
            minLength: schema.minLength || null,
            maxLength: schema.maxLength || null,
            pattern: schema.pattern || null,
            disabled: schema.ui?.disabled || false,
            visible: schema.ui?.visible !== false
        });
    }
}

// Register in WidgetRegistry if available
if (typeof WidgetRegistry !== 'undefined') {
    WidgetRegistry.register('text', TextInput);
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TextInput;
}
