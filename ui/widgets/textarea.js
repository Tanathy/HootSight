/**
 * HootSight - Textarea Widget
 * Schema-compatible multi-line text input
 * 
 * Schema UI properties used:
 *   - widget: "textarea"
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
 * 
 * Usage:
 *   const textarea = new Textarea('notes', {
 *       label: 'Training Notes',
 *       description: 'Additional notes for this training run',
 *       rows: 4,
 *       placeholder: 'Enter notes...'
 *   });
 *   container.appendChild(textarea.getElement());
 */

class Textarea {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            description: options.description || '',
            default: options.default || '',
            placeholder: options.placeholder || '',
            disabled: options.disabled || false,
            visible: options.visible !== false,
            rows: options.rows || 3,
            minLength: options.minLength || null,
            maxLength: options.maxLength || null,
            resize: options.resize !== false // allow resize by default
        };
        
        this._value = this.options.default;
        this._changeCallbacks = [];
        
        this._build();
    }
    
    _build() {
        // Main wrapper
        this.element = Q('<div>', { class: 'widget widget-textarea' }).get(0);
        this.element.id = this.id;
        
        if (!this.options.visible) {
            this.element.style.display = 'none';
        }
        
        // Label
        if (this.options.label) {
            this.labelEl = Q('<label>', { class: 'widget-label', text: this.options.label }).get(0);
            this.labelEl.setAttribute('for', this.id + '-input');
            // Add lang key attribute for live translation
            if (this.options.labelLangKey) {
                this.labelEl.setAttribute('data-lang-key', this.options.labelLangKey);
            }
            Q(this.element).append(this.labelEl);
        }
        
        // Textarea
        this.textarea = Q('<textarea>', { class: 'textarea-input', id: this.id + '-input' }).get(0);
        this.textarea.value = this._value;
        this.textarea.rows = this.options.rows;
        
        if (this.options.placeholder) {
            this.textarea.placeholder = this.options.placeholder;
            // Add lang key for placeholder
            if (this.options.placeholderLangKey) {
                this.textarea.setAttribute('data-lang-key', this.options.placeholderLangKey);
                this.textarea.setAttribute('data-lang-placeholder', 'true');
            }
        }
        if (this.options.minLength !== null) {
            this.textarea.minLength = this.options.minLength;
        }
        if (this.options.maxLength !== null) {
            this.textarea.maxLength = this.options.maxLength;
        }
        if (!this.options.resize) {
            this.textarea.style.resize = 'none';
        }
        
        Q(this.element).append(this.textarea);
        
        // Description
        if (this.options.description) {
            this.descEl = Q('<div>', { class: 'widget-description', text: this.options.description }).get(0);
            // Add lang key attribute for live translation
            if (this.options.descriptionLangKey) {
                this.descEl.setAttribute('data-lang-key', this.options.descriptionLangKey);
            }
            Q(this.element).append(this.descEl);
        }
        
        // Event listeners
        this._setupEvents();
        
        // Disabled state
        if (this.options.disabled) {
            this.disable();
        }
    }
    
    _setupEvents() {
        Q(this.textarea).on('input', () => {
            const previousValue = this._value;
            this._value = this.textarea.value;
            
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
     * @returns {Textarea}
     */
    set(value) {
        const previousValue = this._value;
        this._value = String(value);
        this.textarea.value = this._value;
        
        if (previousValue !== this._value) {
            this._changeCallbacks.forEach(cb => cb(this._value, previousValue));
        }
        return this;
    }
    
    /**
     * Register change callback
     * @param {Function} callback - Called with (newValue, previousValue)
     * @returns {Textarea}
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
     * @returns {Textarea}
     */
    enable() {
        this.options.disabled = false;
        Q(this.element).removeClass('disabled');
        this.textarea.disabled = false;
        return this;
    }
    
    /**
     * Disable widget
     * @returns {Textarea}
     */
    disable() {
        this.options.disabled = true;
        Q(this.element).addClass('disabled');
        this.textarea.disabled = true;
        return this;
    }
    
    /**
     * Show widget
     * @returns {Textarea}
     */
    show() {
        this.options.visible = true;
        this.element.style.display = '';
        return this;
    }
    
    /**
     * Hide widget
     * @returns {Textarea}
     */
    hide() {
        this.options.visible = false;
        this.element.style.display = 'none';
        return this;
    }
    
    /**
     * Focus textarea
     * @returns {Textarea}
     */
    focus() {
        this.textarea.focus();
        return this;
    }
    
    /**
     * Create from schema definition
     * @param {string} id - Widget ID
     * @param {Object} schema - Schema property definition
     * @param {Function} lang - Localization function (optional)
     * @returns {Textarea}
     */
    static fromSchema(id, schema, lang = null) {
        const translate = lang || (key => key);
        
        return new Textarea(id, {
            label: translate(schema.description || ''),
            description: schema.ui?.description ? translate(schema.ui.description) : '',
            default: schema.default || '',
            placeholder: schema.ui?.placeholder ? translate(schema.ui.placeholder) : '',
            rows: schema.ui?.rows || 3,
            minLength: schema.minLength || null,
            maxLength: schema.maxLength || null,
            resize: schema.ui?.resize !== false,
            disabled: schema.ui?.disabled || false,
            visible: schema.ui?.visible !== false
        });
    }
}

// Register in WidgetRegistry if available
if (typeof WidgetRegistry !== 'undefined') {
    WidgetRegistry.register('textarea', Textarea);
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Textarea;
}
