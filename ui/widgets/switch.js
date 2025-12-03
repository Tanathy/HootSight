/**
 * HootSight - Switch Widget
 * Schema-compatible toggle/boolean component
 * 
 * Schema UI properties used:
 *   - widget: "switch"
 *   - group: string (group name)
 *   - order: number (display order)
 *   - visible: boolean (default true)
 *   - disabled: boolean (default false)
 * 
 * Schema properties used:
 *   - type: "boolean"
 *   - default: boolean (default value)
 *   - description: string (tooltip/description)
 * 
 * Usage:
 *   const toggle = new Switch('pretrained', {
 *       label: 'Use Pretrained Weights',
 *       description: 'Load pretrained ImageNet weights',
 *       default: true
 *   });
 *   container.appendChild(toggle.getElement());
 */

class Switch {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            description: options.description || '',
            default: options.default || false,
            disabled: options.disabled || false,
            visible: options.visible !== false
        };
        
        this._value = this.options.default;
        this._changeCallbacks = [];
        
        this._build();
    }
    
    _build() {
        // Main wrapper
        this.element = Q('<div>', { class: 'widget widget-switch' }).get(0);
        this.element.id = this.id;
        
        if (!this.options.visible) {
            this.element.style.display = 'none';
        }
        
        // Switch row (label + switch)
        this.switchRow = Q('<div>', { class: 'switch-row' }).get(0);
        
        // Label
        if (this.options.label) {
            this.labelEl = Q('<label>', { class: 'widget-label switch-label', text: this.options.label }).get(0);
            this.labelEl.setAttribute('for', this.id + '-input');
            // Add lang key attribute for live translation
            if (this.options.labelLangKey) {
                this.labelEl.setAttribute('data-lang-key', this.options.labelLangKey);
            }
            this.switchRow.appendChild(this.labelEl);
        }
        
        // Switch container
        this.switchContainer = Q('<div>', { class: 'switch-container' }).get(0);
        
        // Hidden checkbox
        this.checkbox = Q('<input>', { type: 'checkbox', class: 'switch-input', id: this.id + '-input' }).get(0);
        this.checkbox.checked = this._value;
        
        // Switch track
        this.track = Q('<div>', { class: 'switch-track' }).get(0);
        
        // Switch thumb
        this.thumb = Q('<div>', { class: 'switch-thumb' }).get(0);
        this.track.appendChild(this.thumb);
        
        // Assemble switch
        this.switchContainer.appendChild(this.checkbox);
        this.switchContainer.appendChild(this.track);
        this.switchRow.appendChild(this.switchContainer);
        this.element.appendChild(this.switchRow);
        
        // Description
        if (this.options.description) {
            this.descEl = Q('<div>', { class: 'widget-description', text: this.options.description }).get(0);
            // Add lang key attribute for live translation
            if (this.options.descriptionLangKey) {
                this.descEl.setAttribute('data-lang-key', this.options.descriptionLangKey);
            }
            this.element.appendChild(this.descEl);
        }
        
        // Update visual state
        this._updateVisualState();
        
        // Event listeners
        this._setupEvents();
        
        // Disabled state
        if (this.options.disabled) {
            this.disable();
        }
    }
    
    _updateVisualState() {
        if (this._value) {
            Q(this.track).addClass('active');
        } else {
            Q(this.track).removeClass('active');
        }
    }
    
    _setupEvents() {
        // Click on track
        Q(this.track).on('click', () => {
            if (!this.options.disabled) {
                this.toggle();
            }
        });
        
        // Click on label
        if (this.labelEl) {
            Q(this.labelEl).on('click', () => {
                if (!this.options.disabled) {
                    this.toggle();
                }
            });
        }
        
        // Checkbox change (for form compatibility)
        Q(this.checkbox).on('change', () => {
            const newValue = this.checkbox.checked;
            if (newValue !== this._value) {
                this._setValue(newValue);
            }
        });
        
        // Keyboard support
        this.track.setAttribute('tabindex', '0');
        Q(this.track).on('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                if (!this.options.disabled) {
                    this.toggle();
                }
            }
        });
    }
    
    _setValue(value) {
        const previousValue = this._value;
        this._value = value;
        this.checkbox.checked = value;
        this._updateVisualState();
        
        // Trigger callbacks
        if (previousValue !== value) {
            this._changeCallbacks.forEach(cb => cb(value, previousValue));
        }
    }
    
    /**
     * Toggle the switch
     * @returns {Switch}
     */
    toggle() {
        this._setValue(!this._value);
        return this;
    }
    
    /**
     * Get current value
     * @returns {boolean}
     */
    get() {
        return this._value;
    }
    
    /**
     * Set value
     * @param {boolean} value
     * @returns {Switch}
     */
    set(value) {
        this._setValue(!!value);
        return this;
    }
    
    /**
     * Register change callback
     * @param {Function} callback - Called with (newValue, previousValue)
     * @returns {Switch}
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
     * @returns {Switch}
     */
    enable() {
        this.options.disabled = false;
        Q(this.element).removeClass('disabled');
        this.checkbox.disabled = false;
        return this;
    }
    
    /**
     * Disable widget
     * @returns {Switch}
     */
    disable() {
        this.options.disabled = true;
        Q(this.element).addClass('disabled');
        this.checkbox.disabled = true;
        return this;
    }
    
    /**
     * Show widget
     * @returns {Switch}
     */
    show() {
        this.options.visible = true;
        this.element.style.display = '';
        return this;
    }
    
    /**
     * Hide widget
     * @returns {Switch}
     */
    hide() {
        this.options.visible = false;
        this.element.style.display = 'none';
        return this;
    }
    
    /**
     * Create from schema definition
     * @param {string} id - Widget ID
     * @param {Object} schema - Schema property definition
     * @param {Function} lang - Localization function (optional)
     * @returns {Switch}
     */
    static fromSchema(id, schema, lang = null) {
        const translate = lang || (key => key);
        
        return new Switch(id, {
            label: translate(schema.description || ''),
            description: schema.ui?.description ? translate(schema.ui.description) : '',
            default: schema.default || false,
            disabled: schema.ui?.disabled || false,
            visible: schema.ui?.visible !== false
        });
    }
}

// Register in WidgetRegistry if available
if (typeof WidgetRegistry !== 'undefined') {
    WidgetRegistry.register('switch', Switch);
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Switch;
}
