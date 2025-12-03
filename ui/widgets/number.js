/**
 * HootSight - Number Widget
 * Schema-compatible number input component using qte.js
 * 
 * Schema UI properties used:
 *   - widget: "number"
 *   - group: string (group name)
 *   - order: number (display order)
 *   - visible: boolean (default true)
 *   - disabled: boolean (default false)
 * 
 * Schema properties used:
 *   - type: "number" | "integer"
 *   - minimum: number (min value)
 *   - maximum: number (max value)
 *   - default: number (default value)
 *   - description: string (tooltip/description)
 * 
 * Usage:
 *   const num = new NumberInput('batch_size', {
 *       label: 'Batch Size',
 *       min: 1,
 *       max: 512,
 *       step: 1,
 *       default: 32
 *   });
 *   container.appendChild(num.getElement());
 */

class NumberInput {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            description: options.description || '',
            min: options.min ?? null,
            max: options.max ?? null,
            step: options.step ?? 1,
            default: options.default ?? 0,
            integer: options.integer ?? false,
            precision: options.precision ?? null, // decimal places for display
            disabled: options.disabled || false,
            visible: options.visible !== false
        };
        
        this._value = this.options.default;
        this._changeCallbacks = [];
        
        this._build();
    }
    
    _build() {
        // Main wrapper
        this.element = Q('<div>', { class: 'widget widget-number' }).get(0);
        this.element.id = this.id;
        
        if (!this.options.visible) {
            Q(this.element).css('display', 'none');
        }
        
        // Label
        if (this.options.label) {
            this.labelEl = Q('<label>', { 
                class: 'widget-label', 
                text: this.options.label,
                for: this.id + '-input'
            }).get(0);
            // Add lang key attribute for live translation
            if (this.options.labelLangKey) {
                this.labelEl.setAttribute('data-lang-key', this.options.labelLangKey);
            }
            this.element.appendChild(this.labelEl);
        }
        
        // Input container (controls wrapper)
        this.inputContainer = Q('<div>', { class: 'number-widget-controls' }).get(0);
        
        // Decrease button
        this.decreaseBtn = Q('<button>', { 
            class: 'number-btn number-decrease',
            type: 'button',
            text: '-'
        }).get(0);
        
        // Input
        this.input = Q('<input>', {
            type: 'number',
            class: 'widget-input number-input',
            id: this.id + '-input',
            value: this._value
        }).get(0);
        
        if (this.options.min !== null) this.input.min = this.options.min;
        if (this.options.max !== null) this.input.max = this.options.max;
        if (this.options.step) this.input.step = this.options.step;
        
        // Increase button
        this.increaseBtn = Q('<button>', { 
            class: 'number-btn number-increase',
            type: 'button',
            text: '+'
        }).get(0);
        
        // Assemble
        this.inputContainer.appendChild(this.decreaseBtn);
        this.inputContainer.appendChild(this.input);
        this.inputContainer.appendChild(this.increaseBtn);
        this.element.appendChild(this.inputContainer);
        
        // Description
        if (this.options.description) {
            this.descEl = Q('<div>', { 
                class: 'widget-description', 
                text: this.options.description 
            }).get(0);
            // Add lang key attribute for live translation
            if (this.options.descriptionLangKey) {
                this.descEl.setAttribute('data-lang-key', this.options.descriptionLangKey);
            }
            this.element.appendChild(this.descEl);
        }
        
        this._setupEvents();
        
        if (this.options.disabled) {
            this.disable();
        }
    }
    
    _setupEvents() {
        // Input change
        Q(this.input).on('change', () => {
            this._setValue(this._parseValue(this.input.value));
        });
        
        Q(this.input).on('input', () => {
            // Live validation feedback could go here
        });
        
        // Decrease button
        Q(this.decreaseBtn).on('click', () => {
            const step = this.options.step || 1;
            this._setValue(this._value - step);
        });
        
        // Increase button
        Q(this.increaseBtn).on('click', () => {
            const step = this.options.step || 1;
            this._setValue(this._value + step);
        });
        
        // Keyboard support
        Q(this.input).on('keydown', (e) => {
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                const step = this.options.step || 1;
                this._setValue(this._value + step);
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                const step = this.options.step || 1;
                this._setValue(this._value - step);
            }
        });
    }
    
    _parseValue(val) {
        const parsed = this.options.integer ? parseInt(val, 10) : parseFloat(val);
        return isNaN(parsed) ? this.options.default : parsed;
    }
    
    _clampValue(val) {
        if (this.options.min !== null && val < this.options.min) val = this.options.min;
        if (this.options.max !== null && val > this.options.max) val = this.options.max;
        return val;
    }
    
    _formatValue(val) {
        const { precision, integer, step } = this.options;
        
        if (integer) {
            return Math.round(val).toString();
        }
        
        // Use explicit precision if provided
        if (precision !== null) {
            return val.toFixed(precision);
        }
        
        // For very small numbers, use scientific notation
        if (val !== 0 && Math.abs(val) < 0.0001) {
            return val.toExponential(2);
        }
        
        // Auto-detect decimals from step
        const stepDecimals = (step.toString().split('.')[1] || '').length;
        return val.toFixed(Math.max(stepDecimals, 2));
    }
    
    _setValue(val) {
        const previousValue = this._value;
        this._value = this._clampValue(this._parseValue(val));
        this.input.value = this._formatValue(this._value);
        
        // Update button states
        this._updateButtonStates();
        
        if (previousValue !== this._value) {
            this._changeCallbacks.forEach(cb => cb(this._value, previousValue));
        }
    }
    
    _updateButtonStates() {
        // Disable decrease if at min
        if (this.options.min !== null && this._value <= this.options.min) {
            Q(this.decreaseBtn).addClass('disabled');
        } else {
            Q(this.decreaseBtn).removeClass('disabled');
        }
        
        // Disable increase if at max
        if (this.options.max !== null && this._value >= this.options.max) {
            Q(this.increaseBtn).addClass('disabled');
        } else {
            Q(this.increaseBtn).removeClass('disabled');
        }
    }
    
    /**
     * Get current value
     * @returns {number}
     */
    get() {
        return this._value;
    }
    
    /**
     * Set value
     * @param {number} value
     * @returns {NumberInput}
     */
    set(value) {
        this._setValue(value);
        return this;
    }
    
    /**
     * Register change callback
     * @param {Function} callback
     * @returns {NumberInput}
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
     * @returns {NumberInput}
     */
    enable() {
        this.options.disabled = false;
        Q(this.element).removeClass('disabled');
        this.input.disabled = false;
        this.decreaseBtn.disabled = false;
        this.increaseBtn.disabled = false;
        return this;
    }
    
    /**
     * Disable widget
     * @returns {NumberInput}
     */
    disable() {
        this.options.disabled = true;
        Q(this.element).addClass('disabled');
        this.input.disabled = true;
        this.decreaseBtn.disabled = true;
        this.increaseBtn.disabled = true;
        return this;
    }
    
    /**
     * Show widget
     * @returns {NumberInput}
     */
    show() {
        this.options.visible = true;
        Q(this.element).css('display', '');
        return this;
    }
    
    /**
     * Hide widget
     * @returns {NumberInput}
     */
    hide() {
        this.options.visible = false;
        Q(this.element).css('display', 'none');
        return this;
    }
}
