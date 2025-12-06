/**
 * HootSight - AutoOrNumber Widget
 * Schema-compatible auto/manual number input using qte.js
 * 
 * Combines a Switch (Auto mode) with a NumberInput for manual values.
 * When Auto is ON, the number input is disabled.
 * When Auto is OFF, user can enter a manual value.
 * 
 * Schema UI properties used:
 *   - widget: "auto_or_number"
 *   - group: string (group name)
 *   - order: number (display order)
 * 
 * Schema properties used:
 *   - oneOf: [{ type: "string", enum: ["auto"] }, { type: "integer/number", ... }]
 *   - default: "auto" | number
 * 
 * Usage:
 *   const widget = new AutoOrNumber('batch_size', {
 *       label: 'Batch Size',
 *       min: 1,
 *       max: 512,
 *       default: 'auto'  // or a number
 *   });
 *   container.appendChild(widget.getElement());
 */

class AutoOrNumber {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            description: options.description || '',
            min: options.min ?? 1,
            max: options.max ?? 100,
            step: options.step ?? 1,
            default: options.default ?? 'auto',
            numberDefault: options.numberDefault ?? options.min ?? 1,
            integer: options.integer ?? true,
            precision: options.precision ?? null, // decimal places for display
            disabled: options.disabled || false,
            visible: options.visible !== false
        };
        
        // Determine initial state
        this._isAuto = this.options.default === 'auto';
        this._numberValue = typeof this.options.default === 'number' 
            ? this.options.default 
            : this.options.numberDefault;
        
        this._changeCallbacks = [];
        
        this._build();
    }
    
    _build() {
        // Main wrapper
        this.element = Q('<div>', { class: 'widget widget-auto-or-number' }).get(0);
        this.element.id = this.id;
        
        if (!this.options.visible) {
            Q(this.element).css('display', 'none');
        }
        
        // Label
        if (this.options.label) {
            this.labelEl = Q('<label>', { 
                class: 'widget-label', 
                text: this.options.label 
            }).get(0);
            // Add lang key attribute for live translation
            if (this.options.labelLangKey) {
                this.labelEl.setAttribute('data-lang-key', this.options.labelLangKey);
            }
            Q(this.element).append(this.labelEl);
        }
        
        // Controls container
        this.controlsContainer = Q('<div>', { class: 'auto-or-number-controls' }).get(0);
        
        // Auto switch (mini version using standard switch styling)
        this.autoSwitchContainer = Q('<div>', { class: 'auto-switch-container' }).get(0);
        
        this.autoLabel = Q('<span>', { 
            class: 'auto-label', 
            text: 'Auto' 
        }).get(0);
        
        // Build switch using standard classes
        this.switchTrack = Q('<div>', { class: 'switch-track' }).get(0);
        this.switchThumb = Q('<div>', { class: 'switch-thumb' }).get(0);
        Q(this.switchTrack).append(this.switchThumb);
        
        if (this._isAuto) {
            Q(this.switchTrack).addClass('active');
        }
        
        Q(this.autoSwitchContainer).append(this.autoLabel);
        Q(this.autoSwitchContainer).append(this.switchTrack);
        
        // Number input section
        this.numberContainer = Q('<div>', { class: 'number-section' }).get(0);
        
        // Decrease button
        this.decreaseBtn = Q('<button>', { 
            class: 'number-btn number-decrease',
            type: 'button',
            text: '-'
        }).get(0);
        
        // Input
        this.numberInput = Q('<input>', {
            type: 'number',
            class: 'widget-input number-input',
            id: this.id + '-input',
            value: this._numberValue,
            min: this.options.min,
            max: this.options.max,
            step: this.options.step
        }).get(0);
        
        // Increase button
        this.increaseBtn = Q('<button>', { 
            class: 'number-btn number-increase',
            type: 'button',
            text: '+'
        }).get(0);
        
        Q(this.numberContainer).append(this.decreaseBtn);
        Q(this.numberContainer).append(this.numberInput);
        Q(this.numberContainer).append(this.increaseBtn);
        
        // Assemble controls
        Q(this.controlsContainer).append(this.autoSwitchContainer);
        Q(this.controlsContainer).append(this.numberContainer);
        Q(this.element).append(this.controlsContainer);
        
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
            Q(this.element).append(this.descEl);
        }
        
        this._updateState();
        this._setupEvents();
        
        if (this.options.disabled) {
            this.disable();
        }
    }
    
    _setupEvents() {
        // Auto switch toggle
        Q(this.switchTrack).on('click', () => {
            if (this.options.disabled) return;
            this._isAuto = !this._isAuto;
            this._updateState();
            this._notifyChange();
        });
        
        Q(this.autoLabel).on('click', () => {
            if (this.options.disabled) return;
            this._isAuto = !this._isAuto;
            this._updateState();
            this._notifyChange();
        });
        
        // Number input change
        Q(this.numberInput).on('change', () => {
            this._setNumberValue(this._parseValue(this.numberInput.value));
            this._notifyChange();
        });
        
        // Decrease button
        Q(this.decreaseBtn).on('click', () => {
            if (this.options.disabled || this._isAuto) return;
            this._setNumberValue(this._numberValue - this.options.step);
            this._notifyChange();
        });
        
        // Increase button
        Q(this.increaseBtn).on('click', () => {
            if (this.options.disabled || this._isAuto) return;
            this._setNumberValue(this._numberValue + this.options.step);
            this._notifyChange();
        });
        
        // Keyboard support
        Q(this.numberInput).on('keydown', (e) => {
            if (this._isAuto) return;
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                this._setNumberValue(this._numberValue + this.options.step);
                this._notifyChange();
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                this._setNumberValue(this._numberValue - this.options.step);
                this._notifyChange();
            }
        });
    }
    
    _parseValue(val) {
        const parsed = this.options.integer ? parseInt(val, 10) : parseFloat(val);
        return isNaN(parsed) ? this.options.numberDefault : parsed;
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
    
    _setNumberValue(val) {
        this._numberValue = this._clampValue(this._parseValue(val));
        this.numberInput.value = this._formatValue(this._numberValue);
        this._updateButtonStates();
    }
    
    _updateState() {
        if (this._isAuto) {
            Q(this.switchTrack).addClass('active');
            Q(this.numberContainer).addClass('disabled');
            this.numberInput.disabled = true;
            this.decreaseBtn.disabled = true;
            this.increaseBtn.disabled = true;
        } else {
            Q(this.switchTrack).removeClass('active');
            Q(this.numberContainer).removeClass('disabled');
            this.numberInput.disabled = false;
            this.decreaseBtn.disabled = false;
            this.increaseBtn.disabled = false;
        }
        this._updateButtonStates();
    }
    
    _updateButtonStates() {
        if (this._isAuto) return;
        
        // Disable decrease if at min
        if (this.options.min !== null && this._numberValue <= this.options.min) {
            Q(this.decreaseBtn).addClass('disabled');
        } else {
            Q(this.decreaseBtn).removeClass('disabled');
        }
        
        // Disable increase if at max
        if (this.options.max !== null && this._numberValue >= this.options.max) {
            Q(this.increaseBtn).addClass('disabled');
        } else {
            Q(this.increaseBtn).removeClass('disabled');
        }
    }
    
    _notifyChange() {
        const value = this.get();
        this._changeCallbacks.forEach(cb => cb(value));
    }
    
    /**
     * Get current value
     * @returns {string|number} "auto" or number
     */
    get() {
        return this._isAuto ? 'auto' : this._numberValue;
    }
    
    /**
     * Set value
     * @param {string|number} value - "auto" or number
     * @returns {AutoOrNumber}
     */
    set(value) {
        if (value === 'auto') {
            this._isAuto = true;
        } else {
            this._isAuto = false;
            this._setNumberValue(value);
        }
        this._updateState();
        return this;
    }
    
    /**
     * Register change callback
     * @param {Function} callback - Called with (value)
     * @returns {AutoOrNumber}
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
     * @returns {AutoOrNumber}
     */
    enable() {
        this.options.disabled = false;
        Q(this.element).removeClass('disabled');
        this._updateState();
        return this;
    }
    
    /**
     * Disable widget
     * @returns {AutoOrNumber}
     */
    disable() {
        this.options.disabled = true;
        Q(this.element).addClass('disabled');
        this.numberInput.disabled = true;
        this.decreaseBtn.disabled = true;
        this.increaseBtn.disabled = true;
        return this;
    }
    
    /**
     * Show widget
     * @returns {AutoOrNumber}
     */
    show() {
        this.options.visible = true;
        Q(this.element).css('display', '');
        return this;
    }
    
    /**
     * Hide widget
     * @returns {AutoOrNumber}
     */
    hide() {
        this.options.visible = false;
        Q(this.element).css('display', 'none');
        return this;
    }
}
