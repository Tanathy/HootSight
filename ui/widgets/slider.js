/**
 * Slider Widget with Number Input
 * Custom slider with synchronized number input field
 * 
 * Supports:
 *   - Linear scale (default)
 *   - Logarithmic scale (scale: "log") - ideal for learning rate, weight decay, etc.
 *   - Custom precision for floating point values
 */
class Slider {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            labelLangKey: options.labelLangKey || null,
            description: options.description || '',
            descriptionLangKey: options.descriptionLangKey || null,
            min: options.min ?? 0,
            max: options.max ?? 100,
            step: options.step ?? 1,
            default: options.default ?? options.min ?? 0,
            disabled: options.disabled || false,
            scale: options.scale || 'linear',  // 'linear' or 'log'
            precision: options.precision ?? null, // decimal places for display
            ...options
        };
        
        // For log scale, ensure min > 0
        if (this.options.scale === 'log' && this.options.min <= 0) {
            this.options.min = 1e-8;
        }
        
        this._value = this._clamp(this.options.default);
        this._callbacks = [];
        this._element = null;
        this._track = null;
        this._fill = null;
        this._thumb = null;
        this._input = null;
        this._isDragging = false;
        
        this._build();
    }
    
    _clamp(value) {
        const { min, max, step, scale } = this.options;
        
        // For log scale, just clamp to range (no stepping)
        if (scale === 'log') {
            return Math.min(max, Math.max(min, value));
        }
        
        // Linear: Round to step
        const stepped = Math.round((value - min) / step) * step + min;
        return Math.min(max, Math.max(min, stepped));
    }
    
    _valueToPercent(value) {
        const { min, max, scale } = this.options;
        if (max === min) return 0;
        
        if (scale === 'log') {
            // Logarithmic scale
            const logMin = Math.log10(min);
            const logMax = Math.log10(max);
            const logValue = Math.log10(value);
            return ((logValue - logMin) / (logMax - logMin)) * 100;
        }
        
        // Linear scale
        return ((value - min) / (max - min)) * 100;
    }
    
    _percentToValue(percent) {
        const { min, max, scale } = this.options;
        
        if (scale === 'log') {
            // Logarithmic scale
            const logMin = Math.log10(min);
            const logMax = Math.log10(max);
            const logValue = logMin + (percent / 100) * (logMax - logMin);
            return Math.pow(10, logValue);
        }
        
        // Linear scale
        return min + (percent / 100) * (max - min);
    }
    
    _build() {
        // Container
        this._element = Q('<div>', { class: 'slider-container', id: `slider-${this.id}` }).get();
        
        // Label row
        if (this.options.label) {
            const labelRow = Q('<div>', { class: 'slider-label-row' });
            
            const labelAttrs = { class: 'slider-label', text: this.options.label };
            if (this.options.labelLangKey) {
                labelAttrs['data-lang-key'] = this.options.labelLangKey;
            }
            const label = Q('<label>', labelAttrs).get();
            labelRow.append(label);
            
            Q(this._element).append(labelRow.get());
        }
        
        // Control row (slider + input)
        const controlRow = Q('<div>', { class: 'slider-control-row' });
        
        // Slider wrapper
        const sliderWrapper = Q('<div>', { class: 'slider-wrapper' });
        
        // Track
        this._track = Q('<div>', { class: 'slider-track' }).get();
        
        // Fill
        this._fill = Q('<div>', { class: 'slider-fill' }).get();
        Q(this._track).append(this._fill);
        
        // Thumb
        this._thumb = Q('<div>', { class: 'slider-thumb' }).get();
        Q(this._track).append(this._thumb);
        
        sliderWrapper.append(this._track);
        controlRow.append(sliderWrapper.get());
        
        // Number input
        this._input = Q('<input>', { type: 'text', class: 'slider-input' }).get();
        this._input.value = this._formatValue(this._value);
        
        if (this.options.disabled) {
            this._input.disabled = true;
        }
        
        controlRow.append(this._input);
        Q(this._element).append(controlRow.get());
        
        // Description
        if (this.options.description) {
            const descAttrs = { class: 'slider-description', text: this.options.description };
            if (this.options.descriptionLangKey) {
                descAttrs['data-lang-key'] = this.options.descriptionLangKey;
            }
            const desc = Q('<div>', descAttrs).get();
            Q(this._element).append(desc);
        }
        
        // Update visual
        this._updateVisual();
        
        // Bind events
        this._bindEvents();
        
        // Disabled state
        if (this.options.disabled) {
            Q(this._element).addClass('disabled');
        }
    }
    
    _formatValue(value) {
        const { precision, step, scale } = this.options;
        
        // Use explicit precision if provided
        if (precision !== null) {
            return value.toFixed(precision);
        }
        
        // For log scale, use scientific notation for very small numbers
        if (scale === 'log' && value < 0.001) {
            return value.toExponential(2);
        }
        
        // Auto-detect decimals from step
        const stepDecimals = (step.toString().split('.')[1] || '').length;
        return value.toFixed(Math.max(stepDecimals, scale === 'log' ? 6 : 0));
    }
    
    _updateVisual() {
        const percent = this._valueToPercent(this._value);
        this._fill.style.width = `${percent}%`;
        this._thumb.style.left = `${percent}%`;
    }
    
    _bindEvents() {
        // Track click/drag
        Q(this._track).on('mousedown', (e) => this._onTrackMouseDown(e));
        Q(document).on('mousemove', (e) => this._onDocumentMouseMove(e));
        Q(document).on('mouseup', () => this._onDocumentMouseUp());
        
        // Input events
        Q(this._input).on('input', () => this._onInputChange());
        Q(this._input).on('blur', () => this._onInputBlur());
        Q(this._input).on('keydown', (e) => {
            if (e.key === 'Enter') {
                this._onInputBlur();
                this._input.blur();
            }
        });
        
        // Keyboard on track
        this._track.tabIndex = 0;
        Q(this._track).on('keydown', (e) => this._onTrackKeyDown(e));
    }
    
    _onTrackMouseDown(e) {
        if (this.options.disabled) return;
        
        e.preventDefault();
        this._isDragging = true;
        Q(this._element).addClass('dragging');
        this._updateFromMouse(e);
    }
    
    _onDocumentMouseMove(e) {
        if (!this._isDragging) return;
        this._updateFromMouse(e);
    }
    
    _onDocumentMouseUp() {
        if (!this._isDragging) return;
        this._isDragging = false;
        Q(this._element).removeClass('dragging');
    }
    
    _updateFromMouse(e) {
        const rect = this._track.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
        const rawValue = this._percentToValue(percent);
        const newValue = this._clamp(rawValue);
        
        if (newValue !== this._value) {
            const oldValue = this._value;
            this._value = newValue;
            this._input.value = this._formatValue(newValue);
            this._updateVisual();
            this._fireChange(oldValue);
        }
    }
    
    _onInputChange() {
        // Allow typing - don't validate on every keystroke
    }
    
    _onInputBlur() {
        const parsed = parseFloat(this._input.value);
        
        if (isNaN(parsed)) {
            // Invalid - reset to current value
            this._input.value = this._formatValue(this._value);
            return;
        }
        
        const newValue = this._clamp(parsed);
        
        if (newValue !== this._value) {
            const oldValue = this._value;
            this._value = newValue;
            this._input.value = this._formatValue(newValue);
            this._updateVisual();
            this._fireChange(oldValue);
        } else {
            // Value same but might need formatting fix
            this._input.value = this._formatValue(this._value);
        }
    }
    
    _onTrackKeyDown(e) {
        if (this.options.disabled) return;
        
        const { step, min, max } = this.options;
        let newValue = this._value;
        
        switch (e.key) {
            case 'ArrowRight':
            case 'ArrowUp':
                newValue = this._clamp(this._value + step);
                break;
            case 'ArrowLeft':
            case 'ArrowDown':
                newValue = this._clamp(this._value - step);
                break;
            case 'Home':
                newValue = min;
                break;
            case 'End':
                newValue = max;
                break;
            case 'PageUp':
                newValue = this._clamp(this._value + step * 10);
                break;
            case 'PageDown':
                newValue = this._clamp(this._value - step * 10);
                break;
            default:
                return;
        }
        
        e.preventDefault();
        
        if (newValue !== this._value) {
            const oldValue = this._value;
            this._value = newValue;
            this._input.value = this._formatValue(newValue);
            this._updateVisual();
            this._fireChange(oldValue);
        }
    }
    
    _fireChange(oldValue) {
        for (const cb of this._callbacks) {
            cb(this._value, oldValue);
        }
    }
    
    // Public API
    get() {
        return this._value;
    }
    
    set(value, silent = false) {
        const newValue = this._clamp(value);
        if (newValue === this._value) return;
        
        const oldValue = this._value;
        this._value = newValue;
        this._input.value = this._formatValue(newValue);
        this._updateVisual();
        
        if (!silent) {
            this._fireChange(oldValue);
        }
    }
    
    setMin(min) {
        this.options.min = min;
        this.set(this._value);
    }
    
    setMax(max) {
        this.options.max = max;
        this.set(this._value);
    }
    
    setRange(min, max) {
        this.options.min = min;
        this.options.max = max;
        this.set(this._value);
    }
    
    onChange(callback) {
        this._callbacks.push(callback);
        return this;
    }
    
    getElement() {
        return this._element;
    }
    
    setDisabled(disabled) {
        this.options.disabled = disabled;
        this._input.disabled = disabled;
        Q(this._element).toggleClass('disabled', disabled);
    }
    
    // Schema compatibility
    static fromSchema(id, schema) {
        const ui = schema.ui || {};
        return new Slider(id, {
            label: schema.title || id,
            description: schema.description || '',
            min: schema.minimum ?? 0,
            max: schema.maximum ?? 100,
            step: schema.step ?? 1,
            default: schema.default,
            disabled: ui.disabled || false
        });
    }
}
