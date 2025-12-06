/**
 * HootSight - Dropdown Widget
 * Schema-compatible dropdown/select component
 * 
 * Schema UI properties used:
 *   - widget: "dropdown"
 *   - group: string (group name)
 *   - order: number (display order)
 *   - visible: boolean (default true)
 *   - disabled: boolean (default false)
 * 
 * Schema properties used:
 *   - type: "string"
 *   - enum: string[] (options)
 *   - enum_descriptor: { [value]: "localization.key" } (dynamic description per selected value)
 *   - default: string (default value)
 *   - description: string (static tooltip/description)
 * 
 * Usage:
 *   const dropdown = new Dropdown('model_type', {
 *       label: 'Model Type',
 *       description: 'Select the model architecture',
 *       options: ['resnet', 'mobilenet', 'efficientnet'],
 *       optionLabels: { resnet: 'ResNet', mobilenet: 'MobileNet', efficientnet: 'EfficientNet' },
 *       enumDescriptor: { resnet: 'schema.model.task.resnet_desc', mobilenet: 'schema.model.task.mobilenet_desc' },
 *       default: 'resnet'
 *   });
 *   container.appendChild(dropdown.getElement());
 */

class Dropdown {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            description: options.description || '',
            options: options.options || [],
            optionLabels: options.optionLabels || {},
            enumDescriptor: options.enumDescriptor || null,
            default: options.default || null,
            disabled: options.disabled || false,
            visible: options.visible !== false
        };
        
        this._value = this.options.default || (this.options.options.length > 0 ? this.options.options[0] : null);
        this._changeCallbacks = [];
        this._isOpen = false;
        
        this._build();
    }
    
    _build() {
        // Main wrapper
        this.element = Q('<div>', { class: 'widget widget-dropdown' }).get(0);
        this.element.id = this.id;
        
        if (!this.options.visible) {
            this.element.style.display = 'none';
        }
        
        // Label
        if (this.options.label) {
            this.labelEl = Q('<label>', { class: 'widget-label', text: this.options.label }).get(0);
            this.labelEl.setAttribute('for', this.id + '-select');
            // Add lang key attribute for live translation
            if (this.options.labelLangKey) {
                this.labelEl.setAttribute('data-lang-key', this.options.labelLangKey);
            }
            Q(this.element).append(this.labelEl);
        }
        
        // Dropdown container
        this.dropdownContainer = Q('<div>', { class: 'dropdown-container' }).get(0);
        
        // Selected display
        this.selectedDisplay = Q('<div>', { class: 'dropdown-selected' }).get(0);
        
        // Arrow (create once, reused in _updateSelectedDisplay)
        this.arrow = Q('<span>', { class: 'dropdown-arrow', text: '\u25BC' }).get(0);
        
        this._updateSelectedDisplay();
        
        // Options container
        this.optionsContainer = Q('<div>', { class: 'dropdown-options' }).get(0);
        this._buildOptions();
        
        // Hidden input for form compatibility
        this.hiddenInput = Q('<input>', { type: 'hidden', id: this.id + '-select' }).get(0);
        this.hiddenInput.value = this._value || '';
        
        // Assemble
        Q(this.dropdownContainer).append(this.selectedDisplay).append(this.optionsContainer);
        Q(this.element).append(this.dropdownContainer).append(this.hiddenInput);
        
        // Description (static)
        if (this.options.description) {
            this.descEl = Q('<div>', { class: 'widget-description', text: this.options.description }).get(0);
            // Add lang key attribute for live translation
            if (this.options.descriptionLangKey) {
                this.descEl.setAttribute('data-lang-key', this.options.descriptionLangKey);
            }
            Q(this.element).append(this.descEl);
        }
        
        // Dynamic description from enum_descriptor (changes based on selected value)
        if (this.options.enumDescriptor) {
            this.dynamicDescEl = Q('<div>', { class: 'widget-dynamic-description' }).get(0);
            Q(this.element).append(this.dynamicDescEl);
            this._updateDynamicDescription();
        }
        
        // Event listeners
        this._setupEvents();
        
        // Disabled state
        if (this.options.disabled) {
            this.disable();
        }
    }
    
    _buildOptions() {
        Q(this.optionsContainer).empty();
        
        this.options.options.forEach(value => {
            const label = this.options.optionLabels[value] || value;
            const optionEl = Q('<div>', { class: 'dropdown-option', text: label }).get(0);
            optionEl.dataset.value = value;
            
            if (value === this._value) {
                Q(optionEl).addClass('selected');
            }
            
            Q(optionEl).on('click', (e) => {
                e.stopPropagation();
                this._selectOption(value);
            });
            
            Q(this.optionsContainer).append(optionEl);
        });
    }
    
    _updateSelectedDisplay() {
        const label = this.options.optionLabels[this._value] || this._value || 'Select...';
        
        // Clear and rebuild
        Q(this.selectedDisplay).empty();
        
        // Add text
        const textSpan = Q('<span>', { text: label }).get(0);
        Q(this.selectedDisplay).append(textSpan);
        
        // Re-add arrow
        if (!this.arrow) {
            this.arrow = Q('<span>', { class: 'dropdown-arrow', text: '\u25BC' }).get(0);
        }
        Q(this.selectedDisplay).append(this.arrow);
    }
    
    _updateDynamicDescription() {
        if (!this.dynamicDescEl || !this.options.enumDescriptor) return;
        
        const descKey = this.options.enumDescriptor[this._value];
        if (descKey) {
            // Use lang() for localization, fallback to key if lang not available
            const text = typeof lang === 'function' ? lang(descKey) : descKey;
            Q(this.dynamicDescEl).text(text);
            this.dynamicDescEl.style.display = '';
        } else {
            // No description for this value, hide the element
            Q(this.dynamicDescEl).text('');
            this.dynamicDescEl.style.display = 'none';
        }
    }
    
    _selectOption(value) {
        const previousValue = this._value;
        this._value = value;
        this.hiddenInput.value = value;
        
        this._updateSelectedDisplay();
        this._updateDynamicDescription();
        this._close();
        
        // Update selected state in options
        Q(this.optionsContainer).find('.dropdown-option').getAll().forEach(opt => {
            Q(opt).removeClass('selected');
            if (opt.dataset.value === value) {
                Q(opt).addClass('selected');
            }
        });
        
        // Trigger callbacks
        if (previousValue !== value) {
            this._changeCallbacks.forEach(cb => cb(value, previousValue));
        }
    }
    
    _setupEvents() {
        // Toggle on click
        Q(this.selectedDisplay).on('click', (e) => {
            e.stopPropagation();
            if (this._isOpen) {
                this._close();
            } else {
                this._open();
            }
        });
        
        // Close on outside click
        Q(document).on('click', (e) => {
            if (this._isOpen && !this.element.contains(e.target)) {
                this._close();
            }
        });
        
        // Keyboard navigation
        Q(this.selectedDisplay).on('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                if (this._isOpen) {
                    this._close();
                } else {
                    this._open();
                }
            } else if (e.key === 'Escape') {
                this._close();
            } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                this._navigateOptions(e.key === 'ArrowDown' ? 1 : -1);
            }
        });
        
        // Make focusable
        this.selectedDisplay.setAttribute('tabindex', '0');
    }
    
    _open() {
        if (this.options.disabled) return;
        this._isOpen = true;
        
        // Portal the options to body for proper overflow handling
        this._portalOptions();
        
        Q(this.dropdownContainer).addClass('open');
    }
    
    _portalOptions() {
        // Move options container to body for proper positioning above overflow containers
        if (!this._isPortaled) {
            Q(document.body).append(this.optionsContainer);
            this._isPortaled = true;
            
            // Add portal-specific styles
            this.optionsContainer.style.position = 'fixed';
            this.optionsContainer.style.zIndex = '10000';
        }
        
        // Position the options relative to the selected display
        this._positionPortaledOptions();
        
        // Update position on scroll/resize
        this._scrollHandler = () => this._positionPortaledOptions();
        Q(window).on('scroll', this._scrollHandler, true);
        Q(window).on('resize', this._scrollHandler);
    }
    
    _positionPortaledOptions() {
        const rect = this.selectedDisplay.getBoundingClientRect();
        const optionsHeight = Math.min(this.optionsContainer.scrollHeight, 200);
        const viewportHeight = window.innerHeight;
        
        // Calculate available space
        const spaceBelow = viewportHeight - rect.bottom - 8;
        const spaceAbove = rect.top - 8;
        
        // Determine direction
        const openUpward = optionsHeight > spaceBelow && spaceAbove > spaceBelow;
        
        // Position the options
        this.optionsContainer.style.left = rect.left + 'px';
        this.optionsContainer.style.width = rect.width + 'px';
        
        if (openUpward) {
            this.optionsContainer.style.top = 'auto';
            this.optionsContainer.style.bottom = (viewportHeight - rect.top + 4) + 'px';
            Q(this.dropdownContainer).addClass('dropup');
        } else {
            this.optionsContainer.style.top = (rect.bottom + 4) + 'px';
            this.optionsContainer.style.bottom = 'auto';
            Q(this.dropdownContainer).removeClass('dropup');
        }
        
        // Show the options
        this.optionsContainer.style.display = 'block';
    }
    
    _unportalOptions() {
        if (this._isPortaled) {
            // Hide and move back to container
            this.optionsContainer.style.display = 'none';
            Q(this.dropdownContainer).append(this.optionsContainer);
            this._isPortaled = false;
            
            // Reset styles
            this.optionsContainer.style.position = '';
            this.optionsContainer.style.zIndex = '';
            this.optionsContainer.style.left = '';
            this.optionsContainer.style.top = '';
            this.optionsContainer.style.bottom = '';
            this.optionsContainer.style.width = '';
            
            // Remove listeners
            if (this._scrollHandler) {
                window.removeEventListener('scroll', this._scrollHandler, true);
                window.removeEventListener('resize', this._scrollHandler);
                this._scrollHandler = null;
            }
        }
    }
    
    _close() {
        this._isOpen = false;
        this._unportalOptions();
        Q(this.dropdownContainer).removeClass('open').removeClass('dropup');
    }
    
    _navigateOptions(direction) {
        const currentIndex = this.options.options.indexOf(this._value);
        let newIndex = currentIndex + direction;
        
        if (newIndex < 0) newIndex = this.options.options.length - 1;
        if (newIndex >= this.options.options.length) newIndex = 0;
        
        this._selectOption(this.options.options[newIndex]);
    }
    
    /**
     * Get current value
     * @returns {string|null}
     */
    get() {
        return this._value;
    }
    
    /**
     * Set value
     * @param {string} value
     * @returns {Dropdown}
     */
    set(value) {
        if (this.options.options.includes(value)) {
            this._selectOption(value);
        }
        return this;
    }
    
    /**
     * Update options list
     * @param {string[]} options - New options array
     * @param {Object} optionLabels - New option labels
     * @returns {Dropdown}
     */
    setOptions(options, optionLabels = {}) {
        this.options.options = options;
        this.options.optionLabels = { ...this.options.optionLabels, ...optionLabels };
        
        // Reset value if current is not in new options
        if (!options.includes(this._value)) {
            this._value = options.length > 0 ? options[0] : null;
            this.hiddenInput.value = this._value || '';
        }
        
        this._buildOptions();
        this._updateSelectedDisplay();
        return this;
    }

    /**
     * Update a single option's label
     * @param {string} value - Option value to update
     * @param {string} label - New label for the option
     * @returns {Dropdown}
     */
    updateOptionLabel(value, label) {
        this.options.optionLabels[value] = label;
        
        // Update option element in list
        const optionEl = Q(this.optionsContainer).find(`[data-value="${value}"]`).get(0);
        if (optionEl) {
            Q(optionEl).text(label);
        }
        
        // Update selected display if this is the current value
        if (this._value === value) {
            this._updateSelectedDisplay();
        }
        
        return this;
    }
    
    /**
     * Register change callback
     * @param {Function} callback - Called with (newValue, previousValue)
     * @returns {Dropdown}
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
     * @returns {Dropdown}
     */
    enable() {
        this.options.disabled = false;
        Q(this.element).removeClass('disabled');
        return this;
    }
    
    /**
     * Disable widget
     * @returns {Dropdown}
     */
    disable() {
        this.options.disabled = true;
        Q(this.element).addClass('disabled');
        this._close();
        return this;
    }
    
    /**
     * Show widget
     * @returns {Dropdown}
     */
    show() {
        this.options.visible = true;
        this.element.style.display = '';
        return this;
    }
    
    /**
     * Hide widget
     * @returns {Dropdown}
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
     * @returns {Dropdown}
     */
    static fromSchema(id, schema, lang = null) {
        const translate = lang || (key => key);
        
        const options = schema.enum || [];
        const optionLabels = {};
        
        if (schema.enum_descriptor) {
            for (const [value, key] of Object.entries(schema.enum_descriptor)) {
                optionLabels[value] = translate(key);
            }
        }
        
        return new Dropdown(id, {
            label: translate(schema.description || ''),
            description: schema.ui?.description ? translate(schema.ui.description) : '',
            options: options,
            optionLabels: optionLabels,
            default: schema.default,
            disabled: schema.ui?.disabled || false,
            visible: schema.ui?.visible !== false
        });
    }
}

// Register in WidgetRegistry if available
if (typeof WidgetRegistry !== 'undefined') {
    WidgetRegistry.register('dropdown', Dropdown);
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Dropdown;
}
