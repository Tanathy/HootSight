/**
 * HootSight - Array Widget
 * Schema-compatible tag/list input with add, edit, delete
 * 
 * Schema UI properties used:
 *   - widget: "array"
 *   - group: string (group name)
 *   - order: number (display order)
 *   - visible: boolean (default true)
 *   - disabled: boolean (default false)
 * 
 * Schema properties used:
 *   - type: "array"
 *   - items: { type: "string" }
 *   - default: string[] (default values)
 *   - description: string (tooltip/description)
 * 
 * Usage:
 *   const tags = new ArrayInput('extensions', {
 *       label: 'File Extensions',
 *       description: 'Supported image file extensions',
 *       default: ['.jpg', '.png', '.webp'],
 *       placeholder: 'Add extension...'
 *   });
 *   container.appendChild(tags.getElement());
 */

class ArrayInput {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            description: options.description || '',
            default: options.default || [],
            placeholder: options.placeholder || 'Add item...',
            disabled: options.disabled || false,
            visible: options.visible !== false
        };
        
        this._value = [...this.options.default];
        this._changeCallbacks = [];
        this._editingIndex = null;
        
        this._build();
    }
    
    _build() {
        // Main wrapper
        this.element = Q('<div>', { class: 'widget widget-array' }).get(0);
        this.element.id = this.id;
        
        if (!this.options.visible) {
            this.element.style.display = 'none';
        }
        
        // Label
        if (this.options.label) {
            this.labelEl = Q('<label>', { class: 'widget-label', text: this.options.label }).get(0);
            this.element.appendChild(this.labelEl);
        }
        
        // Tags container
        this.tagsContainer = Q('<div>', { class: 'array-tags' }).get(0);
        this.element.appendChild(this.tagsContainer);
        
        // Input row
        this.inputRow = Q('<div>', { class: 'array-input-row' }).get(0);
        
        this.input = Q('<input>', { type: 'text', class: 'array-input', placeholder: this.options.placeholder }).get(0);
        this.addBtn = Q('<button>', { type: 'button', class: 'array-add-btn', text: '+' }).get(0);
        
        this.inputRow.appendChild(this.input);
        this.inputRow.appendChild(this.addBtn);
        this.element.appendChild(this.inputRow);
        
        // Description
        if (this.options.description) {
            this.descEl = Q('<div>', { class: 'widget-description', text: this.options.description }).get(0);
            this.element.appendChild(this.descEl);
        }
        
        // Render initial tags
        this._renderTags();
        
        // Event listeners
        this._setupEvents();
        
        // Disabled state
        if (this.options.disabled) {
            this.disable();
        }
    }
    
    _renderTags() {
        this.tagsContainer.innerHTML = '';
        
        this._value.forEach((item, index) => {
            const tag = Q('<div>', { class: 'array-tag' }).get(0);
            
            if (this._editingIndex === index) {
                // Edit mode
                const editInput = Q('<input>', { type: 'text', class: 'array-tag-edit', value: item }).get(0);
                tag.appendChild(editInput);
                
                Q(editInput).on('blur', () => {
                    this._finishEdit(index, editInput.value);
                });
                
                Q(editInput).on('keydown', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        this._finishEdit(index, editInput.value);
                    } else if (e.key === 'Escape') {
                        this._cancelEdit();
                    }
                });
                
                // Focus after render
                setTimeout(() => editInput.focus(), 0);
            } else {
                // Display mode
                const text = Q('<span>', { class: 'array-tag-text', text: item }).get(0);
                const deleteBtn = Q('<span>', { class: 'array-tag-delete', text: '\u00D7' }).get(0);
                
                tag.appendChild(text);
                tag.appendChild(deleteBtn);
                
                // Double-click to edit
                Q(text).on('dblclick', () => {
                    if (!this.options.disabled) {
                        this._startEdit(index);
                    }
                });
                
                // Delete
                Q(deleteBtn).on('click', (e) => {
                    e.stopPropagation();
                    if (!this.options.disabled) {
                        this._removeItem(index);
                    }
                });
            }
            
            this.tagsContainer.appendChild(tag);
        });
    }
    
    _setupEvents() {
        // Add on button click
        Q(this.addBtn).on('click', () => {
            this._addItem();
        });
        
        // Add on Enter
        Q(this.input).on('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this._addItem();
            }
        });
    }
    
    _addItem() {
        const value = this.input.value.trim();
        if (value && !this.options.disabled) {
            const previousValue = [...this._value];
            this._value.push(value);
            this.input.value = '';
            this._renderTags();
            this._triggerChange(previousValue);
        }
    }
    
    _removeItem(index) {
        const previousValue = [...this._value];
        this._value.splice(index, 1);
        this._renderTags();
        this._triggerChange(previousValue);
    }
    
    _startEdit(index) {
        this._editingIndex = index;
        this._renderTags();
    }
    
    _finishEdit(index, newValue) {
        const trimmed = newValue.trim();
        if (trimmed) {
            const previousValue = [...this._value];
            this._value[index] = trimmed;
            this._editingIndex = null;
            this._renderTags();
            this._triggerChange(previousValue);
        } else {
            // Empty = remove
            this._editingIndex = null;
            this._removeItem(index);
        }
    }
    
    _cancelEdit() {
        this._editingIndex = null;
        this._renderTags();
    }
    
    _triggerChange(previousValue) {
        this._changeCallbacks.forEach(cb => cb([...this._value], previousValue));
    }
    
    /**
     * Get current value
     * @returns {string[]}
     */
    get() {
        return [...this._value];
    }
    
    /**
     * Set value
     * @param {string[]} value
     * @returns {ArrayInput}
     */
    set(value) {
        const previousValue = [...this._value];
        this._value = Array.isArray(value) ? [...value] : [];
        this._editingIndex = null;
        this._renderTags();
        this._triggerChange(previousValue);
        return this;
    }
    
    /**
     * Add an item
     * @param {string} item
     * @returns {ArrayInput}
     */
    add(item) {
        if (item && typeof item === 'string') {
            const previousValue = [...this._value];
            this._value.push(item.trim());
            this._renderTags();
            this._triggerChange(previousValue);
        }
        return this;
    }
    
    /**
     * Remove item by value
     * @param {string} item
     * @returns {ArrayInput}
     */
    remove(item) {
        const index = this._value.indexOf(item);
        if (index !== -1) {
            this._removeItem(index);
        }
        return this;
    }
    
    /**
     * Register change callback
     * @param {Function} callback - Called with (newValue, previousValue)
     * @returns {ArrayInput}
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
     * @returns {ArrayInput}
     */
    enable() {
        this.options.disabled = false;
        Q(this.element).removeClass('disabled');
        this.input.disabled = false;
        this.addBtn.disabled = false;
        return this;
    }
    
    /**
     * Disable widget
     * @returns {ArrayInput}
     */
    disable() {
        this.options.disabled = true;
        Q(this.element).addClass('disabled');
        this.input.disabled = true;
        this.addBtn.disabled = true;
        return this;
    }
    
    /**
     * Show widget
     * @returns {ArrayInput}
     */
    show() {
        this.options.visible = true;
        this.element.style.display = '';
        return this;
    }
    
    /**
     * Hide widget
     * @returns {ArrayInput}
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
     * @returns {ArrayInput}
     */
    static fromSchema(id, schema, lang = null) {
        const translate = lang || (key => key);
        
        return new ArrayInput(id, {
            label: translate(schema.description || ''),
            description: schema.ui?.description ? translate(schema.ui.description) : '',
            default: schema.default || [],
            placeholder: schema.ui?.placeholder ? translate(schema.ui.placeholder) : 'Add item...',
            disabled: schema.ui?.disabled || false,
            visible: schema.ui?.visible !== false
        });
    }
}

// Register in WidgetRegistry if available
if (typeof WidgetRegistry !== 'undefined') {
    WidgetRegistry.register('array', ArrayInput);
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ArrayInput;
}
