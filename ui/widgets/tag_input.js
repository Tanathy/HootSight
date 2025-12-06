/**
 * HootSight - TagInput Widget
 * Tag/label input with autocomplete suggestions
 * 
 * Features:
 *   - Add tags via Enter key or clicking suggestions
 *   - Remove tags via X button
 *   - Autocomplete with recent and matching suggestions
 *   - Keyboard navigation (arrow keys, escape)
 *   - Comma-separated values as alternative input
 * 
 * Usage:
 *   const tags = new TagInput('image-tags', {
 *       label: 'Tags',
 *       placeholder: '+ add tag',
 *       suggestions: ['cat', 'dog', 'bird'],
 *       recentTags: ['cat', 'dog']
 *   });
 *   container.appendChild(tags.getElement());
 *   
 *   tags.onChange((newTags, added, removed) => {
 *       console.log('Tags changed:', newTags);
 *   });
 */

class TagInput {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            labelLangKey: options.labelLangKey || null,
            description: options.description || '',
            descriptionLangKey: options.descriptionLangKey || null,
            placeholder: options.placeholder || '+ tag',
            placeholderLangKey: options.placeholderLangKey || null,
            disabled: options.disabled || false,
            visible: options.visible !== false,
            suggestions: options.suggestions || [],      // All available tags
            recentTags: options.recentTags || [],        // Recently used tags (shown first)
            maxSuggestions: options.maxSuggestions || 8,
            allowCustom: options.allowCustom !== false,  // Allow custom tags not in suggestions
            delimiter: options.delimiter || ','          // Delimiter for splitting input
        };
        
        this._tags = [];
        this._changeCallbacks = [];
        
        this._build();
    }
    
    _build() {
        // Main wrapper
        this.element = Q('<div>', { class: 'widget widget-tag-input' }).get(0);
        this.element.id = this.id;
        
        if (!this.options.visible) {
            this.element.style.display = 'none';
        }
        
        // Label
        if (this.options.label) {
            this.labelEl = Q('<label>', { class: 'widget-label', text: this.options.label }).get(0);
            if (this.options.labelLangKey) {
                this.labelEl.setAttribute('data-lang-key', this.options.labelLangKey);
            }
            Q(this.element).append(this.labelEl);
        }
        
        // Tags container (holds tags + input)
        this.tagsContainer = Q('<div>', { class: 'tag-input-container' }).get(0);
        
        // Tags wrapper
        this.tagsWrapper = Q('<div>', { class: 'tag-input-tags' }).get(0);
        Q(this.tagsContainer).append(this.tagsWrapper);
        
        // Input wrapper (for positioning suggestions)
        this.inputWrapper = Q('<div>', { class: 'tag-input-wrapper' }).get(0);
        
        // Input
        this.input = Q('<input>', { 
            type: 'text', 
            class: 'tag-input-field',
            placeholder: this.options.placeholder
        }).get(0);
        if (this.options.placeholderLangKey) {
            this.input.setAttribute('data-lang-key', this.options.placeholderLangKey);
            this.input.setAttribute('data-lang-placeholder', 'true');
        }
        Q(this.inputWrapper).append(this.input);
        
        // Suggestions dropdown
        this.suggestionsEl = Q('<div>', { class: 'tag-input-suggestions hidden' }).get(0);
        Q(this.inputWrapper).append(this.suggestionsEl);
        
        Q(this.tagsContainer).append(this.inputWrapper);
        Q(this.element).append(this.tagsContainer);
        
        // Description
        if (this.options.description) {
            this.descEl = Q('<div>', { class: 'widget-description', text: this.options.description }).get(0);
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
        // Input events
        Q(this.input).on('input', () => this._handleInput());
        Q(this.input).on('keydown', (e) => this._handleKeydown(e));
        Q(this.input).on('focus', () => this._handleFocus());
        Q(this.input).on('blur', () => this._handleBlur());
        
        // Click on container focuses input
        Q(this.tagsContainer).on('click', (e) => {
            if (e.target === this.tagsContainer || e.target === this.tagsWrapper) {
                this.input.focus();
            }
        });
    }
    
    _handleInput() {
        const value = this.input.value;
        
        // Check for delimiter input (e.g., comma)
        if (value.includes(this.options.delimiter)) {
            const parts = value.split(this.options.delimiter);
            // Add all complete parts as tags
            parts.slice(0, -1).forEach(part => {
                const trimmed = part.trim();
                if (trimmed) {
                    this._addTagInternal(trimmed);
                }
            });
            // Keep the last part in input
            this.input.value = parts[parts.length - 1].trim();
        }
        
        this._showSuggestions();
    }
    
    _handleKeydown(e) {
        switch (e.key) {
            case 'Enter':
                e.preventDefault();
                const selected = Q(this.suggestionsEl).find('.selected').get(0);
                if (selected) {
                    this._addTagInternal(selected.dataset.tag);
                } else {
                    const value = this.input.value.trim();
                    if (value) {
                        this._addTagInternal(value);
                    }
                }
                this.input.value = '';
                this._hideSuggestions();
                break;
                
            case 'Escape':
                this._hideSuggestions();
                this.input.blur();
                break;
                
            case 'Backspace':
                if (this.input.value === '' && this._tags.length > 0) {
                    // Remove last tag when backspace on empty input
                    this._removeTagInternal(this._tags[this._tags.length - 1]);
                }
                break;
                
            case 'ArrowDown':
                e.preventDefault();
                this._navigateSuggestions(1);
                break;
                
            case 'ArrowUp':
                e.preventDefault();
                this._navigateSuggestions(-1);
                break;
        }
    }
    
    _handleFocus() {
        Q(this.tagsContainer).addClass('focused');
        if (this.input.value.length > 0 || this.options.recentTags.length > 0) {
            this._showSuggestions();
        }
    }
    
    _handleBlur() {
        Q(this.tagsContainer).removeClass('focused');
        // Delay to allow click on suggestion
        setTimeout(() => this._hideSuggestions(), 150);
    }
    
    _showSuggestions() {
        const query = this.input.value.trim().toLowerCase();
        Q(this.suggestionsEl).empty();
        
        let suggestions = [];
        
        if (query.length === 0) {
            // Show recent tags when empty
            suggestions = this.options.recentTags
                .filter(t => !this._tags.includes(t.toLowerCase()))
                .slice(0, this.options.maxSuggestions);
        } else {
            // Filter by query - recent first, then all
            const recentMatches = this.options.recentTags
                .filter(t => t.toLowerCase().includes(query) && !this._tags.includes(t.toLowerCase()));
            const allMatches = this.options.suggestions
                .filter(t => t.toLowerCase().includes(query) && 
                            !this._tags.includes(t.toLowerCase()) && 
                            !recentMatches.map(r => r.toLowerCase()).includes(t.toLowerCase()));
            suggestions = [...recentMatches, ...allMatches].slice(0, this.options.maxSuggestions);
        }
        
        if (suggestions.length === 0) {
            this._hideSuggestions();
            return;
        }
        
        suggestions.forEach((tag, index) => {
            const item = Q('<div>', { 
                class: 'tag-suggestion-item' + (index === 0 ? ' selected' : ''),
                'data-tag': tag
            }).get(0);
            
            // Highlight matching part
            if (query.length > 0) {
                const tagLower = tag.toLowerCase();
                const idx = tagLower.indexOf(query);
                if (idx !== -1) {
                    const before = tag.substring(0, idx);
                    const match = tag.substring(idx, idx + query.length);
                    const after = tag.substring(idx + query.length);
                    item.innerHTML = `${before}<strong>${match}</strong>${after}`;
                } else {
                    item.textContent = tag;
                }
            } else {
                item.textContent = tag;
            }
            
            Q(item).on('mousedown', (e) => {
                e.preventDefault();
                this._addTagInternal(tag);
                this.input.value = '';
                this._hideSuggestions();
                this.input.focus();
            });
            
            Q(this.suggestionsEl).append(item);
        });
        
        Q(this.suggestionsEl).removeClass('hidden');
    }
    
    _hideSuggestions() {
        Q(this.suggestionsEl).addClass('hidden');
    }
    
    _navigateSuggestions(direction) {
        const items = Q(this.suggestionsEl).find('.tag-suggestion-item').getAll();
        if (items.length === 0) return;
        
        let currentIndex = -1;
        items.forEach((item, idx) => {
            if (item.classList.contains('selected')) {
                currentIndex = idx;
                Q(item).removeClass('selected');
            }
        });
        
        let newIndex = currentIndex + direction;
        if (newIndex < 0) newIndex = items.length - 1;
        if (newIndex >= items.length) newIndex = 0;
        
        Q(items[newIndex]).addClass('selected');
    }
    
    _createTagElement(tag) {
        const tagEl = Q('<span>', { class: 'tag-input-tag', 'data-tag': tag });
        
        const text = Q('<span>', { class: 'tag-input-tag-text', text: tag }).get(0);
        tagEl.append(text);
        
        const removeBtn = Q('<span>', { class: 'tag-input-tag-remove', text: '\u00D7' }).get(0);
        Q(removeBtn).on('click', (e) => {
            e.stopPropagation();
            this._removeTagInternal(tag);
        });
        tagEl.append(removeBtn);
        
        return tagEl.get();
    }
    
    _renderTags() {
        Q(this.tagsWrapper).empty();
        this._tags.forEach(tag => {
            Q(this.tagsWrapper).append(this._createTagElement(tag));
        });
    }
    
    _addTagInternal(tag) {
        const normalized = tag.trim().toLowerCase();
        if (!normalized || this._tags.includes(normalized)) return;
        
        // Check if custom tags are allowed
        if (!this.options.allowCustom) {
            const exists = this.options.suggestions.some(s => s.toLowerCase() === normalized);
            if (!exists) return;
        }
        
        this._tags.push(normalized);
        this._renderTags();
        this._notifyChange(normalized, null);
    }
    
    _removeTagInternal(tag) {
        const normalized = tag.toLowerCase();
        const index = this._tags.indexOf(normalized);
        if (index === -1) return;
        
        this._tags.splice(index, 1);
        this._renderTags();
        this._notifyChange(null, normalized);
    }
    
    _notifyChange(added, removed) {
        this._changeCallbacks.forEach(cb => cb([...this._tags], added, removed));
    }
    
    // Public API
    
    /**
     * Get current tags
     * @returns {string[]}
     */
    get() {
        return [...this._tags];
    }
    
    /**
     * Set tags (replaces all)
     * @param {string[]} tags
     * @returns {TagInput}
     */
    set(tags) {
        this._tags = (tags || []).map(t => t.trim().toLowerCase()).filter(Boolean);
        this._renderTags();
        return this;
    }
    
    /**
     * Add a single tag
     * @param {string} tag
     * @returns {TagInput}
     */
    addTag(tag) {
        this._addTagInternal(tag);
        return this;
    }
    
    /**
     * Remove a single tag
     * @param {string} tag
     * @returns {TagInput}
     */
    removeTag(tag) {
        this._removeTagInternal(tag);
        return this;
    }
    
    /**
     * Clear all tags
     * @returns {TagInput}
     */
    clear() {
        this._tags = [];
        this._renderTags();
        this._notifyChange(null, null);
        return this;
    }
    
    /**
     * Update available suggestions
     * @param {string[]} suggestions
     * @returns {TagInput}
     */
    setSuggestions(suggestions) {
        this.options.suggestions = suggestions || [];
        return this;
    }
    
    /**
     * Update recent tags
     * @param {string[]} recentTags
     * @returns {TagInput}
     */
    setRecentTags(recentTags) {
        this.options.recentTags = recentTags || [];
        return this;
    }
    
    /**
     * Register change callback
     * @param {Function} callback - Called with (tags[], addedTag, removedTag)
     * @returns {TagInput}
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
     * @returns {TagInput}
     */
    enable() {
        this.options.disabled = false;
        Q(this.element).removeClass('disabled');
        this.input.disabled = false;
        return this;
    }
    
    /**
     * Disable widget
     * @returns {TagInput}
     */
    disable() {
        this.options.disabled = true;
        Q(this.element).addClass('disabled');
        this.input.disabled = true;
        return this;
    }
    
    /**
     * Show widget
     * @returns {TagInput}
     */
    show() {
        this.options.visible = true;
        this.element.style.display = '';
        return this;
    }
    
    /**
     * Hide widget
     * @returns {TagInput}
     */
    hide() {
        this.options.visible = false;
        this.element.style.display = 'none';
        return this;
    }
    
    /**
     * Focus input
     * @returns {TagInput}
     */
    focus() {
        this.input.focus();
        return this;
    }
}
