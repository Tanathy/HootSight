/**
 * Heading widget - renders title + description pair
 */
class Heading {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            title: options.title || '',
            description: options.description || '',
            tag: options.tag || 'h2'
        };
        
        this._element = document.createElement('div');
        this._element.className = 'heading';
        this._element.id = `heading-${this.id}`;
        
        this._titleEl = document.createElement(this._resolveTag(this.options.tag));
        this._titleEl.className = 'heading-title';
        
        this._descriptionEl = document.createElement('div');
        this._descriptionEl.className = 'heading-description';
        
        this._element.appendChild(this._titleEl);
        this._element.appendChild(this._descriptionEl);
        
        this.setTitle(this.options.title);
        this.setDescription(this.options.description);
    }
    
    _resolveTag(tag) {
        const allowed = ['h1', 'h2', 'h3', 'h4'];
        return allowed.includes(tag) ? tag : 'h2';
    }
    
    _updateVisibility() {
        const hasTitle = this._titleEl.textContent.trim().length > 0;
        const hasDescription = this._descriptionEl.textContent.trim().length > 0;
        this._titleEl.style.display = hasTitle ? '' : 'none';
        this._descriptionEl.style.display = hasDescription ? '' : 'none';
        this._element.style.display = hasTitle || hasDescription ? '' : 'none';
    }
    
    setTitle(title = '') {
        this.options.title = title || '';
        this._titleEl.textContent = this.options.title;
        this._updateVisibility();
    }
    
    setDescription(description = '') {
        this.options.description = description || '';
        this._descriptionEl.textContent = this.options.description;
        this._updateVisibility();
    }
    
    setData({ title, description } = {}) {
        if (typeof title !== 'undefined') {
            this.setTitle(title);
        }
        if (typeof description !== 'undefined') {
            this.setDescription(description);
        }
    }
    
    getElement() {
        return this._element;
    }
    
    static fromSchema(id, schema = {}) {
        const ui = schema.ui || {};
        return new Heading(id, {
            title: schema.title || ui.title || '',
            description: schema.description || ui.description || '',
            tag: ui.heading_tag || 'h2'
        });
    }
}
