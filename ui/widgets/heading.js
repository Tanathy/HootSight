/**
 * Heading widget - renders title + description pair
 */
class Heading {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            title: options.title || '',
            description: options.description || '',
            titleLangKey: options.titleLangKey || null,
            descriptionLangKey: options.descriptionLangKey || null,
            tag: options.tag || 'h2'
        };
        
        this._element = Q('<div>', { class: 'heading', id: `heading-${this.id}` }).get();
        
        const resolvedTag = this._resolveTag(this.options.tag);
        const titleAttrs = { class: 'heading-title' };
        if (this.options.titleLangKey) {
            titleAttrs['data-lang-key'] = this.options.titleLangKey;
        }
        this._titleEl = Q(`<${resolvedTag}>`, titleAttrs).get();
        
        const descAttrs = { class: 'heading-description' };
        if (this.options.descriptionLangKey) {
            descAttrs['data-lang-key'] = this.options.descriptionLangKey;
        }
        this._descriptionEl = Q('<div>', descAttrs).get();
        
        Q(this._element).append(this._titleEl).append(this._descriptionEl);
        
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
        Q(this._titleEl).css('display', hasTitle ? '' : 'none');
        Q(this._descriptionEl).css('display', hasDescription ? '' : 'none');
        Q(this._element).css('display', hasTitle || hasDescription ? '' : 'none');
    }
    
    setTitle(title = '') {
        this.options.title = title || '';
        Q(this._titleEl).text(this.options.title);
        this._updateVisibility();
    }
    
    setDescription(description = '') {
        this.options.description = description || '';
        Q(this._descriptionEl).text(this.options.description);
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
