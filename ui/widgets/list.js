/**
 * List widget - renders JSON-like data as nested UL
 */
class ListWidget {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            labelLangKey: options.labelLangKey || null,
            description: options.description || '',
            descriptionLangKey: options.descriptionLangKey || null,
            data: typeof options.data !== 'undefined' ? options.data : null,
            emptyText: options.emptyText || 'No data'
        };
        
        this._element = Q('<div>', { class: 'widget list-widget', id: `list-${this.id}` }).get();
        
        if (this.options.label) {
            this._labelEl = Q('<label>', { class: 'widget-label', text: this.options.label }).get();
            if (this.options.labelLangKey) {
                this._labelEl.setAttribute('data-lang-key', this.options.labelLangKey);
            }
            Q(this._element).append(this._labelEl);
        }
        
        if (this.options.description) {
            this._descriptionEl = Q('<div>', { class: 'widget-description', text: this.options.description }).get();
            if (this.options.descriptionLangKey) {
                this._descriptionEl.setAttribute('data-lang-key', this.options.descriptionLangKey);
            }
            Q(this._element).append(this._descriptionEl);
        }
        
        this._listEl = Q('<ul>', { class: 'list' }).get();
        Q(this._element).append(this._listEl);
        
        this._render();
    }
    
    setData(data) {
        this.options.data = data;
        this._render();
    }
    
    _render() {
        Q(this._listEl).empty();
        const data = this.options.data;
        if (data === null || typeof data === 'undefined') {
            const placeholder = Q('<li>', { class: 'list-item' });
            const keyEl = Q('<span>', { class: 'list-key', text: '-' }).get();
            const valueEl = Q('<div>', { class: 'list-value', text: this.options.emptyText }).get();
            placeholder.append(keyEl).append(valueEl);
            Q(this._listEl).append(placeholder.get());
            return;
        }
        this._buildList(this._listEl, data);
    }
    
    _buildList(container, data) {
        if (Array.isArray(data)) {
            data.forEach((value, index) => {
                this._appendItem(container, index, value);
            });
        } else if (typeof data === 'object' && data !== null) {
            Object.keys(data).forEach(key => {
                this._appendItem(container, key, data[key]);
            });
        } else {
            this._appendItem(container, '-', data);
        }
    }
    
    _appendItem(container, key, value) {
        const li = Q('<li>', { class: 'list-item' });
        
        const keyText = key === null || typeof key === 'undefined' ? '-' : key.toString();
        const keyEl = Q('<span>', { class: 'list-key', text: keyText }).get();
        li.append(keyEl);
        
        const valueEl = Q('<div>', { class: 'list-value' });
        
        if (value !== null && typeof value === 'object') {
            const nestedList = Q('<ul>', { class: 'list' }).get();
            this._buildList(nestedList, value);
            valueEl.append(nestedList);
        } else {
            const text = value === null || typeof value === 'undefined' ? '—' : value.toString();
            valueEl.text(text);
        }
        
        li.append(valueEl.get());
        Q(container).append(li.get());
    }
    
    getElement() {
        return this._element;
    }
    
    static fromSchema(id, schema = {}) {
        const data = typeof schema.default !== 'undefined'
            ? schema.default
            : (Array.isArray(schema.examples) ? schema.examples[0] : null);
        return new ListWidget(id, {
            label: schema.title || id,
            description: schema.description || '',
            data
        });
    }
}
