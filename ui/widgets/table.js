/**
 * Table widget - simple key:value table renderer
 */
class TableWidget {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            description: options.description || '',
            data: typeof options.data !== 'undefined' ? options.data : null,
            emptyText: options.emptyText || 'No rows'
        };
        
        this._element = Q('<div>', { class: 'widget table-widget', id: `table-${this.id}` }).get();
        
        if (this.options.label) {
            this._labelEl = Q('<label>', { class: 'widget-label', text: this.options.label }).get();
            Q(this._element).append(this._labelEl);
        }
        
        if (this.options.description) {
            this._descriptionEl = Q('<div>', { class: 'widget-description', text: this.options.description }).get();
            Q(this._element).append(this._descriptionEl);
        }
        
        this._table = Q('<table>', { class: 'table' }).get();
        Q(this._element).append(this._table);
        
        this._tbody = Q('<tbody>').get();
        Q(this._table).append(this._tbody);
        
        this._render();
    }
    
    setData(data) {
        this.options.data = data;
        this._render();
    }
    
    _render() {
        Q(this._tbody).empty();
        const data = this.options.data;
        if (data === null || typeof data === 'undefined') {
            const row = Q('<tr>', { class: 'table-row' });
            const keyCell = Q('<td>', { class: 'table-key', text: '' }).get();
            const valueCell = Q('<td>', { class: 'table-value', text: this.options.emptyText }).get();
            row.append(keyCell).append(valueCell);
            Q(this._tbody).append(row.get());
            return;
        }
        if (Array.isArray(data)) {
            data.forEach((value, index) => this._appendRow(index, value));
        } else if (typeof data === 'object') {
            Object.keys(data).forEach(key => this._appendRow(key, data[key]));
        } else {
            this._appendRow('-', data);
        }
    }
    
    _appendRow(key, value) {
        const row = Q('<tr>', { class: 'table-row' });
        
        const keyText = key === null || typeof key === 'undefined' ? '' : key.toString();
        const keyCell = Q('<td>', { class: 'table-key', text: keyText }).get();
        row.append(keyCell);
        
        const valueCell = Q('<td>', { class: 'table-value' }).get();
        
        if (value !== null && typeof value === 'object') {
            const nestedList = Q('<ul>', { class: 'list' }).get();
            this._buildList(nestedList, value);
            Q(valueCell).append(nestedList);
        } else {
            const text = value === null || typeof value === 'undefined' ? '—' : value.toString();
            Q(valueCell).text(text);
        }
        
        row.append(valueCell);
        Q(this._tbody).append(row.get());
    }
    
    _buildList(container, data) {
        if (Array.isArray(data)) {
            data.forEach((value, index) => {
                const li = Q('<li>', { class: 'list-item' });
                const keyEl = Q('<span>', { class: 'list-key', text: index.toString() }).get();
                li.append(keyEl);
                const valueEl = Q('<div>', { class: 'list-value' });
                if (value !== null && typeof value === 'object') {
                    const nested = Q('<ul>', { class: 'list' }).get();
                    this._buildList(nested, value);
                    valueEl.append(nested);
                } else {
                    const text = value === null || typeof value === 'undefined' ? '—' : value.toString();
                    valueEl.text(text);
                }
                li.append(valueEl.get());
                Q(container).append(li.get());
            });
        } else if (typeof data === 'object' && data !== null) {
            Object.keys(data).forEach(key => {
                const li = Q('<li>', { class: 'list-item' });
                const keyEl = Q('<span>', { class: 'list-key', text: key.toString() }).get();
                li.append(keyEl);
                const valueEl = Q('<div>', { class: 'list-value' });
                const valueData = data[key];
                if (valueData !== null && typeof valueData === 'object') {
                    const nested = Q('<ul>', { class: 'list' }).get();
                    this._buildList(nested, valueData);
                    valueEl.append(nested);
                } else {
                    const text = valueData === null || typeof valueData === 'undefined' ? '—' : valueData.toString();
                    valueEl.text(text);
                }
                li.append(valueEl.get());
                Q(container).append(li.get());
            });
        }
    }
    
    getElement() {
        return this._element;
    }
    
    static fromSchema(id, schema = {}) {
        const data = typeof schema.default !== 'undefined'
            ? schema.default
            : (Array.isArray(schema.examples) ? schema.examples[0] : null);
        return new TableWidget(id, {
            label: schema.title || id,
            description: schema.description || '',
            data
        });
    }
}
