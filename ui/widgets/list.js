/**
 * List widget - renders JSON-like data as nested UL
 */
class ListWidget {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            description: options.description || '',
            data: typeof options.data !== 'undefined' ? options.data : null,
            emptyText: options.emptyText || 'No data'
        };
        
        this._element = document.createElement('div');
        this._element.className = 'widget list-widget';
        this._element.id = `list-${this.id}`;
        
        if (this.options.label) {
            this._labelEl = document.createElement('label');
            this._labelEl.className = 'widget-label';
            this._labelEl.textContent = this.options.label;
            this._element.appendChild(this._labelEl);
        }
        
        if (this.options.description) {
            this._descriptionEl = document.createElement('div');
            this._descriptionEl.className = 'widget-description';
            this._descriptionEl.textContent = this.options.description;
            this._element.appendChild(this._descriptionEl);
        }
        
        this._listEl = document.createElement('ul');
        this._listEl.className = 'list';
        this._element.appendChild(this._listEl);
        
        this._render();
    }
    
    setData(data) {
        this.options.data = data;
        this._render();
    }
    
    _render() {
        this._listEl.innerHTML = '';
        const data = this.options.data;
        if (data === null || typeof data === 'undefined') {
            const placeholder = document.createElement('li');
            placeholder.className = 'list-item';
            const keyEl = document.createElement('span');
            keyEl.className = 'list-key';
            keyEl.textContent = '-';
            const valueEl = document.createElement('div');
            valueEl.className = 'list-value';
            valueEl.textContent = this.options.emptyText;
            placeholder.appendChild(keyEl);
            placeholder.appendChild(valueEl);
            this._listEl.appendChild(placeholder);
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
        const li = document.createElement('li');
        li.className = 'list-item';
        
        const keyEl = document.createElement('span');
        keyEl.className = 'list-key';
        keyEl.textContent = key === null || typeof key === 'undefined'
            ? '-'
            : key.toString();
        li.appendChild(keyEl);
        
        const valueEl = document.createElement('div');
        valueEl.className = 'list-value';
        
        if (value !== null && typeof value === 'object') {
            const nestedList = document.createElement('ul');
            nestedList.className = 'list';
            this._buildList(nestedList, value);
            valueEl.appendChild(nestedList);
        } else {
            valueEl.textContent = value === null || typeof value === 'undefined'
                ? '—'
                : value.toString();
        }
        
        li.appendChild(valueEl);
        container.appendChild(li);
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
