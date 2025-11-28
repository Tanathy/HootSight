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
        
        this._element = document.createElement('div');
        this._element.className = 'widget table-widget';
        this._element.id = `table-${this.id}`;
        
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
        
        this._table = document.createElement('table');
        this._table.className = 'table';
        this._element.appendChild(this._table);
        
        this._tbody = document.createElement('tbody');
        this._table.appendChild(this._tbody);
        
        this._render();
    }
    
    setData(data) {
        this.options.data = data;
        this._render();
    }
    
    _render() {
        this._tbody.innerHTML = '';
        const data = this.options.data;
        if (data === null || typeof data === 'undefined') {
            const row = document.createElement('tr');
            row.className = 'table-row';
            const keyCell = document.createElement('td');
            keyCell.className = 'table-key';
            keyCell.textContent = '';
            const valueCell = document.createElement('td');
            valueCell.className = 'table-value';
            valueCell.textContent = this.options.emptyText;
            row.appendChild(keyCell);
            row.appendChild(valueCell);
            this._tbody.appendChild(row);
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
        const row = document.createElement('tr');
        row.className = 'table-row';
        
        const keyCell = document.createElement('td');
        keyCell.className = 'table-key';
        keyCell.textContent = key === null || typeof key === 'undefined'
            ? ''
            : key.toString();
        row.appendChild(keyCell);
        
        const valueCell = document.createElement('td');
        valueCell.className = 'table-value';
        
        if (value !== null && typeof value === 'object') {
            const nestedList = document.createElement('ul');
            nestedList.className = 'list';
            this._buildList(nestedList, value);
            valueCell.appendChild(nestedList);
        } else {
            valueCell.textContent = value === null || typeof value === 'undefined'
                ? '—'
                : value.toString();
        }
        
        row.appendChild(valueCell);
        this._tbody.appendChild(row);
    }
    
    _buildList(container, data) {
        if (Array.isArray(data)) {
            data.forEach((value, index) => {
                const li = document.createElement('li');
                li.className = 'list-item';
                const keyEl = document.createElement('span');
                keyEl.className = 'list-key';
                keyEl.textContent = index.toString();
                li.appendChild(keyEl);
                const valueEl = document.createElement('div');
                valueEl.className = 'list-value';
                if (value !== null && typeof value === 'object') {
                    const nested = document.createElement('ul');
                    nested.className = 'list';
                    this._buildList(nested, value);
                    valueEl.appendChild(nested);
                } else {
                    valueEl.textContent = value === null || typeof value === 'undefined'
                        ? '—'
                        : value.toString();
                }
                li.appendChild(valueEl);
                container.appendChild(li);
            });
        } else if (typeof data === 'object' && data !== null) {
            Object.keys(data).forEach(key => {
                const li = document.createElement('li');
                li.className = 'list-item';
                const keyEl = document.createElement('span');
                keyEl.className = 'list-key';
                keyEl.textContent = key.toString();
                li.appendChild(keyEl);
                const valueEl = document.createElement('div');
                valueEl.className = 'list-value';
                const valueData = data[key];
                if (valueData !== null && typeof valueData === 'object') {
                    const nested = document.createElement('ul');
                    nested.className = 'list';
                    this._buildList(nested, valueData);
                    valueEl.appendChild(nested);
                } else {
                    valueEl.textContent = valueData === null || typeof valueData === 'undefined'
                        ? '—'
                        : valueData.toString();
                }
                li.appendChild(valueEl);
                container.appendChild(li);
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
