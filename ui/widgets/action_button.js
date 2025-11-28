/**
 * ActionButton widget - simple button used in header and elsewhere
 */
class ActionButton {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            className: options.className || 'btn btn-secondary',
            title: options.title || '',
            disabled: options.disabled || false,
            onClick: options.onClick || null
        };

        this._callbacks = [];
        this._element = document.createElement('button');
        this._element.className = this.options.className;
        this._element.id = `action-${this.id}`;
        this._element.textContent = this.options.label || '';
        if (this.options.title) this._element.title = this.options.title;
        if (this.options.disabled) this.setDisabled(true);

        this._element.addEventListener('click', (e) => {
            if (this.options.disabled) return;
            for (const cb of this._callbacks) cb(e);
            if (typeof this.options.onClick === 'function') {
                try { this.options.onClick(e); } catch (err) { console.error(err); }
            }
        });
    }

    setLabel(label = '') {
        this.options.label = label;
        this._element.textContent = label;
    }

    setDisabled(disabled = true) {
        this.options.disabled = !!disabled;
        this._element.disabled = !!disabled;
        this._element.classList.toggle('disabled', !!disabled);
    }

    onClick(cb) {
        if (typeof cb === 'function') {
            this._callbacks.push(cb);
        }
        return this;
    }

    getElement() {
        return this._element;
    }

    get() {
        return this.options;
    }

    static fromSchema(id, schema = {}) {
        const ui = schema.ui || {};
        return new ActionButton(id, {
            label: schema.title || ui.title || id,
            className: ui.className || 'btn btn-secondary',
            title: schema.description || ui.description || '',
            disabled: ui.disabled || false
        });
    }
}
