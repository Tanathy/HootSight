/**
 * ActionButton widget - simple button used in header and elsewhere
 */
class ActionButton {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            label: options.label || '',
            labelLangKey: options.labelLangKey || null,
            className: options.className || 'btn btn-secondary',
            title: options.title || '',
            titleLangKey: options.titleLangKey || null,
            disabled: options.disabled || false,
            onClick: options.onClick || null
        };

        this._callbacks = [];
        this._element = document.createElement('button');
        this._element.className = this.options.className;
        this._element.id = `action-${this.id}`;
        this._element.textContent = this.options.label || '';
        
        // Add lang key for live translation
        if (this.options.labelLangKey) {
            this._element.setAttribute('data-lang-key', this.options.labelLangKey);
        }
        if (this.options.title) {
            this._element.title = this.options.title;
            if (this.options.titleLangKey) {
                this._element.setAttribute('data-lang-title', 'true');
                this._element.setAttribute('data-lang-key', this.options.titleLangKey);
            }
        }
        if (this.options.disabled) this.setDisabled(true);

        this._element.addEventListener('click', async (e) => {
            if (this.options.disabled) return;
            for (const cb of this._callbacks) {
                try {
                    const result = cb(e);
                    if (result instanceof Promise) await result;
                } catch (err) {
                    console.error('ActionButton callback error:', err);
                }
            }
            if (typeof this.options.onClick === 'function') {
                try {
                    const result = this.options.onClick(e);
                    if (result instanceof Promise) await result;
                } catch (err) {
                    console.error('ActionButton onClick error:', err);
                }
            }
        });
    }

    setLabel(label = '', langKey = null) {
        this.options.label = label;
        this._element.textContent = label;
        if (langKey) {
            this.options.labelLangKey = langKey;
            this._element.setAttribute('data-lang-key', langKey);
        }
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
