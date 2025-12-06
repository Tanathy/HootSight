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
            icon: options.icon || null,
            disabled: options.disabled || false,
            onClick: options.onClick || null
        };

        this._callbacks = [];
        
        const attrs = {
            class: this.options.className,
            id: `action-${this.id}`
        };
        
        if (this.options.labelLangKey) {
            attrs['data-lang-key'] = this.options.labelLangKey;
        }
        if (this.options.title) {
            attrs.title = this.options.title;
            if (this.options.titleLangKey) {
                attrs['data-lang-title'] = 'true';
                attrs['data-lang-key'] = this.options.titleLangKey;
            }
        }
        
        this._element = Q('<button>', attrs).get();
        
        // Add icon if specified
        if (this.options.icon) {
            const iconEl = Q('<img>', { 
                src: this.options.icon, 
                alt: '', 
                class: 'btn-icon-img' 
            }).get(0);
            Q(this._element).append(iconEl);
        }
        
        // Add label text (after icon if present)
        if (this.options.label) {
            if (this.options.icon) {
                // If icon exists, wrap text in span for proper styling
                const textSpan = Q('<span>', { text: this.options.label }).get(0);
                Q(this._element).append(textSpan);
            } else {
                Q(this._element).text(this.options.label);
            }
        }
        
        if (this.options.disabled) this.setDisabled(true);

        Q(this._element).on('click', async (e) => {
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
        // If no icon, just set text. If icon exists, find the span
        if (!this.options.icon) {
            Q(this._element).text(label);
        } else {
            const textSpan = Q(this._element).find('span').get(0);
            if (textSpan) {
                Q(textSpan).text(label);
            }
        }
        if (langKey) {
            this.options.labelLangKey = langKey;
            Q(this._element).attr('data-lang-key', langKey);
        }
    }

    setDisabled(disabled = true) {
        this.options.disabled = !!disabled;
        Q(this._element).prop('disabled', !!disabled).toggleClass('disabled', !!disabled);
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
            icon: ui.icon || null,
            disabled: ui.disabled || false
        });
    }
}
