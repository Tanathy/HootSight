/**
 * Checkbox widget - styled checkbox matching site design
 */
class Checkbox {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            checked: options.checked || false,
            disabled: options.disabled || false,
            label: options.label || '',
            labelLangKey: options.labelLangKey || null,
            onChange: options.onChange || null
        };

        this._element = Q('<label>', { class: 'checkbox-wrapper' }).get(0);
        
        this._input = Q('<input>', { 
            type: 'checkbox',
            id: `checkbox-${this.id}`,
            class: 'checkbox-input'
        }).get(0);
        
        this._input.checked = this.options.checked;
        this._input.disabled = this.options.disabled;
        
        this._checkmark = Q('<span>', { class: 'checkbox-checkmark' }).get(0);
        
        Q(this._element).append(this._input).append(this._checkmark);
        
        // Add label if provided
        if (this.options.label) {
            this._labelEl = Q('<span>', { 
                class: 'checkbox-label',
                text: this.options.label
            }).get(0);
            if (this.options.labelLangKey) {
                this._labelEl.setAttribute('data-lang-key', this.options.labelLangKey);
            }
            Q(this._element).append(this._labelEl);
        }
        
        // Event listener
        Q(this._input).on('change', (e) => {
            if (typeof this.options.onChange === 'function') {
                this.options.onChange(e.target.checked, e);
            }
        });
    }

    isChecked() {
        return this._input.checked;
    }

    setChecked(checked) {
        this._input.checked = !!checked;
        return this;
    }

    setDisabled(disabled) {
        this._input.disabled = !!disabled;
        Q(this._element).toggleClass('disabled', !!disabled);
        return this;
    }

    setIndeterminate(indeterminate) {
        this._input.indeterminate = !!indeterminate;
        return this;
    }

    getElement() {
        return this._element;
    }

    getInput() {
        return this._input;
    }
}
