/**
 * HeaderActions widget - container for action buttons in the header
 */
class HeaderActions {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            actions: options.actions || []
        };

        this._element = document.createElement('div');
        this._element.className = 'header-actions';
        this._element.id = `header-actions-${this.id}`;
        this._actions = [];

        if (this.options.actions && this.options.actions.length) {
            this.options.actions.forEach(a => this.addAction(a));
        }
    }

    addAction(actionConfig) {
        let action = null;
        if (actionConfig instanceof ActionButton) {
            action = actionConfig;
        } else if (typeof actionConfig === 'object') {
            const id = `${this.id}-action-${this._actions.length + 1}`;
            action = new ActionButton(id, {
                label: actionConfig.label || actionConfig.title || 'Action',
                className: actionConfig.className || actionConfig.btnClass || 'btn btn-secondary',
                title: actionConfig.title || actionConfig.description || '' ,
                disabled: !!actionConfig.disabled,
                onClick: actionConfig.onClick || null
            });
        }
        if (action) {
            this._actions.push(action);
            this._element.appendChild(action.getElement());
        }
        return action;
    }

    clear() {
        this._actions = [];
        this._element.innerHTML = '';
    }

    getElement() {
        return this._element;
    }

    static fromSchema(id, schema = {}) {
        const ui = schema.ui || {};
        const actions = ui.actions || [];
        return new HeaderActions(id, { actions });
    }
}
