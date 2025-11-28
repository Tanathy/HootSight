class PaginaBox {
    constructor(identifier, config = {}) {
        this.id = identifier;
        this.assign = config.assign || null;
        this.element = null;
        this.contentWrapper = null;
        this.contentElement = null;
        this.loadingElement = null;
        this.emptyElement = null;
        this.errorElement = null;
        this.pageSizeSelects = [];
        this.pageButtonContainers = [];
        this.summaryLabels = [];
        this.controlRows = [];
        this.hooks = {};
        this.renderItem = null;
        this.items = [];
        this.loading = false;
        this.error = null;

        this.config = this._normalizeConfig(config);
        this.state = {
            page: this.config.defaultPage,
            pageSize: this.config.defaultPageSize,
            totalItems: 0,
            totalPages: 1
        };

        this._build();

        if (window.ModuleAPI && ModuleAPI.paginate && typeof ModuleAPI.paginate._attachComponent === 'function') {
            ModuleAPI.paginate._attachComponent(this.assign, this);
        }
    }

    _normalizeConfig(config) {
        const pageSizeOptions = this._normalizePageSizes(config.pageSizeOptions, config.defaultPageSize);
        const defaultPageSize = this._resolveDefaultPageSize(config.defaultPageSize, pageSizeOptions);
        const controlsPosition = (config.controlsPosition || 'bottom').toString().toLowerCase();
        const controlsAlign = (config.controlsAlign || 'right').toString().toLowerCase();
        return {
            assign: config.assign || null,
            controlsPosition,
            controlsAlign: ['left', 'center', 'right'].includes(controlsAlign) ? controlsAlign : 'right',
            maxButtons: Math.max(parseInt(config.maxButtons, 10) || 7, 3),
            autoLoading: config.autoLoading !== false,
            showPageSize: config.showPageSize !== false,
            pageSizeOptions,
            defaultPageSize,
            defaultPage: Math.max(parseInt(config.defaultPage, 10) || 1, 1),
            contentClass: config.contentClass || '',
            emptyState: config.emptyState || null,
            action: config.action || null,
            pageAction: config.pageAction || null,
            pageSizeAction: config.pageSizeAction || null
        };
    }

    _normalizePageSizes(rawOptions, fallback) {
        const list = Array.isArray(rawOptions) ? rawOptions : [];
        const normalized = list
            .map((value) => parseInt(value, 10))
            .filter((value) => Number.isFinite(value) && value > 0);
        if (fallback) {
            const n = parseInt(fallback, 10);
            if (Number.isFinite(n) && n > 0) {
                normalized.push(n);
            }
        }
        if (!normalized.length) {
            return [25, 50, 100];
        }
        const unique = Array.from(new Set(normalized));
        unique.sort((a, b) => a - b);
        return unique;
    }

    _resolveDefaultPageSize(requested, options) {
        const parsed = parseInt(requested, 10);
        if (Number.isFinite(parsed) && parsed > 0) {
            return parsed;
        }
        if (options && options.length) {
            return options[0];
        }
        return 25;
    }

    _build() {
        this.element = Q('<div>', { class: 'paginabox' }).get(0);
        this.element.id = this.id;

        this.contentWrapper = Q('<div>', { class: 'paginabox-content-wrapper' }).get(0);
        this.contentElement = Q('<div>', { class: 'paginabox-content' }).get(0);
        this.contentElement.id = `${this.id}-content`;
        if (this.config.contentClass) {
            Q(this.contentElement).addClass(this.config.contentClass);
        }
        this.contentWrapper.appendChild(this.contentElement);

        this.loadingElement = Q('<div>', { class: 'paginabox-loading', text: lang('ui.paginabox.loading') }).get(0);
        this.emptyElement = Q('<div>', { class: 'paginabox-empty' }).get(0);
        this.errorElement = Q('<div>', { class: 'paginabox-error' }).get(0);

        const positions = this._resolveControlPositions();
        positions.forEach((position) => {
            const row = this._createControlRow(position);
            this.controlRows.push(row);
        });

        this.element.appendChild(this.contentWrapper);

        this.render();
    }

    _resolveControlPositions() {
        const positions = [];
        switch (this.config.controlsPosition) {
            case 'top':
                positions.push('top');
                break;
            case 'both':
                positions.push('top', 'bottom');
                break;
            case 'none':
            case 'hidden':
                break;
            default:
                positions.push('bottom');
        }
        return positions;
    }

    _createControlRow(position) {
        const wrapper = Q('<div>', {
            class: `paginabox-controls align-${this.config.controlsAlign}`
        }).get(0);
        const left = Q('<div>', { class: 'paginabox-controls-left' }).get(0);
        const right = Q('<div>', { class: 'paginabox-controls-right' }).get(0);

        wrapper.appendChild(left);
        wrapper.appendChild(right);

        if (position === 'top') {
            this.element.insertBefore(wrapper, this.contentWrapper);
        } else {
            this.element.appendChild(wrapper);
        }

        if (this.config.showPageSize && this.config.pageSizeOptions.length > 1) {
            const pageSizeControls = this._createPageSizeSelector();
            left.appendChild(pageSizeControls.container);
            this.pageSizeSelects.push(pageSizeControls.select);
        }

        const navControls = this._createNavControls();
        right.appendChild(navControls.container);
        this.pageButtonContainers.push(navControls.buttonsContainer);
        this.summaryLabels.push(navControls.summaryLabel);

        return { wrapper, left, right };
    }

    _createPageSizeSelector() {
        const labelText = lang('ui.paginabox.items_per_page');
        const container = Q('<label>', { class: 'paginabox-pagesize' }).get(0);
        const label = Q('<span>', { class: 'paginabox-pagesize-label', text: labelText }).get(0);
        const select = Q('<select>', { class: 'paginabox-pagesize-select' }).get(0);

        this.config.pageSizeOptions.forEach((value) => {
            const option = Q('<option>', { value: value, text: String(value) }).get(0);
            select.appendChild(option);
        });

        Q(select).on('change', (event) => {
            const next = parseInt(event.target.value, 10);
            if (Number.isFinite(next) && next > 0) {
                this._changePageSize(next);
            }
        });

        container.appendChild(label);
        container.appendChild(select);
        return { container, select };
    }

    _createNavControls() {
        const container = Q('<div>', { class: 'paginabox-nav' }).get(0);
        const summaryLabel = Q('<span>', { class: 'paginabox-summary' }).get(0);
        const buttonsContainer = Q('<div>', { class: 'paginabox-buttons' }).get(0);
        container.appendChild(summaryLabel);
        container.appendChild(buttonsContainer);
        return { container, summaryLabel, buttonsContainer };
    }

    getElement() {
        return this.element;
    }

    get() {
        return { ...this.state };
    }

    set(value) {
        if (!value || typeof value !== 'object') {
            return;
        }

        if (value.contentClass) {
            Q(this.contentElement).removeClass(this.config.contentClass);
            this.config.contentClass = value.contentClass;
            Q(this.contentElement).addClass(this.config.contentClass);
        }

        if (Array.isArray(value.pageSizeOptions) && value.pageSizeOptions.length) {
            this.config.pageSizeOptions = this._normalizePageSizes(value.pageSizeOptions, value.defaultPageSize);
            this._rebuildPageSizeOptions();
        }

        if (value.defaultPageSize) {
            this.config.defaultPageSize = this._resolveDefaultPageSize(value.defaultPageSize, this.config.pageSizeOptions);
            if (!this.config.pageSizeOptions.includes(this.config.defaultPageSize)) {
                this.config.pageSizeOptions.push(this.config.defaultPageSize);
                this.config.pageSizeOptions.sort((a, b) => a - b);
                this._rebuildPageSizeOptions();
            }
        }

        if (value.pageSize && Number.isFinite(parseInt(value.pageSize, 10))) {
            const nextSize = Math.max(parseInt(value.pageSize, 10), 1);
            this.state.pageSize = nextSize;
        }

        if (value.page && Number.isFinite(parseInt(value.page, 10))) {
            this.state.page = Math.max(parseInt(value.page, 10), 1);
        }

        if (Number.isFinite(value.totalItems)) {
            this.state.totalItems = Math.max(parseInt(value.totalItems, 10), 0);
        }

        if (Number.isFinite(value.totalPages)) {
            this.state.totalPages = Math.max(parseInt(value.totalPages, 10), 1);
        } else if (this.state.totalItems && this.state.pageSize) {
            this.state.totalPages = Math.max(1, Math.ceil(this.state.totalItems / this.state.pageSize));
        }

        if (Array.isArray(value.items)) {
            this.items = value.items;
        }

        if (typeof value.renderItem === 'function') {
            this.renderItem = value.renderItem;
        }

        if (typeof value.emptyState === 'string') {
            this.config.emptyState = value.emptyState;
        }

        if (value.error !== undefined) {
            this._setError(value.error);
        }

        if (typeof value.loading === 'boolean') {
            this.loading = value.loading;
        }

        this.render();
    }

    render() {
        const totalPages = Math.max(this.state.totalPages || 1, 1);
        if (this.state.page > totalPages) {
            this.state.page = totalPages;
        }
        this._updateControls();
        this._updateContent();
    }

    attachHooks(hooks = {}) {
        if (!hooks || typeof hooks !== 'object') {
            return;
        }
        this.hooks = { ...this.hooks, ...hooks };
    }

    setLoading(flag) {
        this.loading = !!flag;
        this.render();
    }

    destroy() {
        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
        this.element = null;
        if (window.ModuleAPI && ModuleAPI.paginate && typeof ModuleAPI.paginate._detachComponent === 'function') {
            ModuleAPI.paginate._detachComponent(this.assign, this);
        }
    }

    _updateControls() {
        const totalPages = Math.max(this.state.totalPages || 1, 1);
        const currentPage = Math.min(Math.max(this.state.page || 1, 1), totalPages);
        const summaryText = lang('ui.paginabox.page_summary', {
            current: currentPage,
            total: totalPages
        });

        this.summaryLabels.forEach((label) => {
            Q(label).text(summaryText);
        });

        this.pageSizeSelects.forEach((select) => {
            select.value = String(this.state.pageSize);
        });

        this.pageButtonContainers.forEach((container) => {
            this._buildPageButtons(container, currentPage, totalPages);
        });
    }

    _buildPageButtons(container, currentPage, totalPages) {
        Q(container).empty();

        const addButton = (labelKey, targetPage, type, disabled) => {
            const label = lang(labelKey);
            const button = Q('<button>', {
                class: `paginabox-button paginabox-${type}`,
                type: 'button',
                text: label
            }).get(0);
            if (disabled) {
                button.disabled = true;
                Q(button).addClass('disabled');
            } else {
                Q(button).on('click', () => this._gotoPage(targetPage, type));
            }
            container.appendChild(button);
        };

        addButton('ui.paginabox.first', 1, 'first', currentPage <= 1);
        addButton('ui.paginabox.previous', currentPage - 1, 'prev', currentPage <= 1);

        const pageNumbers = this._computePageNumbers(currentPage, totalPages);
        pageNumbers.forEach((pageNumber) => {
            const button = Q('<button>', {
                class: 'paginabox-button paginabox-page',
                type: 'button',
                text: String(pageNumber)
            }).get(0);
            if (pageNumber === currentPage) {
                Q(button).addClass('active');
                button.disabled = true;
            } else {
                Q(button).on('click', () => this._gotoPage(pageNumber, 'page'));
            }
            container.appendChild(button);
        });

        addButton('ui.paginabox.next', currentPage + 1, 'next', currentPage >= totalPages);
        addButton('ui.paginabox.last', totalPages, 'last', currentPage >= totalPages);
    }

    _computePageNumbers(currentPage, totalPages) {
        const maxButtons = Math.max(this.config.maxButtons, 3);
        if (totalPages <= maxButtons) {
            return Array.from({ length: totalPages }, (_, index) => index + 1);
        }
        const half = Math.floor(maxButtons / 2);
        let start = currentPage - half;
        let end = currentPage + half;
        if (maxButtons % 2 === 0) {
            end -= 1;
        }
        if (start < 1) {
            start = 1;
            end = maxButtons;
        }
        if (end > totalPages) {
            end = totalPages;
            start = totalPages - maxButtons + 1;
        }
        const pages = [];
        for (let page = start; page <= end; page += 1) {
            pages.push(page);
        }
        return pages;
    }

    _updateContent() {
        Q(this.contentElement).empty();

        if (this.loading) {
            this.contentElement.appendChild(this.loadingElement);
            return;
        }

        if (this.error) {
            this.errorElement.textContent = this.error;
            this.contentElement.appendChild(this.errorElement);
            return;
        }

        if (!this.items || !this.items.length) {
            this.emptyElement.textContent = this._resolveEmptyText();
            this.contentElement.appendChild(this.emptyElement);
            return;
        }

        // Client-side pagination: calculate which items to show on current page
        const startIndex = (this.state.page - 1) * this.state.pageSize;
        const endIndex = startIndex + this.state.pageSize;
        const pageItems = this.items.slice(startIndex, endIndex);

        // Update total pages based on actual items count
        const calculatedTotalPages = Math.max(1, Math.ceil(this.items.length / this.state.pageSize));
        if (this.state.totalPages !== calculatedTotalPages) {
            this.state.totalPages = calculatedTotalPages;
            this.state.totalItems = this.items.length;
            this._updateControls();
        }

        // Render only the items for the current page
        pageItems.forEach((item, index) => {
            const globalIndex = startIndex + index;
            const node = this._resolveItemNode(item, globalIndex);
            this._appendNode(node);
        });
    }

    _resolveEmptyText() {
        if (this.config.emptyState) {
            try {
                return lang(this.config.emptyState);
            } catch (error) {
                return this.config.emptyState;
            }
        }
        return lang('ui.paginabox.empty');
    }

    _resolveItemNode(item, index) {
        if (this.renderItem && typeof this.renderItem === 'function') {
            try {
                const rendered = this.renderItem(item, index);
                if (rendered) {
                    return rendered;
                }
            } catch (error) {
                console.error('PaginaBox renderItem failed:', error);
            }
        }
        if (item instanceof Element || item instanceof Text || item instanceof DocumentFragment) {
            return item;
        }
        if (Array.isArray(item)) {
            return item;
        }
        if (typeof item === 'string') {
            const wrapper = Q('<div>', { class: 'paginabox-item-string', text: item }).get(0);
            return wrapper;
        }
        try {
            const wrapper = Q('<pre>', { class: 'paginabox-item-json' }).get(0);
            wrapper.textContent = JSON.stringify(item, null, 2);
            return wrapper;
        } catch (error) {
            const fallback = Q('<div>', { class: 'paginabox-item-string', text: String(item) }).get(0);
            return fallback;
        }
    }

    _appendNode(node) {
        if (!node) {
            return;
        }
        if (Array.isArray(node)) {
            node.forEach((child) => this._appendNode(child));
            return;
        }
        if (node instanceof DocumentFragment) {
            this.contentElement.appendChild(node);
            return;
        }
        if (node instanceof Element || node instanceof Text) {
            if (node.parentNode && node.parentNode !== this.contentElement) {
                node.parentNode.removeChild(node);
            }
            this.contentElement.appendChild(node);
            return;
        }
        if (typeof node === 'string') {
            const wrapper = document.createElement('div');
            wrapper.className = 'paginabox-item-string';
            wrapper.textContent = node;
            this.contentElement.appendChild(wrapper);
        }
    }

    _gotoPage(page, reason) {
        const totalPages = Math.max(this.state.totalPages || 1, 1);
        const target = Math.min(Math.max(page, 1), totalPages);
        if (target === this.state.page) {
            return;
        }
        this.state.page = target;
        
        // For client-side pagination: just re-render with new page
        // For server-side pagination: set loading if autoLoading is enabled
        if (this.config.autoLoading) {
            this.setLoading(true);
        } else {
            this.render();
        }
        
        this._emitChange('pageChange', {
            page: this.state.page,
            pageSize: this.state.pageSize,
            reason,
            assign: this.assign
        });
    }

    _changePageSize(pageSize) {
        if (pageSize === this.state.pageSize) {
            return;
        }
        this.state.pageSize = pageSize;
        this.state.page = 1;
        
        // For client-side pagination: recalculate total pages and render
        // For server-side pagination: set loading if autoLoading is enabled
        if (this.config.autoLoading) {
            this.setLoading(true);
        } else {
            // Recalculate total pages with new page size
            if (this.items && this.items.length) {
                this.state.totalPages = Math.max(1, Math.ceil(this.items.length / this.state.pageSize));
            }
            this.render();
        }
        
        this._emitChange('pageSizeChange', {
            page: this.state.page,
            pageSize: this.state.pageSize,
            reason: 'pageSize',
            assign: this.assign
        });
    }

    _emitChange(event, payload) {
        if (window.ModuleAPI && ModuleAPI.paginate && typeof ModuleAPI.paginate.notify === 'function') {
            ModuleAPI.paginate.notify(this.assign, event, payload);
        }

        const handlerMap = {
            pageChange: this.hooks.onPageChange,
            pageSizeChange: this.hooks.onPageSizeChange
        };
        const fallback = this.hooks.onChange;

        const handlers = [];
        const primary = handlerMap[event];
        if (typeof primary === 'function') {
            handlers.push(primary);
        }
        if (typeof fallback === 'function') {
            handlers.push(fallback);
        }
        handlers.forEach((handler) => {
            try {
                handler(payload);
            } catch (error) {
                console.error('PaginaBox hook failed:', error);
            }
        });

        const actionName = event === 'pageChange'
            ? (this.config.pageAction || this.config.action)
            : (this.config.pageSizeAction || this.config.action);
        if (actionName) {
            this._invokeAction(actionName, payload);
        }
    }

    _invokeAction(actionName, payload) {
        try {
            let target = null;
            if (actionName.includes('.')) {
                const parts = actionName.split('.');
                target = window;
                for (const part of parts) {
                    if (target && typeof target === 'object' && part in target) {
                        target = target[part];
                    } else {
                        target = null;
                        break;
                    }
                }
            } else {
                target = window[actionName];
            }
            if (typeof target === 'function') {
                target(payload);
                return;
            }
            if (window.COMPONENTS_BUILDER && typeof window.COMPONENTS_BUILDER.executeAction === 'function') {
                window.COMPONENTS_BUILDER.executeAction(actionName, {
                    id: this.id,
                    assign: this.assign,
                    payload
                });
            } else {
                console.warn('PaginaBox action not found:', actionName);
            }
        } catch (error) {
            console.error('PaginaBox action failed:', actionName, error);
        }
    }

    _rebuildPageSizeOptions() {
        this.pageSizeSelects.forEach((select) => {
            Q(select).empty();
            this.config.pageSizeOptions.forEach((value) => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = String(value);
                select.appendChild(option);
            });
            select.value = String(this.state.pageSize);
        });
    }

    _setError(error) {
        if (!error) {
            this.error = null;
            return;
        }
        if (typeof error === 'string') {
            this.error = error;
            return;
        }
        if (error && typeof error === 'object' && error.message) {
            this.error = error.message;
            return;
        }
        try {
            this.error = JSON.stringify(error);
        } catch (_) {
            this.error = String(error);
        }
    }
}

function createPaginaBox(identifier, config) {
    return new PaginaBox(identifier, config);
}
