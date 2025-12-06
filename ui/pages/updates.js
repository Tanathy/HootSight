/**
 * HootSight - Updates Page
 * System update checker and applier
 */

const UpdatesPage = {
    name: 'updates',

    /**
     * State
     */
    _state: 'idle', // idle, checking, ready, applying, error
    _updateData: null,
    _filePaths: [],

    /**
     * UI References
     */
    _statusEl: null,
    _checkBtn: null,
    _applyBtn: null,
    _tableContainer: null,
    _summaryEl: null,

    /**
     * Build the Updates page
     * @param {HTMLElement} container
     */
    build: function(container) {
        Q(container).empty();
        this._state = 'idle';
        this._updateData = null;
        this._filePaths = [];
        this._summaryEl = null;

        // Page heading
        const heading = new Heading('updates-heading', {
            title: lang('updates_ui.page_title'),
            titleLangKey: 'updates_ui.page_title',
            description: lang('updates_ui.page_description'),
            descriptionLangKey: 'updates_ui.page_description'
        });
        Q(container).append(heading.getElement());

        // Main card
        const card = Q('<div>', { class: 'updates-card' }).get(0);

        // Intro text
        const intro = Q('<p>', { 
            class: 'updates-intro',
            text: lang('updates_ui.intro')
        }).get(0);
        intro.setAttribute('data-lang-key', 'updates_ui.intro');
        Q(card).append(intro);

        // Status area
        this._statusEl = Q('<div>', { class: 'updates-status' }).get(0);
        this._updateStatusUI();
        Q(card).append(this._statusEl);

        // Table container for update list
        this._tableContainer = Q('<div>', { class: 'updates-table-container' }).get(0);
        Q(card).append(this._tableContainer);

        Q(container).append(card);
    },

    /**
     * Setup header action buttons (called by app.js after page build)
     */
    setupHeaderActions: function() {
        HeaderActions.clear().add([
            {
                id: 'check-updates',
                label: lang('updates_ui.check_button'),
                labelLangKey: 'updates_ui.check_button',
                type: 'primary',
                onClick: () => this._checkForUpdates()
            },
            {
                id: 'apply-updates',
                label: lang('updates_ui.apply_button'),
                labelLangKey: 'updates_ui.apply_button',
                type: 'success',
                disabled: true,
                onClick: () => this._applyUpdates()
            }
        ]);

        // Store references for later updates
        this._checkBtn = HeaderActions.get('check-updates');
        this._applyBtn = HeaderActions.get('apply-updates');
    },

    /**
     * Update the status UI based on current state
     */
    _updateStatusUI: function() {
        if (!this._statusEl) return;

        const statusEl = Q(this._statusEl);
        const stateClasses = ['status-idle', 'status-checking', 'status-ready', 'status-up-to-date', 'status-applying', 'status-applied', 'status-error'];
        let stateClass = '';
        let statusText = '';
        let statusKey = '';

        switch (this._state) {
            case 'idle':
                stateClass = 'status-idle';
                statusText = lang('updates_ui.status_idle');
                statusKey = 'updates_ui.status_idle';
                break;
            case 'checking':
                stateClass = 'status-checking';
                statusText = lang('updates_ui.status_checking');
                statusKey = 'updates_ui.status_checking';
                break;
            case 'ready':
                if (this._updateData?.has_updates) {
                    stateClass = 'status-ready';
                    statusText = lang('updates_ui.status_ready');
                    statusKey = 'updates_ui.status_ready';
                } else {
                    stateClass = 'status-up-to-date';
                    statusText = lang('updates_ui.status_up_to_date');
                    statusKey = 'updates_ui.status_up_to_date';
                }
                break;
            case 'applying':
                stateClass = 'status-applying';
                statusText = lang('updates_ui.status_applying');
                statusKey = 'updates_ui.status_applying';
                break;
            case 'applied':
                stateClass = 'status-applied';
                statusText = lang('updates_ui.status_applied');
                statusKey = 'updates_ui.status_applied';
                break;
            case 'error':
                stateClass = 'status-error';
                statusText = this._updateData?.message || lang('updates_ui.status_failed');
                statusKey = 'updates_ui.status_failed';
                break;
        }

        // Remove all state classes then add the current one
        stateClasses.forEach(cls => statusEl.removeClass(cls));
        statusEl.addClass(stateClass);
        statusEl.text(statusText);
        if (statusKey) {
            this._statusEl.setAttribute('data-lang-key', statusKey);
        }
    },

    /**
     * Check for updates from the server
     */
    _checkForUpdates: async function() {
        this._state = 'checking';
        this._updateStatusUI();
        this._checkBtn.setDisabled(true);
        this._applyBtn.setDisabled(true);
        Q(this._tableContainer).empty();

        try {
            const response = await fetch('/system/updates/check');
            const data = await response.json();

            this._updateData = data;

            if (data.status === 'error') {
                this._state = 'error';
            } else {
                this._state = 'ready';
                this._buildUpdateTable(data);
            }
        } catch (err) {
            console.error('Update check failed:', err);
            this._state = 'error';
            this._updateData = { message: err.message };
        }

        this._updateStatusUI();
        this._checkBtn.setDisabled(false);
        this._updateApplyButton();
    },

    /**
     * Build the update files table
     * @param {Object} data - Update data from API
     */
    _buildUpdateTable: function(data) {
        Q(this._tableContainer).empty();
        const files = data.files || [];
        const orphaned = data.orphaned || [];
        this._filePaths = files.map(f => f.path);

        const totalCount = files.length + orphaned.length;

        if (totalCount === 0) {
            const noUpdates = Q('<div>', { 
                class: 'updates-no-updates',
                text: lang('updates_ui.no_updates')
            }).get(0);
            noUpdates.setAttribute('data-lang-key', 'updates_ui.no_updates');
            Q(this._tableContainer).append(noUpdates);
            return;
        }

        // Summary bar
        const summary = Q('<div>', { class: 'updates-summary' }).get(0);
        let summaryText = '';
        if (files.length > 0) {
            summaryText = `<strong>${files.length}</strong> ${lang('updates_ui.files_to_update')}`;
        }
        if (orphaned.length > 0) {
            if (summaryText) summaryText += ' | ';
            summaryText += `<strong>${orphaned.length}</strong> ${lang('updates_ui.orphaned_count')}`;
        }
        Q(summary).html(`<span class="updates-summary-count">${summaryText}</span>`);
        this._summaryEl = summary;
        Q(this._tableContainer).append(summary);

        // Scrollable wrapper
        const wrapper = Q('<div>', { class: 'updates-table-wrapper' }).get(0);

        // Table
        const table = Q('<table>', { class: 'updates-table' }).get(0);

        // Header
        const thead = Q('<thead>').get(0);
        const headerRow = Q('<tr>').get(0);

        // File column
        const thFile = Q('<th>', { 
            class: 'col-file',
            text: lang('updates_ui.table_header_file')
        }).get(0);
        thFile.setAttribute('data-lang-key', 'updates_ui.table_header_file');
        Q(headerRow).append(thFile);

        // Status column
        const thStatus = Q('<th>', { 
            class: 'col-status',
            text: lang('updates_ui.table_header_status')
        }).get(0);
        thStatus.setAttribute('data-lang-key', 'updates_ui.table_header_status');
        Q(headerRow).append(thStatus);

        Q(thead).append(headerRow);
        Q(table).append(thead);

        // Body
        const tbody = Q('<tbody>').get(0);

        // Add update files
        files.forEach(file => {
            const row = Q('<tr>').get(0);
            row.dataset.path = file.path;

            // File path
            const tdFile = Q('<td>', { 
                class: 'file-path',
                text: file.path,
                title: file.path
            }).get(0);
            Q(row).append(tdFile);

            // Status
            const tdStatus = Q('<td>').get(0);
            const statusBadge = Q('<span>', { 
                class: `status-badge status-${file.status}`,
                text: file.status_label || file.status
            }).get(0);
            Q(tdStatus).append(statusBadge);
            Q(row).append(tdStatus);

            Q(tbody).append(row);
        });

        // Add orphaned files
        orphaned.forEach(path => {
            const row = Q('<tr>', { class: 'orphaned-row' }).get(0);

            // File path
            const tdFile = Q('<td>', { 
                class: 'file-path',
                text: path,
                title: path
            }).get(0);
            Q(row).append(tdFile);

            // Status
            const tdStatus = Q('<td>').get(0);
            const statusBadge = Q('<span>', { 
                class: 'status-badge status-orphaned',
                text: lang('updates.status.orphaned')
            }).get(0);
            Q(tdStatus).append(statusBadge);
            Q(row).append(tdStatus);

            Q(tbody).append(row);
        });

        Q(table).append(tbody);
        Q(wrapper).append(table);
        Q(this._tableContainer).append(wrapper);
    },

    /**
     * Update apply button state
     */
    _updateApplyButton: function() {
        const canApply = this._state === 'ready' && this._updateData?.has_updates && this._filePaths.length > 0;
        this._applyBtn.setDisabled(!canApply);
    },

    /**
     * Apply all updates
     */
    _applyUpdates: async function() {
        if (this._filePaths.length === 0) return;

        this._state = 'applying';
        this._updateStatusUI();
        this._checkBtn.setDisabled(true);
        this._applyBtn.setDisabled(true);

        try {
            const response = await fetch('/system/updates/apply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ paths: this._filePaths })
            });
            const data = await response.json();

            if (data.status === 'error') {
                this._state = 'error';
                this._updateData = data;
            } else {
                this._state = 'applied';
                this._updateData = data;
                
                // Refresh the table to show remaining updates
                if (data.remaining && data.remaining.length > 0) {
                    this._buildUpdateTable({ files: data.remaining, has_updates: true });
                } else {
                    Q(this._tableContainer).empty();
                    const success = Q('<div>', {
                        class: 'updates-no-updates updates-success',
                        text: lang('updates_ui.status_applied'),
                        'data-lang-key': 'updates_ui.status_applied'
                    }).get(0);
                    Q(this._tableContainer).append(success);
                }
            }
        } catch (err) {
            console.error('Update apply failed:', err);
            this._state = 'error';
            this._updateData = { message: err.message };
        }

        this._updateStatusUI();
        this._checkBtn.setDisabled(false);
        this._updateApplyButton();
    }
};

// Register page
Pages.register('updates', UpdatesPage);
