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
    _selectedPaths: new Set(),

    /**
     * UI References
     */
    _statusEl: null,
    _checkBtn: null,
    _applyBtn: null,
    _tableContainer: null,
    _selectAllCheckbox: null,
    _summaryEl: null,

    /**
     * Build the Updates page
     * @param {HTMLElement} container
     */
    build: function(container) {
        container.innerHTML = '';
        this._state = 'idle';
        this._updateData = null;
        this._selectedPaths = new Set();
        this._summaryEl = null;

        // Page heading
        const heading = new Heading('updates-heading', {
            title: lang('updates_ui.page_title'),
            titleLangKey: 'updates_ui.page_title',
            description: lang('updates_ui.page_description'),
            descriptionLangKey: 'updates_ui.page_description'
        });
        container.appendChild(heading.getElement());

        // Main card
        const card = document.createElement('div');
        card.className = 'updates-card';

        // Intro text
        const intro = document.createElement('p');
        intro.className = 'updates-intro';
        intro.textContent = lang('updates_ui.intro');
        intro.setAttribute('data-lang-key', 'updates_ui.intro');
        card.appendChild(intro);

        // Status area
        this._statusEl = document.createElement('div');
        this._statusEl.className = 'updates-status';
        this._updateStatusUI();
        card.appendChild(this._statusEl);

        // Action buttons
        const actionsRow = document.createElement('div');
        actionsRow.className = 'updates-actions';

        this._checkBtn = new ActionButton('check-updates', {
            label: lang('updates_ui.check_button'),
            labelLangKey: 'updates_ui.check_button',
            className: 'btn btn-primary',
            onClick: () => this._checkForUpdates()
        });
        actionsRow.appendChild(this._checkBtn.getElement());

        this._applyBtn = new ActionButton('apply-updates', {
            label: lang('updates_ui.apply_button'),
            labelLangKey: 'updates_ui.apply_button',
            className: 'btn btn-success',
            disabled: true,
            onClick: () => this._applyUpdates()
        });
        actionsRow.appendChild(this._applyBtn.getElement());

        card.appendChild(actionsRow);

        // Table container for update list
        this._tableContainer = document.createElement('div');
        this._tableContainer.className = 'updates-table-container';
        card.appendChild(this._tableContainer);

        container.appendChild(card);
    },

    /**
     * Update the status UI based on current state
     */
    _updateStatusUI: function() {
        if (!this._statusEl) return;

        let statusClass = 'updates-status';
        let statusText = '';
        let statusKey = '';

        switch (this._state) {
            case 'idle':
                statusClass += ' status-idle';
                statusText = lang('updates_ui.status_idle');
                statusKey = 'updates_ui.status_idle';
                break;
            case 'checking':
                statusClass += ' status-checking';
                statusText = lang('updates_ui.status_checking');
                statusKey = 'updates_ui.status_checking';
                break;
            case 'ready':
                if (this._updateData?.has_updates) {
                    statusClass += ' status-ready';
                    statusText = lang('updates_ui.status_ready');
                    statusKey = 'updates_ui.status_ready';
                } else {
                    statusClass += ' status-up-to-date';
                    statusText = lang('updates_ui.status_up_to_date');
                    statusKey = 'updates_ui.status_up_to_date';
                }
                break;
            case 'applying':
                statusClass += ' status-applying';
                statusText = lang('updates_ui.status_applying');
                statusKey = 'updates_ui.status_applying';
                break;
            case 'applied':
                statusClass += ' status-applied';
                statusText = lang('updates_ui.status_applied');
                statusKey = 'updates_ui.status_applied';
                break;
            case 'error':
                statusClass += ' status-error';
                statusText = this._updateData?.message || lang('updates_ui.status_failed');
                statusKey = 'updates_ui.status_failed';
                break;
        }

        this._statusEl.className = statusClass;
        this._statusEl.textContent = statusText;
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
        this._tableContainer.innerHTML = '';

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
        this._tableContainer.innerHTML = '';
        this._selectedPaths = new Set();

        const files = data.files || [];

        if (files.length === 0) {
            const noUpdates = document.createElement('div');
            noUpdates.className = 'updates-no-updates';
            noUpdates.textContent = lang('updates_ui.no_updates');
            noUpdates.setAttribute('data-lang-key', 'updates_ui.no_updates');
            this._tableContainer.appendChild(noUpdates);
            return;
        }

        // Select all by default
        files.forEach(f => this._selectedPaths.add(f.path));

        // Summary bar
        const summary = document.createElement('div');
        summary.className = 'updates-summary';
        summary.innerHTML = `
            <span class="updates-summary-count"><strong>${files.length}</strong> ${lang('updates_ui.files_to_update')}</span>
            <span class="updates-summary-selection">${this._selectedPaths.size} ${lang('updates_ui.selected')}</span>
        `;
        this._summaryEl = summary;
        this._tableContainer.appendChild(summary);

        // Scrollable wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'updates-table-wrapper';

        // Table
        const table = document.createElement('table');
        table.className = 'updates-table';

        // Header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');

        // Select all checkbox
        const thSelect = document.createElement('th');
        thSelect.className = 'col-select';
        this._selectAllCheckbox = document.createElement('input');
        this._selectAllCheckbox.type = 'checkbox';
        this._selectAllCheckbox.checked = true;
        this._selectAllCheckbox.addEventListener('change', () => this._toggleSelectAll());
        thSelect.appendChild(this._selectAllCheckbox);
        headerRow.appendChild(thSelect);

        // File column
        const thFile = document.createElement('th');
        thFile.className = 'col-file';
        thFile.textContent = lang('updates_ui.table_header_file');
        thFile.setAttribute('data-lang-key', 'updates_ui.table_header_file');
        headerRow.appendChild(thFile);

        // Status column
        const thStatus = document.createElement('th');
        thStatus.className = 'col-status';
        thStatus.textContent = lang('updates_ui.table_header_status');
        thStatus.setAttribute('data-lang-key', 'updates_ui.table_header_status');
        headerRow.appendChild(thStatus);

        // Local hash column
        const thLocal = document.createElement('th');
        thLocal.className = 'col-hash';
        thLocal.textContent = lang('updates_ui.table_header_local');
        thLocal.setAttribute('data-lang-key', 'updates_ui.table_header_local');
        headerRow.appendChild(thLocal);

        // Remote hash column
        const thRemote = document.createElement('th');
        thRemote.className = 'col-hash';
        thRemote.textContent = lang('updates_ui.table_header_remote');
        thRemote.setAttribute('data-lang-key', 'updates_ui.table_header_remote');
        headerRow.appendChild(thRemote);

        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Body
        const tbody = document.createElement('tbody');

        files.forEach(file => {
            const row = document.createElement('tr');
            row.dataset.path = file.path;

            // Checkbox
            const tdSelect = document.createElement('td');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = true;
            checkbox.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this._selectedPaths.add(file.path);
                } else {
                    this._selectedPaths.delete(file.path);
                }
                this._updateSelectAllState();
                this._updateApplyButton();
            });
            tdSelect.appendChild(checkbox);
            row.appendChild(tdSelect);

            // File path
            const tdFile = document.createElement('td');
            tdFile.className = 'file-path';
            tdFile.textContent = file.path;
            tdFile.title = file.path;
            row.appendChild(tdFile);

            // Status
            const tdStatus = document.createElement('td');
            const statusBadge = document.createElement('span');
            statusBadge.className = `status-badge status-${file.status}`;
            statusBadge.textContent = file.status_label || file.status;
            tdStatus.appendChild(statusBadge);
            row.appendChild(tdStatus);

            // Local hash
            const tdLocal = document.createElement('td');
            tdLocal.className = 'hash-cell';
            tdLocal.textContent = file.local_checksum || lang('updates_ui.hash_missing');
            row.appendChild(tdLocal);

            // Remote hash
            const tdRemote = document.createElement('td');
            tdRemote.className = 'hash-cell';
            tdRemote.textContent = file.remote_checksum || lang('updates_ui.hash_missing');
            row.appendChild(tdRemote);

            tbody.appendChild(row);
        });

        table.appendChild(tbody);
        wrapper.appendChild(table);
        this._tableContainer.appendChild(wrapper);

        // Orphaned files section
        if (data.orphaned && data.orphaned.length > 0) {
            this._buildOrphanedSection(data.orphaned);
        }
    },

    /**
     * Build orphaned files section
     * @param {Array} orphaned - List of orphaned file paths
     */
    _buildOrphanedSection: function(orphaned) {
        const section = document.createElement('div');
        section.className = 'updates-orphaned';

        const title = document.createElement('h3');
        title.textContent = lang('updates_ui.orphaned_title');
        title.setAttribute('data-lang-key', 'updates_ui.orphaned_title');
        section.appendChild(title);

        const list = document.createElement('ul');
        orphaned.forEach(path => {
            const li = document.createElement('li');
            li.textContent = path;
            list.appendChild(li);
        });
        section.appendChild(list);

        this._tableContainer.appendChild(section);
    },

    /**
     * Toggle select all checkbox
     */
    _toggleSelectAll: function() {
        const isChecked = this._selectAllCheckbox.checked;
        const checkboxes = this._tableContainer.querySelectorAll('tbody input[type="checkbox"]');
        
        checkboxes.forEach(cb => {
            cb.checked = isChecked;
            const path = cb.closest('tr').dataset.path;
            if (isChecked) {
                this._selectedPaths.add(path);
            } else {
                this._selectedPaths.delete(path);
            }
        });

        this._updateApplyButton();
    },

    /**
     * Update select all checkbox state based on individual selections
     */
    _updateSelectAllState: function() {
        if (!this._selectAllCheckbox) return;
        
        const checkboxes = this._tableContainer.querySelectorAll('tbody input[type="checkbox"]');
        const allChecked = Array.from(checkboxes).every(cb => cb.checked);
        const someChecked = Array.from(checkboxes).some(cb => cb.checked);
        
        this._selectAllCheckbox.checked = allChecked;
        this._selectAllCheckbox.indeterminate = someChecked && !allChecked;
        
        this._updateSummarySelection();
    },

    /**
     * Update summary selection count
     */
    _updateSummarySelection: function() {
        if (!this._summaryEl) return;
        const selectionEl = this._summaryEl.querySelector('.updates-summary-selection');
        if (selectionEl) {
            selectionEl.textContent = `${this._selectedPaths.size} ${lang('updates_ui.selected')}`;
        }
    },

    /**
     * Update apply button state
     */
    _updateApplyButton: function() {
        const hasSelection = this._selectedPaths.size > 0;
        const canApply = this._state === 'ready' && this._updateData?.has_updates && hasSelection;
        this._applyBtn.setDisabled(!canApply);
    },

    /**
     * Apply selected updates
     */
    _applyUpdates: async function() {
        if (this._selectedPaths.size === 0) return;

        this._state = 'applying';
        this._updateStatusUI();
        this._checkBtn.setDisabled(true);
        this._applyBtn.setDisabled(true);

        try {
            const response = await fetch('/system/updates/apply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ paths: Array.from(this._selectedPaths) })
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
                    this._tableContainer.innerHTML = '';
                    const success = document.createElement('div');
                    success.className = 'updates-no-updates updates-success';
                    success.textContent = lang('updates_ui.status_applied');
                    success.setAttribute('data-lang-key', 'updates_ui.status_applied');
                    this._tableContainer.appendChild(success);
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
