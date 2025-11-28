/**
 * HootSight - Dataset Page
 * Dataset management with folder browser, image grid, tagging, and sync/build capabilities
 */

const DatasetPage = {
    /**
     * Page identifier
     */
    name: 'dataset',

    /**
     * Internal state
     */
    _container: null,
    _projectName: null,
    _datasetType: null,         // auto-detected or overridden
    _currentFolder: null,       // Currently selected folder path
    _folderTree: null,          // Cached folder tree
    _images: [],                // Current page images
    _currentPage: 1,
    _pageSize: 50,
    _totalPages: 1,
    _searchQuery: '',
    _searchMode: 'filename',    // 'filename' or 'annotation'
    _selectedImages: new Set(), // Selected image IDs for bulk operations
    _lastSelectedIndex: -1,     // For shift+click range selection
    _allTags: [],               // All unique tags in the project (for suggestions)
    _recentTags: [],            // Recently used tags
    _duplicates: new Map(),     // Hash -> [image_ids] for duplicate detection
    _showDuplicatesOnly: false, // Filter to show only duplicates
    _duplicateIds: new Set(),   // Set of image IDs that are duplicates
    _keyboardHandler: null,     // Keyboard event handler reference
    _activeZoom: null,          // Active zoom/pan state for image manipulation
    
    // UI element references
    _folderBrowser: null,
    _imageGrid: null,
    _toolbar: null,
    _uploadZone: null,
    _bulkActionsBar: null,
    
    // Widget references
    _searchInput: null,
    _searchModeDropdown: null,
    _typeDropdown: null,

    /**
     * Build the Dataset page
     * @param {HTMLElement} container
     */
    build: async function(container) {
        container.innerHTML = '';
        this._container = container;
        this._selectedImages.clear();

        // Check for active project
        this._projectName = Config.getActiveProject();
        if (!this._projectName) {
            this._showNoProjectMessage();
            return;
        }

        // Get project info and dataset type
        await this._loadProjectInfo();

        // Build layout
        this._buildLayout();

        // Load initial data
        await this._loadFolderTree();
        await this._loadImages();
    },

    /**
     * Show no project selected message
     */
    _showNoProjectMessage: function() {
        const wrapper = Q('<div>', { class: 'no-project-message' }).get(0);
        
        const icon = Q('<div>', { class: 'no-project-icon', text: '!' }).get(0);
        const title = Q('<h2>', { text: lang('dataset_page.no_project.title') }).get(0);
        const desc = Q('<p>', { text: lang('dataset_page.no_project.description') }).get(0);
        
        const btn = Q('<button>', { 
            class: 'btn btn-primary',
            text: lang('dataset_page.no_project.button')
        }).get(0);
        
        Q(btn).on('click', () => {
            if (typeof Navigation !== 'undefined') {
                Navigation.navigateTo('projects');
            }
        });
        
        wrapper.appendChild(icon);
        wrapper.appendChild(title);
        wrapper.appendChild(desc);
        wrapper.appendChild(btn);
        this._container.appendChild(wrapper);
    },

    /**
     * Load project info
     */
    _loadProjectInfo: async function() {
        try {
            const data = await API.projects.get(this._projectName);
            this._datasetType = data.dataset_type || 'unknown';
        } catch (err) {
            console.error('Failed to load project info:', err);
            this._datasetType = 'unknown';
        }
    },

    /**
     * Build main layout
     */
    _buildLayout: function() {
        // Main container with flex layout
        const main = Q('<div>', { class: 'dataset-page' }).get(0);

        // Toolbar (includes pagination and status)
        this._toolbar = this._buildToolbar();
        main.appendChild(this._toolbar);

        // Content area (folder browser + image grid)
        const content = Q('<div>', { class: 'dataset-content' }).get(0);

        // Folder browser (left sidebar)
        this._folderBrowser = this._buildFolderBrowser();
        content.appendChild(this._folderBrowser);

        // Image grid (main area)
        this._imageGrid = this._buildImageGrid();
        content.appendChild(this._imageGrid);

        main.appendChild(content);

        // Bulk actions bar (hidden by default)
        this._bulkActionsBar = this._buildBulkActionsBar();
        main.appendChild(this._bulkActionsBar);

        this._container.appendChild(main);

        // Load all tags for suggestions
        this._loadAllTags();

        // Setup keyboard shortcuts
        this._setupKeyboardShortcuts();

        // Setup context menu for image cards
        this._setupContextMenu();

        // Setup global zoom/pan event handlers
        this._setupGlobalZoomPanHandlers();

        // Start initial discovery with progress
        this._startInitialDiscovery();
    },

    /**
     * Setup context menu for dataset page elements
     */
    _setupContextMenu: function() {
        // Context menu for image cards
        ContextMenu.register('.image-card', (element, event) => {
            const imageId = element.dataset.id;
            const image = this._images.find(img => img.relative_path === imageId);
            if (!image) return [];

            const isSelected = this._selectedImages.has(imageId);
            const selectedCount = this._selectedImages.size;

            const items = [];

            // View/Edit item
            items.push({
                label: lang('context_menu.view_image'),
                icon: 'view.svg',
                action: () => this._viewImage(image)
            });

            items.push({ type: 'separator' });

            // Selection options
            if (!isSelected) {
                items.push({
                    label: lang('context_menu.select'),
                    icon: 'check.svg',
                    action: () => {
                        this._selectedImages.add(imageId);
                        this._updateSelectionVisuals();
                        this._updateBulkActionsBar();
                    }
                });
            } else {
                items.push({
                    label: lang('context_menu.deselect'),
                    icon: 'x.svg',
                    action: () => {
                        this._selectedImages.delete(imageId);
                        this._updateSelectionVisuals();
                        this._updateBulkActionsBar();
                    }
                });
            }

            if (selectedCount > 0) {
                items.push({
                    label: lang('context_menu.select_all'),
                    icon: 'select-all.svg',
                    shortcut: 'Ctrl+A',
                    action: () => this._selectAllImages()
                });
                items.push({
                    label: lang('context_menu.clear_selection'),
                    icon: 'x.svg',
                    shortcut: 'Esc',
                    action: () => this._clearSelection()
                });
            }

            items.push({ type: 'separator' });

            // Copy filename
            items.push({
                label: lang('context_menu.copy_filename'),
                icon: 'copy.svg',
                action: () => navigator.clipboard.writeText(image.filename)
            });

            // Copy path
            items.push({
                label: lang('context_menu.copy_path'),
                icon: 'copy.svg',
                action: () => navigator.clipboard.writeText(image.relative_path)
            });

            items.push({ type: 'separator' });

            // Delete options
            if (selectedCount > 1 && isSelected) {
                items.push({
                    label: lang('context_menu.delete_selected', { count: selectedCount }),
                    icon: 'trash.svg',
                    shortcut: 'Del',
                    danger: true,
                    action: () => this._bulkDelete()
                });
            } else {
                items.push({
                    label: lang('context_menu.delete'),
                    icon: 'trash.svg',
                    danger: true,
                    action: () => this._deleteImage(image)
                });
            }

            return items;
        });

        // Context menu for folder tree items
        ContextMenu.register('.folder-item', (element, event) => {
            const folderPath = element.dataset.path;
            const folderName = element.querySelector('.folder-name')?.textContent || '';
            
            // Don't show for root folder
            if (!folderPath) return [];
            
            return [
                {
                    label: lang('context_menu.open_folder'),
                    icon: 'folder_open.svg',
                    action: () => {
                        this._currentFolder = folderPath;
                        this._currentPage = 1;
                        this._loadImages();
                    }
                },
                { type: 'separator' },
                {
                    label: lang('context_menu.new_subfolder'),
                    icon: 'folder_closed.svg',
                    action: () => this._createFolder(folderPath)
                },
                {
                    label: lang('dataset_page.rename_folder'),
                    icon: 'edit.svg',
                    action: () => this._renameFolder({ path: folderPath, name: folderName })
                },
                { type: 'separator' },
                {
                    label: lang('context_menu.copy_path'),
                    icon: 'copy.svg',
                    action: () => navigator.clipboard.writeText(folderPath)
                },
                { type: 'separator' },
                {
                    label: lang('dataset_page.delete_folder'),
                    icon: 'trash.svg',
                    danger: true,
                    action: () => this._deleteFolder({ path: folderPath, name: folderName })
                }
            ];
        });
    },

    /**
     * Setup global zoom/pan event handlers (for mouse move/up during pan)
     */
    _setupGlobalZoomPanHandlers: function() {
        // Mouse move handler for panning
        this._panMoveHandler = (e) => {
            if (!this._activeZoom || !this._activeZoom.isPanning) return;
            
            const { startX, startY, img, state, zoom } = this._activeZoom;
            const panX = (e.clientX - startX) / zoom;
            const panY = (e.clientY - startY) / zoom;
            
            // Update both active zoom state and per-image state
            this._activeZoom.panX = panX;
            this._activeZoom.panY = panY;
            if (state) {
                state.panX = panX;
                state.panY = panY;
            }
            
            img.style.transform = `scale(${zoom}) translate(${panX}px, ${panY}px)`;
        };

        // Mouse up handler to end panning
        this._panUpHandler = () => {
            if (!this._activeZoom) return;
            this._activeZoom.isPanning = false;
            if (this._activeZoom.container) {
                this._activeZoom.container.style.cursor = '';
            }
            // Trigger save callback if available (debounced save of zoom/pan state)
            if (this._activeZoom.saveCallback) {
                this._activeZoom.saveCallback();
            }
            // Clear active zoom reference
            this._activeZoom = null;
        };

        document.addEventListener('mousemove', this._panMoveHandler);
        document.addEventListener('mouseup', this._panUpHandler);
    },

    /**
     * View image in detail (placeholder for future lightbox)
     */
    _viewImage: function(image) {
        // For now, open in new tab. Could be replaced with lightbox
        window.open(image.image_url, '_blank');
    },

    /**
     * Setup header action buttons (called by app.js after page build)
     */
    setupHeaderActions: function() {
        const headerActions = document.getElementById('header-actions');
        if (!headerActions) return;

        // Sync button using ActionButton widget (icon only)
        this._syncBtnWidget = new ActionButton('dataset-sync', {
            label: '',
            className: 'btn btn-secondary btn-icon',
            onClick: () => this._startSync()
        });
        this._syncBtn = this._syncBtnWidget.getElement();
        // Add icon to sync button
        const syncIcon = Q('<img>', { src: '/static/icons/sync.svg', alt: 'Sync', class: 'btn-icon-img' }).get(0);
        this._syncBtn.appendChild(syncIcon);
        this._syncBtn.title = lang('dataset_page.sync_button');
        headerActions.appendChild(this._syncBtn);

        // Build button using ActionButton widget
        this._buildBtnWidget = new ActionButton('dataset-build', {
            label: lang('dataset_page.build_button'),
            className: 'btn btn-primary',
            onClick: () => this._startBuild()
        });
        this._buildBtn = this._buildBtnWidget.getElement();
        headerActions.appendChild(this._buildBtn);
    },

    /**
     * Start initial discovery with progress tracking
     */
    _startInitialDiscovery: async function() {
        // Show progress bar
        ProgressManager.show('discovery', {
            label: lang('dataset_page.progress_discovery'),
            progress: 0,
            status: lang('dataset_page.progress_starting')
        });

        try {
            // Start refresh
            await API.datasetEditor.refresh(this._projectName);
            
            // Poll for status
            await this._pollDiscoveryStatus();
            
        } catch (err) {
            console.error('Discovery failed:', err);
            ProgressManager.hide('discovery');
        }
    },

    /**
     * Poll discovery status
     */
    _pollDiscoveryStatus: async function() {
        const pollInterval = 500;
        
        const poll = async () => {
            try {
                const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(this._projectName)}/refresh/status`);
                const status = await response.json();
                
                if (status.status === 'running' || status.status === 'pending') {
                    const progress = status.total_items > 0 
                        ? Math.round((status.processed_items / status.total_items) * 100)
                        : 0;
                    
                    ProgressManager.update('discovery', progress, `${status.processed_items}/${status.total_items}`);
                    setTimeout(poll, pollInterval);
                } else {
                    // Complete or error
                    ProgressManager.update('discovery', 100, lang('dataset_page.progress_complete'));
                    setTimeout(() => {
                        ProgressManager.hide('discovery');
                        // Refresh data
                        this._loadFolderTree();
                        this._loadImages();
                    }, 1000);
                }
            } catch (err) {
                console.error('Poll error:', err);
                ProgressManager.hide('discovery');
            }
        };
        
        poll();
    },

    /**
     * Setup keyboard shortcuts for the dataset page
     */
    _setupKeyboardShortcuts: function() {
        // Remove existing handler if any
        if (this._keyboardHandler) {
            document.removeEventListener('keydown', this._keyboardHandler);
        }

        this._keyboardHandler = (e) => {
            // Ignore if focus is on input/textarea
            const tag = document.activeElement.tagName.toLowerCase();
            if (tag === 'input' || tag === 'textarea' || tag === 'select') {
                return;
            }

            // CTRL+A - Select all visible images
            if (e.ctrlKey && e.key === 'a') {
                e.preventDefault();
                this._selectAllImages();
                return;
            }

            // DELETE - Delete selected images
            if (e.key === 'Delete' && this._selectedImages.size > 0) {
                e.preventDefault();
                this._bulkDelete();
                return;
            }

            // ESCAPE - Clear selection
            if (e.key === 'Escape' && this._selectedImages.size > 0) {
                e.preventDefault();
                this._clearSelection();
                return;
            }
        };

        document.addEventListener('keydown', this._keyboardHandler);
    },

    /**
     * Select all visible images
     */
    _selectAllImages: function() {
        this._images.forEach(img => {
            this._selectedImages.add(img.relative_path);
        });
        this._updateBulkActionsBar();
        this._refreshImageGrid();
    },

    /**
     * Build toolbar
     */
    _buildToolbar: function() {
        const toolbar = Q('<div>', { class: 'dataset-toolbar' }).get(0);

        // Left section - project info and dataset type
        const leftSection = Q('<div>', { class: 'toolbar-left' }).get(0);
        
        const projectInfo = Q('<div>', { class: 'toolbar-project' }).get(0);
        const projectLabel = Q('<span>', { class: 'project-label', text: lang('dataset_page.project_label') }).get(0);
        const projectName = Q('<span>', { class: 'project-name', text: this._projectName }).get(0);
        projectInfo.appendChild(projectLabel);
        projectInfo.appendChild(projectName);
        leftSection.appendChild(projectInfo);

        // Dataset type selector
        const typeSelector = this._buildDatasetTypeSelector();
        leftSection.appendChild(typeSelector);

        toolbar.appendChild(leftSection);

        // Center section - search
        const centerSection = Q('<div>', { class: 'toolbar-center' }).get(0);
        const search = this._buildSearchBox();
        centerSection.appendChild(search);
        toolbar.appendChild(centerSection);

        // Right section - pagination and status (moved from statusbar)
        const rightSection = Q('<div>', { class: 'toolbar-right' }).get(0);
        
        // Image count
        this._statusCount = Q('<span>', { class: 'status-count' }).get(0);
        rightSection.appendChild(this._statusCount);

        // Pagination controls
        const paginationControls = Q('<div>', { class: 'pagination-controls' }).get(0);
        
        // Previous button
        this._prevBtn = new ActionButton('page-prev', {
            label: '<',
            className: 'btn btn-sm btn-ghost',
            onClick: () => this._goToPage(this._currentPage - 1)
        });
        paginationControls.appendChild(this._prevBtn.getElement());

        // Page dropdown
        this._pageDropdown = new Dropdown('page-select', {
            options: ['1'],
            optionLabels: { '1': '1' },
            default: '1'
        });
        this._pageDropdown.onChange((value) => {
            this._goToPage(parseInt(value, 10));
        });
        const pageDropdownEl = this._pageDropdown.getElement();
        pageDropdownEl.classList.add('pagination-dropdown');
        paginationControls.appendChild(pageDropdownEl);

        // Total pages indicator
        this._totalPagesLabel = Q('<span>', { class: 'total-pages', text: '/ 1' }).get(0);
        paginationControls.appendChild(this._totalPagesLabel);

        // Next button
        this._nextBtn = new ActionButton('page-next', {
            label: '>',
            className: 'btn btn-sm btn-ghost',
            onClick: () => this._goToPage(this._currentPage + 1)
        });
        paginationControls.appendChild(this._nextBtn.getElement());

        rightSection.appendChild(paginationControls);

        // Page size dropdown
        const pageSizeGroup = Q('<div>', { class: 'page-size-group' }).get(0);
        
        const pageSizeLabel = Q('<span>', { class: 'page-size-label', text: lang('dataset_page.per_page') }).get(0);
        pageSizeGroup.appendChild(pageSizeLabel);

        // Must match backend page_sizes: [25, 50, 100, 200, 500, 750]
        const pageSizes = ['25', '50', '100', '200', '500', '750'];
        const pageSizeLabels = {};
        pageSizes.forEach(size => { pageSizeLabels[size] = size; });

        this._pageSizeDropdown = new Dropdown('page-size', {
            options: pageSizes,
            optionLabels: pageSizeLabels,
            default: String(this._pageSize)
        });
        this._pageSizeDropdown.onChange((value) => {
            this._pageSize = parseInt(value, 10);
            this._currentPage = 1;
            this._loadImages();
        });
        const pageSizeEl = this._pageSizeDropdown.getElement();
        pageSizeEl.classList.add('page-size-dropdown');
        pageSizeGroup.appendChild(pageSizeEl);

        rightSection.appendChild(pageSizeGroup);
        toolbar.appendChild(rightSection);

        return toolbar;
    },

    /**
     * Build dataset type selector using Dropdown widget
     */
    _buildDatasetTypeSelector: function() {
        const container = Q('<div>', { class: 'dataset-type-selector' }).get(0);
        
        const types = ['auto', 'multi_label', 'folder_classification', 'annotation', 'mixed'];
        const optionLabels = {
            'auto': `${lang('dataset_page.type_auto')} (${this._datasetType})`,
            'multi_label': lang('dataset_page.types.multi_label'),
            'folder_classification': lang('dataset_page.types.folder_classification'),
            'annotation': lang('dataset_page.types.annotation'),
            'mixed': lang('dataset_page.types.mixed')
        };

        this._typeDropdown = new Dropdown('dataset-type', {
            label: lang('dataset_page.type_label'),
            options: types,
            optionLabels: optionLabels,
            default: 'auto'
        });

        this._typeDropdown.onChange((value) => {
            if (value === 'auto') {
                this._loadProjectInfo().then(() => this._refreshImageGrid());
            } else {
                this._datasetType = value;
                this._refreshImageGrid();
            }
        });

        container.appendChild(this._typeDropdown.getElement());
        return container;
    },

    /**
     * Build search box using TextInput and Dropdown widgets
     */
    _buildSearchBox: function() {
        const container = Q('<div>', { class: 'search-container' }).get(0);

        // Search mode dropdown widget
        this._searchModeDropdown = new Dropdown('search-mode', {
            options: ['filename', 'annotation'],
            optionLabels: {
                'filename': lang('dataset_page.search_filename'),
                'annotation': lang('dataset_page.search_annotation')
            },
            default: 'filename'
        });

        this._searchModeDropdown.onChange((value) => {
            this._searchMode = value;
            if (this._searchQuery) {
                this._loadImages();
            }
        });

        // Get just the dropdown container (without the widget wrapper)
        const dropdownEl = this._searchModeDropdown.getElement();
        dropdownEl.classList.add('search-mode-widget');
        container.appendChild(dropdownEl);

        // Search input widget
        this._searchInput = new TextInput('search-query', {
            placeholder: lang('dataset_page.search_placeholder')
        });

        let searchTimeout = null;
        this._searchInput.onChange((value) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this._searchQuery = value;
                this._currentPage = 1;
                this._loadImages();
            }, 300);
        });

        const inputEl = this._searchInput.getElement();
        inputEl.classList.add('search-input-widget');
        container.appendChild(inputEl);

        // Show Duplicates toggle button
        const dupToggle = Q('<button>', { 
            class: 'btn btn-ghost btn-icon duplicate-toggle',
            title: lang('dataset_page.show_duplicates')
        }).get(0);
        dupToggle.innerHTML = '<img src="/static/icons/copy.svg" alt="Duplicates" class="btn-icon-img">';
        this._dupToggleBtn = dupToggle;
        
        Q(dupToggle).on('click', async () => {
            if (this._showDuplicatesOnly) {
                // Turn off duplicate filter
                this._showDuplicatesOnly = false;
                this._duplicateIds.clear();
                dupToggle.classList.remove('active');
                this._currentPage = 1;
                await this._loadImages();
            } else {
                // Scan for duplicates and show them
                await this._loadDuplicatesFilter();
            }
        });
        container.appendChild(dupToggle);

        return container;
    },

    /**
     * Load duplicate filter - scans for duplicates and filters view
     */
    _loadDuplicatesFilter: async function() {
        this._dupToggleBtn.disabled = true;
        this._dupToggleBtn.classList.add('loading');
        
        try {
            const result = await API.datasetEditor.scanDuplicates(this._projectName);
            
            if (result.total_groups === 0) {
                Modal.alert(lang('dataset_page.duplicates_none'));
                return;
            }
            
            // Collect all duplicate image IDs
            this._duplicateIds.clear();
            result.groups.forEach(group => {
                group.images.forEach(img => {
                    this._duplicateIds.add(img.relative_path);
                });
            });
            
            // Store duplicate groups for display
            this._duplicates.clear();
            result.groups.forEach(group => {
                this._duplicates.set(group.content_hash, group.images.map(img => img.relative_path));
            });
            
            this._showDuplicatesOnly = true;
            this._dupToggleBtn.classList.add('active');
            this._currentPage = 1;
            await this._loadImages();
            
        } catch (err) {
            console.error('Duplicate scan failed:', err);
            Modal.alert(lang('dataset_page.duplicates_error'));
        } finally {
            this._dupToggleBtn.disabled = false;
            this._dupToggleBtn.classList.remove('loading');
        }
    },

    /**
     * Build folder browser
     */
    _buildFolderBrowser: function() {
        const browser = Q('<div>', { class: 'folder-browser' }).get(0);

        // Header
        const header = Q('<div>', { class: 'folder-header' }).get(0);
        const title = Q('<span>', { class: 'folder-title', text: lang('dataset_page.folders') }).get(0);
        header.appendChild(title);

        // New folder button
        const newFolderBtn = Q('<button>', { class: 'btn btn-sm btn-ghost', title: lang('dataset_page.new_folder') }).get(0);
        newFolderBtn.innerHTML = '+';
        Q(newFolderBtn).on('click', () => this._createFolder());
        header.appendChild(newFolderBtn);

        browser.appendChild(header);

        // Folder tree container
        this._folderTreeContainer = Q('<div>', { class: 'folder-tree' }).get(0);
        browser.appendChild(this._folderTreeContainer);

        return browser;
    },

    /**
     * Build image grid
     */
    _buildImageGrid: function() {
        const grid = Q('<div>', { class: 'image-grid-container' }).get(0);

        // Upload zone (drag & drop overlay)
        this._uploadZone = Q('<div>', { class: 'upload-zone' }).get(0);
        this._uploadZone.innerHTML = `
            <div class="upload-zone-content">
                <img src="/static/icons/upload.svg" alt="Upload" class="upload-icon">
                <p>${lang('dataset_page.drop_images')}</p>
            </div>
        `;
        grid.appendChild(this._uploadZone);

        // Image cards container
        this._imageCardsContainer = Q('<div>', { class: 'image-cards' }).get(0);
        grid.appendChild(this._imageCardsContainer);

        // Setup drag & drop
        this._setupDragDrop(grid);

        return grid;
    },

    /**
     * Go to specific page
     */
    _goToPage: function(page) {
        if (page < 1 || page > this._totalPages || page === this._currentPage) return;
        this._currentPage = page;
        this._loadImages();
    },

    /**
     * Setup drag and drop for file upload
     */
    _setupDragDrop: function(container) {
        let dragCounter = 0;

        Q(container).on('dragenter', (e) => {
            e.preventDefault();
            dragCounter++;
            Q(this._uploadZone).addClass('active');
        });

        Q(container).on('dragleave', (e) => {
            e.preventDefault();
            dragCounter--;
            if (dragCounter === 0) {
                Q(this._uploadZone).removeClass('active');
            }
        });

        Q(container).on('dragover', (e) => {
            e.preventDefault();
        });

        Q(container).on('drop', (e) => {
            e.preventDefault();
            dragCounter = 0;
            Q(this._uploadZone).removeClass('active');
            
            const files = Array.from(e.dataTransfer.files).filter(f => 
                f.type.startsWith('image/')
            );
            
            if (files.length > 0) {
                this._uploadFiles(files);
            }
        });
    },

    /**
     * Load folder tree from API
     */
    _loadFolderTree: async function() {
        try {
            this._folderTree = await API.datasetEditor.getFolders(this._projectName);
            this._renderFolderTree();
        } catch (err) {
            console.error('Failed to load folder tree:', err);
        }
    },

    /**
     * Render folder tree
     */
    _renderFolderTree: function() {
        this._folderTreeContainer.innerHTML = '';

        if (!this._folderTree) return;

        const renderNode = (node, depth = 0) => {
            const hasChildren = node.children && node.children.length > 0;
            const item = Q('<div>', { 
                class: 'folder-item' + (node.path === this._currentFolder ? ' active' : ''),
                'data-path': node.path
            }).get(0);
            item.style.paddingLeft = (depth * 16 + 8) + 'px';

            // Icon
            const icon = Q('<img>', {
                src: `/static/icons/${hasChildren ? 'folder_open' : 'folder_closed'}.svg`,
                class: 'folder-icon'
            }).get(0);
            item.appendChild(icon);

            // Name
            const name = Q('<span>', { 
                class: 'folder-name',
                text: node.name || lang('dataset_page.root_folder')
            }).get(0);
            item.appendChild(name);

            // Count
            const count = Q('<span>', { 
                class: 'folder-count',
                text: `(${node.image_count})`
            }).get(0);
            item.appendChild(count);

            // Click handler
            Q(item).on('click', () => {
                this._selectFolder(node.path);
            });

            this._folderTreeContainer.appendChild(item);

            // Render children
            if (hasChildren) {
                node.children.forEach(child => renderNode(child, depth + 1));
            }
        };

        renderNode(this._folderTree);
    },

    /**
     * Select folder
     */
    _selectFolder: function(path) {
        this._currentFolder = path === '' ? null : path;
        this._currentPage = 1;
        this._renderFolderTree();
        this._loadImages();
    },

    /**
     * Load images from API
     */
    _loadImages: async function() {
        try {
            const params = {
                page: this._currentPage,
                page_size: this._pageSize
            };

            if (this._searchQuery) {
                params.search = this._searchQuery;
            }
            if (this._currentFolder) {
                params.folder = this._currentFolder;
            }

            const data = await API.datasetEditor.getItems(this._projectName, params);
            
            // If duplicate filter is active, filter the results
            if (this._showDuplicatesOnly && this._duplicateIds.size > 0) {
                this._images = data.items.filter(img => this._duplicateIds.has(img.relative_path));
            } else {
                this._images = data.items;
            }
            
            this._totalPages = data.total_pages;
            this._renderImages();
            this._updateStatusBar(this._showDuplicatesOnly ? this._duplicateIds.size : data.total_items);
            this._renderPagination();
        } catch (err) {
            console.error('Failed to load images:', err);
        }
    },

    /**
     * Render images
     */
    _renderImages: function() {
        this._imageCardsContainer.innerHTML = '';
        this._lastSelectedIndex = -1;

        this._images.forEach((image, index) => {
            const card = this._createImageCard(image, index);
            this._imageCardsContainer.appendChild(card);
        });

        // Update bulk actions bar visibility
        this._updateBulkActionsBar();
    },

    /**
     * Create image card
     * @param {Object} image - Image data
     * @param {number} index - Index in current images array
     */
    _createImageCard: function(image, index) {
        const isSelected = this._selectedImages.has(image.relative_path);
        const card = Q('<div>', { 
            class: 'image-card' + (isSelected ? ' selected' : ''),
            'data-id': image.relative_path,
            'data-index': index
        }).get(0);

        // Thumbnail container
        const thumbContainer = Q('<div>', { class: 'image-thumb-container' }).get(0);
        
        // Thumbnail
        const thumb = Q('<img>', {
            src: image.image_url,
            class: 'image-thumb',
            alt: image.filename,
            loading: 'lazy'
        }).get(0);
        thumbContainer.appendChild(thumb);

        // Delete button
        const deleteBtn = Q('<button>', { 
            class: 'image-delete-btn',
            title: lang('dataset_page.delete_image')
        }).get(0);
        deleteBtn.innerHTML = '&times;';
        Q(deleteBtn).on('click', (e) => {
            e.stopPropagation();
            this._deleteImage(image);
        });
        thumbContainer.appendChild(deleteBtn);

        // Duplicate indicator - show if this image is in any duplicate group
        if (this._showDuplicatesOnly && this._duplicateIds.has(image.relative_path)) {
            // Find which group this image belongs to
            let dupCount = 0;
            for (const [hash, paths] of this._duplicates) {
                if (paths.includes(image.relative_path)) {
                    dupCount = paths.length;
                    break;
                }
            }
            if (dupCount > 1) {
                const dupIndicator = Q('<span>', { 
                    class: 'duplicate-indicator',
                    title: lang('dataset_page.duplicate_count', { count: dupCount }),
                    text: dupCount
                }).get(0);
                thumbContainer.appendChild(dupIndicator);
            }
        }

        card.appendChild(thumbContainer);

        // Click handler for selection (Windows-style)
        Q(card).on('click', (e) => {
            // Don't select if clicking on buttons or inputs
            if (e.target.closest('button, input, .tag-remove, .tag-add-input')) return;
            this._handleImageSelect(image, index, e);
        });

        // Setup zoom on CTRL + scroll (pass image for bounds persistence)
        this._setupImageZoom(thumbContainer, thumb, image);

        // Filename
        const filename = Q('<div>', { 
            class: 'image-filename',
            text: image.filename,
            title: image.relative_path
        }).get(0);
        card.appendChild(filename);

        // Annotation editor based on dataset type
        const editor = this._createAnnotationEditor(image);
        if (editor) {
            card.appendChild(editor);
        }

        return card;
    },

    /**
     * Handle image selection (Windows-style: Ctrl+Click, Shift+Click)
     */
    _handleImageSelect: function(image, index, event) {
        if (event.shiftKey && this._lastSelectedIndex !== -1) {
            // Shift+Click: Select range
            const start = Math.min(this._lastSelectedIndex, index);
            const end = Math.max(this._lastSelectedIndex, index);
            
            // If not Ctrl, clear current selection first
            if (!event.ctrlKey) {
                this._selectedImages.clear();
            }
            
            for (let i = start; i <= end; i++) {
                if (this._images[i]) {
                    this._selectedImages.add(this._images[i].relative_path);
                }
            }
        } else if (event.ctrlKey) {
            // Ctrl+Click: Toggle single item
            if (this._selectedImages.has(image.relative_path)) {
                this._selectedImages.delete(image.relative_path);
            } else {
                this._selectedImages.add(image.relative_path);
            }
            this._lastSelectedIndex = index;
        } else {
            // Regular click: Select only this item
            this._selectedImages.clear();
            this._selectedImages.add(image.relative_path);
            this._lastSelectedIndex = index;
        }

        // Update visual state
        this._updateSelectionVisuals();
        this._updateBulkActionsBar();
    },

    /**
     * Update selection visuals on all cards
     */
    _updateSelectionVisuals: function() {
        const cards = this._imageCardsContainer.querySelectorAll('.image-card');
        cards.forEach(card => {
            const id = card.dataset.id;
            const isSelected = this._selectedImages.has(id);
            
            if (isSelected) {
                Q(card).addClass('selected');
            } else {
                Q(card).removeClass('selected');
            }
        });
    },

    /**
     * Setup image zoom with CTRL + scroll and pan with CTRL + drag
     * Uses shared state to work with global mouse handlers
     * Loads saved bounds and auto-saves changes with debounce
     */
    _setupImageZoom: function(container, img, image) {
        // Per-image zoom state stored on container
        // Initialize from saved bounds if available
        const savedBounds = image.bounds || {};
        const savedZoom = savedBounds.zoom || 1;
        const savedCenterX = savedBounds.center_x !== undefined ? savedBounds.center_x : 0.5;
        const savedCenterY = savedBounds.center_y !== undefined ? savedBounds.center_y : 0.5;
        
        // Convert center offset to pan pixels (relative to container size)
        // center_x/y of 0.5 means centered (no pan), 0 means full left/top, 1 means full right/bottom
        const containerSize = 512; // Fixed container size
        const initialPanX = (0.5 - savedCenterX) * containerSize;
        const initialPanY = (0.5 - savedCenterY) * containerSize;
        
        container._zoomState = {
            zoom: savedZoom,
            panX: initialPanX,
            panY: initialPanY,
            imageId: image.relative_path
        };
        
        // Apply initial transform if zoomed
        if (savedZoom > 1 || savedCenterX !== 0.5 || savedCenterY !== 0.5) {
            img.style.transform = `scale(${savedZoom}) translate(${initialPanX}px, ${initialPanY}px)`;
            if (savedZoom > 1) {
                container.classList.add('zoomed');
            }
        }
        
        // Debounced save function
        let saveTimeout = null;
        const saveZoomState = () => {
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {
                const state = container._zoomState;
                // Convert pan pixels back to center_x/center_y (0-1 range)
                const center_x = 0.5 - (state.panX / containerSize);
                const center_y = 0.5 - (state.panY / containerSize);
                
                // Clamp values
                const bounds = {
                    zoom: Math.round(state.zoom * 100) / 100,
                    center_x: Math.max(0, Math.min(1, Math.round(center_x * 1000) / 1000)),
                    center_y: Math.max(0, Math.min(1, Math.round(center_y * 1000) / 1000))
                };
                
                // Save to backend
                API.datasetEditor.updateBounds(this._projectName, state.imageId, bounds)
                    .catch(err => console.error('Failed to save zoom/pan:', err));
                
                // Update local image data
                if (image.bounds) {
                    image.bounds.zoom = bounds.zoom;
                    image.bounds.center_x = bounds.center_x;
                    image.bounds.center_y = bounds.center_y;
                }
            }, 1000); // 1 second debounce
        };

        // Zoom on CTRL + scroll
        Q(container).on('wheel', (e) => {
            if (!e.ctrlKey) return;
            e.preventDefault();
            e.stopPropagation();

            const state = container._zoomState;
            const delta = e.deltaY > 0 ? -0.15 : 0.15;
            state.zoom = Math.max(1, Math.min(5, state.zoom + delta));

            // Reset pan when zoom returns to 1
            if (state.zoom === 1) {
                state.panX = 0;
                state.panY = 0;
            }

            img.style.transform = `scale(${state.zoom}) translate(${state.panX}px, ${state.panY}px)`;
            
            // Add zoomed class for visual feedback
            if (state.zoom > 1) {
                container.classList.add('zoomed');
            } else {
                container.classList.remove('zoomed');
            }
            
            // Save after debounce
            saveZoomState();
        });

        // Pan on CTRL + mousedown (when zoomed)
        Q(container).on('mousedown', (e) => {
            const state = container._zoomState;
            if (!e.ctrlKey || state.zoom <= 1) return;
            
            e.preventDefault();
            e.stopPropagation();

            // Set global active zoom state for document-level handlers
            this._activeZoom = {
                isPanning: true,
                zoom: state.zoom,
                panX: state.panX,
                panY: state.panY,
                startX: e.clientX - (state.panX * state.zoom),
                startY: e.clientY - (state.panY * state.zoom),
                img: img,
                container: container,
                state: state,
                saveCallback: saveZoomState
            };

            container.style.cursor = 'grabbing';
        });

        // Double-click to reset zoom
        Q(container).on('dblclick', (e) => {
            if (!e.ctrlKey) return;
            e.preventDefault();
            
            const state = container._zoomState;
            state.zoom = 1;
            state.panX = 0;
            state.panY = 0;
            img.style.transform = '';
            container.classList.remove('zoomed');
            
            // Save reset state
            saveZoomState();
        });

        // Prevent drag during zoom
        Q(container).on('dragstart', (e) => {
            if (container._zoomState.zoom > 1 || e.ctrlKey) {
                e.preventDefault();
            }
        });
    },

    /**
     * Create annotation editor based on dataset type
     */
    _createAnnotationEditor: function(image) {
        if (this._datasetType === 'folder_classification') {
            // No editor for folder classification
            return null;
        }

        const container = Q('<div>', { class: 'annotation-editor' }).get(0);

        if (this._datasetType === 'multi_label' || this._datasetType === 'mixed') {
            // Tag editor using TagInput widget
            const tagInput = new TagInput('tags-' + image.id, {
                placeholder: lang('dataset_page.tag_placeholder'),
                suggestions: this._allTags,
                recentTags: this._recentTags,
                allowCustom: true
            });
            
            // Set existing tags
            tagInput.set(image.tags);
            
            // Handle changes
            let saveTimeout = null;
            tagInput.onChange((tags, added, removed) => {
                // Update recent tags if a tag was added
                if (added) {
                    this._addToRecentTags(added);
                }
                
                // Debounced save
                clearTimeout(saveTimeout);
                saveTimeout = setTimeout(() => {
                    const newAnnotation = tags.join(', ');
                    this._saveAnnotation(image, newAnnotation);
                }, 300);
            });
            
            container.appendChild(tagInput.getElement());
        } else if (this._datasetType === 'annotation') {
            // Textarea widget for free-form annotation
            const textarea = new Textarea('annotation-' + image.id, {
                placeholder: lang('dataset_page.annotation_placeholder'),
                rows: 3,
                resize: true
            });
            
            textarea.set(image.annotation || '');
            
            let saveTimeout = null;
            textarea.onChange((value) => {
                clearTimeout(saveTimeout);
                saveTimeout = setTimeout(() => {
                    this._saveAnnotation(image, value);
                }, 500);
            });

            container.appendChild(textarea.getElement());
        }

        return container;
    },

    /**
     * Add tag to recent tags
     */
    _addToRecentTags: function(tag) {
        const normalized = tag.toLowerCase();
        this._recentTags = this._recentTags.filter(t => t !== normalized);
        this._recentTags.unshift(normalized);
        if (this._recentTags.length > 20) {
            this._recentTags = this._recentTags.slice(0, 20);
        }
        // Add to all tags if not present
        if (!this._allTags.includes(normalized)) {
            this._allTags.push(normalized);
        }
    },

    /**
     * Load all unique tags from the project
     */
    _loadAllTags: async function() {
        try {
            // Get stats which includes label distribution
            const stats = await API.datasetEditor.getStats(this._projectName);
            if (stats && stats.label_distribution) {
                this._allTags = Object.keys(stats.label_distribution).sort();
            }
            
            // Update bulk tag widgets with new suggestions
            if (this._bulkAddTagsWidget) {
                this._bulkAddTagsWidget.setSuggestions(this._allTags);
                this._bulkAddTagsWidget.setRecentTags(this._recentTags);
            }
            if (this._bulkRemoveTagsWidget) {
                this._bulkRemoveTagsWidget.setSuggestions(this._allTags);
                this._bulkRemoveTagsWidget.setRecentTags(this._recentTags);
            }
        } catch (err) {
            console.error('Failed to load tags:', err);
            this._allTags = [];
        }
    },

    /**
     * Save annotation
     */
    _saveAnnotation: async function(image, content) {
        try {
            const updated = await API.datasetEditor.updateAnnotation(
                this._projectName, 
                image.relative_path, 
                content
            );
            
            // Update local cache
            const idx = this._images.findIndex(i => i.id === image.id);
            if (idx !== -1) {
                this._images[idx] = updated;
                // Re-render just this card
                const cards = this._imageCardsContainer.querySelectorAll('.image-card');
                if (cards[idx]) {
                    const newCard = this._createImageCard(updated);
                    cards[idx].replaceWith(newCard);
                }
            }
        } catch (err) {
            console.error('Failed to save annotation:', err);
        }
    },

    /**
     * Delete image
     */
    _deleteImage: async function(image) {
        const confirmed = await Modal.confirm(lang('dataset_page.confirm_delete'));
        if (!confirmed) return;

        try {
            await API.datasetEditor.deleteItems(this._projectName, [image.relative_path]);
            this._loadImages();
            this._loadFolderTree();
        } catch (err) {
            console.error('Failed to delete image:', err);
        }
    },

    /**
     * Upload files
     */
    _uploadFiles: async function(files) {
        if (!files || files.length === 0) return;
        
        try {
            ProgressManager.show('upload', {
                label: lang('dataset_page.uploading'),
                progress: -1,
                status: `${files.length} file(s)`
            });
            
            const targetFolder = this._currentFolder || '';
            const result = await API.datasetEditor.uploadImages(this._projectName, files, targetFolder);
            
            ProgressManager.hide('upload');
            
            if (result.uploaded > 0) {
                // Refresh folder tree and images
                await this._loadFolderTree();
                await this._loadImages();
            }
            
            if (result.failed > 0) {
                Modal.alert(`Uploaded: ${result.uploaded}, Failed: ${result.failed}\n${result.errors.join('\n')}`);
            }
        } catch (err) {
            console.error('Failed to upload files:', err);
            ProgressManager.hide('upload');
            Modal.alert(lang('dataset_page.upload_error') + ': ' + err.message);
        }
    },

    /**
     * Create folder
     * @param {string} [parentPath] - Optional parent folder path (from context menu)
     */
    _createFolder: async function(parentPath) {
        const name = await Modal.prompt(lang('dataset_page.enter_folder_name'));
        if (!name || !name.trim()) return;
        
        // Determine path: use parentPath if provided, else current folder, else root
        const basePath = parentPath !== undefined ? parentPath : this._currentFolder;
        const path = basePath
            ? `${basePath}/${name.trim()}`
            : name.trim();
        
        try {
            const result = await API.datasetEditor.createFolder(this._projectName, path);
            
            if (result.success) {
                await this._loadFolderTree();
                // Select the newly created folder
                this._selectFolder(result.path);
            } else {
                Modal.alert(result.message);
            }
        } catch (err) {
            console.error('Failed to create folder:', err);
            Modal.alert(lang('dataset_page.folder_create_error') + ': ' + err.message);
        }
    },

    /**
     * Rename folder
     */
    _renameFolder: async function(node) {
        const newName = await Modal.prompt(lang('dataset_page.enter_new_name'), node.name);
        if (!newName || !newName.trim() || newName.trim() === node.name) return;
        
        try {
            const result = await API.datasetEditor.renameFolder(this._projectName, node.path, newName.trim());
            
            if (result.success) {
                // Update current folder selection if it was renamed
                if (this._currentFolder === node.path) {
                    this._currentFolder = result.path;
                } else if (this._currentFolder && this._currentFolder.startsWith(node.path + '/')) {
                    this._currentFolder = result.path + this._currentFolder.slice(node.path.length);
                }
                await this._loadFolderTree();
                await this._loadImages();
            } else {
                Modal.alert(result.message);
            }
        } catch (err) {
            console.error('Failed to rename folder:', err);
            Modal.alert(lang('dataset_page.folder_rename_error') + ': ' + err.message);
        }
    },

    /**
     * Delete folder
     */
    _deleteFolder: async function(node) {
        const hasImages = node.image_count > 0;
        const message = hasImages 
            ? lang('dataset_page.delete_folder_confirm_with_images', { count: node.image_count })
            : lang('dataset_page.delete_folder_confirm');
        
        const confirmed = await Modal.confirm(message);
        if (!confirmed) return;
        
        try {
            const result = await API.datasetEditor.deleteFolder(this._projectName, node.path, hasImages);
            
            if (result.success) {
                // Reset folder selection if it was deleted
                if (this._currentFolder === node.path || 
                    (this._currentFolder && this._currentFolder.startsWith(node.path + '/'))) {
                    this._currentFolder = null;
                }
                await this._loadFolderTree();
                await this._loadImages();
            } else {
                Modal.alert(result.message);
            }
        } catch (err) {
            console.error('Failed to delete folder:', err);
            Modal.alert(lang('dataset_page.folder_delete_error') + ': ' + err.message);
        }
    },

    /**
     * Start sync (discovery)
     */
    _startSync: async function() {
        try {
            Q(this._syncBtn).addClass('loading');
            
            // Show progress bar
            ProgressManager.show('sync', {
                label: lang('dataset_page.sync_button'),
                progress: 0,
                status: lang('dataset_page.progress_starting')
            });

            await API.datasetEditor.refresh(this._projectName);
            this._pollSyncStatus();
        } catch (err) {
            console.error('Failed to start sync:', err);
            ProgressManager.hide('sync');
            Q(this._syncBtn).removeClass('loading');
        }
    },

    /**
     * Poll sync status
     */
    _pollSyncStatus: async function() {
        const pollInterval = 500;
        
        const poll = async () => {
            try {
                const response = await fetch(`${API.baseUrl}/dataset/editor/projects/${encodeURIComponent(this._projectName)}/refresh/status`);
                const status = await response.json();
                
                if (status.status === 'running' || status.status === 'pending') {
                    const progress = status.total_items > 0 
                        ? Math.round((status.processed_items / status.total_items) * 100)
                        : 0;
                    
                    ProgressManager.update('sync', progress, `${status.processed_items}/${status.total_items}`);
                    setTimeout(poll, pollInterval);
                } else {
                    // Complete
                    ProgressManager.update('sync', 100, lang('dataset_page.progress_complete'));
                    setTimeout(() => {
                        ProgressManager.hide('sync');
                        Q(this._syncBtn).removeClass('loading');
                        this._loadFolderTree();
                        this._loadImages();
                    }, 1000);
                }
            } catch (err) {
                console.error('Poll error:', err);
                ProgressManager.hide('sync');
                Q(this._syncBtn).removeClass('loading');
            }
        };
        
        poll();
    },

    /**
     * Start build
     */
    _startBuild: async function() {
        try {
            Q(this._buildBtn).attr('disabled', true);
            
            // Show progress bar
            ProgressManager.show('build', {
                label: lang('dataset_page.build_button'),
                progress: 0,
                status: lang('dataset_page.progress_starting')
            });

            await API.datasetEditor.build(this._projectName);
            this._pollBuildStatus();
        } catch (err) {
            console.error('Failed to start build:', err);
            ProgressManager.hide('build');
            Q(this._buildBtn).attr('disabled', false);
        }
    },

    /**
     * Poll build status
     */
    _pollBuildStatus: async function() {
        const pollInterval = 500;
        
        const poll = async () => {
            try {
                const status = await API.datasetEditor.getBuildStatus(this._projectName);
                
                if (status.status === 'running' || status.status === 'pending') {
                    const progress = status.total_items > 0 
                        ? Math.round((status.completed_items / status.total_items) * 100)
                        : 0;
                    
                    ProgressManager.update('build', progress, `${status.completed_items}/${status.total_items}`);
                    setTimeout(poll, pollInterval);
                } else {
                    // Complete
                    ProgressManager.update('build', 100, lang('dataset_page.progress_complete'));
                    
                    setTimeout(() => {
                        ProgressManager.hide('build');
                        Q(this._buildBtn).attr('disabled', false);
                        
                        if (status.status === 'success') {
                            // Show success notification (non-blocking)
                            console.log('Build complete:', status.result);
                        } else if (status.status === 'error') {
                            Modal.alert(lang('dataset_page.build_error') + ': ' + (status.last_error || 'Unknown error'));
                        }
                    }, 1000);
                }
            } catch (err) {
                console.error('Poll error:', err);
                ProgressManager.hide('build');
                Q(this._buildBtn).attr('disabled', false);
            }
        };
        
        poll();
    },

    /**
     * Update status bar
     */
    _updateStatusBar: function(totalItems) {
        this._statusCount.textContent = lang('dataset_page.status_count', { count: totalItems });
    },

    /**
     * Update pagination controls
     */
    _renderPagination: function() {
        // Update prev/next button states
        this._prevBtn.setDisabled(this._currentPage <= 1);
        this._nextBtn.setDisabled(this._currentPage >= this._totalPages);

        // Update page dropdown options
        const pageOptions = [];
        const pageLabels = {};
        for (let i = 1; i <= this._totalPages; i++) {
            const str = String(i);
            pageOptions.push(str);
            pageLabels[str] = str;
        }

        // Rebuild dropdown if pages changed
        if (pageOptions.length > 0) {
            this._pageDropdown.setOptions(pageOptions, pageLabels);
            this._pageDropdown.set(String(this._currentPage));
        }

        // Update total pages label
        this._totalPagesLabel.textContent = `/ ${this._totalPages}`;
    },

    /**
     * Refresh image grid
     */
    _refreshImageGrid: function() {
        this._loadImages();
    },

    /**
     * Build bulk actions bar
     */
    _buildBulkActionsBar: function() {
        const bar = Q('<div>', { class: 'bulk-actions-bar hidden' }).get(0);

        // Selection info section
        const selectionInfo = Q('<div>', { class: 'bulk-selection-info' }).get(0);
        this._bulkSelectionCount = Q('<span>', { class: 'selection-count' }).get(0);
        selectionInfo.appendChild(this._bulkSelectionCount);
        
        // Clear selection button
        this._clearSelectionBtn = new ActionButton('bulk-clear', {
            label: lang('dataset_page.clear_selection'),
            className: 'btn btn-ghost btn-sm',
            onClick: () => this._clearSelection()
        });
        selectionInfo.appendChild(this._clearSelectionBtn.getElement());
        
        // Select all button
        this._selectAllBtn = new ActionButton('bulk-select-all', {
            label: lang('dataset_page.select_all'),
            className: 'btn btn-ghost btn-sm',
            onClick: () => this._selectAll()
        });
        selectionInfo.appendChild(this._selectAllBtn.getElement());
        
        bar.appendChild(selectionInfo);

        // Bulk tag actions section
        const tagActions = Q('<div>', { class: 'bulk-tag-actions' }).get(0);
        
        // Add tags group
        const addTagsGroup = Q('<div>', { class: 'bulk-tag-group' }).get(0);
        
        this._bulkAddTagsWidget = new TagInput('bulk-add-tags', {
            placeholder: lang('dataset_page.bulk_add_tags_placeholder'),
            suggestions: this._allTags,
            recentTags: this._recentTags,
            allowCustom: true
        });
        addTagsGroup.appendChild(this._bulkAddTagsWidget.getElement());
        
        this._bulkAddBtn = new ActionButton('bulk-add-btn', {
            label: lang('dataset_page.bulk_add'),
            className: 'btn btn-primary btn-sm',
            onClick: () => this._bulkAddTags()
        });
        addTagsGroup.appendChild(this._bulkAddBtn.getElement());
        tagActions.appendChild(addTagsGroup);

        // Remove tags group
        const removeTagsGroup = Q('<div>', { class: 'bulk-tag-group' }).get(0);
        
        this._bulkRemoveTagsWidget = new TagInput('bulk-remove-tags', {
            placeholder: lang('dataset_page.bulk_remove_tags_placeholder'),
            suggestions: this._allTags,
            recentTags: this._recentTags,
            allowCustom: true
        });
        removeTagsGroup.appendChild(this._bulkRemoveTagsWidget.getElement());
        
        this._bulkRemoveBtn = new ActionButton('bulk-remove-btn', {
            label: lang('dataset_page.bulk_remove'),
            className: 'btn btn-secondary btn-sm',
            onClick: () => this._bulkRemoveTags()
        });
        removeTagsGroup.appendChild(this._bulkRemoveBtn.getElement());
        tagActions.appendChild(removeTagsGroup);

        bar.appendChild(tagActions);

        // Bulk delete button
        this._bulkDeleteBtn = new ActionButton('bulk-delete-btn', {
            label: lang('dataset_page.bulk_delete'),
            className: 'btn btn-error btn-sm',
            onClick: () => this._bulkDelete()
        });
        bar.appendChild(this._bulkDeleteBtn.getElement());

        return bar;
    },

    /**
     * Update bulk actions bar visibility
     */
    _updateBulkActionsBar: function() {
        const count = this._selectedImages.size;
        
        if (count > 0) {
            this._bulkSelectionCount.textContent = lang('dataset_page.selected_count', { count });
            Q(this._bulkActionsBar).removeClass('hidden');
        } else {
            Q(this._bulkActionsBar).addClass('hidden');
        }
    },

    /**
     * Clear selection
     */
    _clearSelection: function() {
        this._selectedImages.clear();
        this._lastSelectedIndex = -1;
        this._updateSelectionVisuals();
        this._updateBulkActionsBar();
    },

    /**
     * Select all images on current page
     */
    _selectAll: function() {
        this._images.forEach(image => {
            this._selectedImages.add(image.id);
        });
        this._updateSelectionVisuals();
        this._updateBulkActionsBar();
    },

    /**
     * Bulk add tags
     */
    _bulkAddTags: async function() {
        const tags = this._bulkAddTagsWidget.get();
        if (tags.length === 0) return;
        
        // Determine target: selection or current filter
        const items = this._selectedImages.size > 0 
            ? Array.from(this._selectedImages)
            : null;
        
        try {
            const params = {
                add: tags,
                remove: [],
                items: items,
                folder: items ? null : this._currentFolder,
                search: items ? null : this._searchQuery
            };
            
            await API.datasetEditor.bulkTags(this._projectName, params);
            
            // Add to recent tags
            tags.forEach(t => this._addToRecentTags(t));
            
            // Clear and refresh
            this._bulkAddTagsWidget.clear();
            await this._loadImages();
            await this._loadAllTags();
        } catch (err) {
            console.error('Bulk add tags failed:', err);
            Modal.alert(lang('dataset_page.bulk_error'));
        }
    },

    /**
     * Bulk remove tags
     */
    _bulkRemoveTags: async function() {
        const tags = this._bulkRemoveTagsWidget.get();
        if (tags.length === 0) return;
        
        // Determine target: selection or current filter
        const items = this._selectedImages.size > 0 
            ? Array.from(this._selectedImages)
            : null;
        
        try {
            const params = {
                add: [],
                remove: tags,
                items: items,
                folder: items ? null : this._currentFolder,
                search: items ? null : this._searchQuery
            };
            
            await API.datasetEditor.bulkTags(this._projectName, params);
            
            // Clear and refresh
            this._bulkRemoveTagsWidget.clear();
            await this._loadImages();
            await this._loadAllTags();
        } catch (err) {
            console.error('Bulk remove tags failed:', err);
            Modal.alert(lang('dataset_page.bulk_error'));
        }
    },

    /**
     * Bulk delete selected images
     */
    _bulkDelete: async function() {
        const count = this._selectedImages.size;
        if (count === 0) return;
        
        const confirmed = await Modal.confirm(lang('dataset_page.bulk_delete_confirm', { count }));
        if (!confirmed) return;
        
        try {
            const items = Array.from(this._selectedImages);
            await API.datasetEditor.deleteItems(this._projectName, items);
            
            this._selectedImages.clear();
            await this._loadFolderTree();
            await this._loadImages();
        } catch (err) {
            console.error('Bulk delete failed:', err);
            Modal.alert(lang('dataset_page.bulk_error'));
        }
    }
};

// Register page
Pages.register('dataset', DatasetPage);
