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

        // Get project info and dataset type (before building layout)
        await this._loadProjectInfo();

        // Build layout (creates the dropdown)
        this._buildLayout();

        // Now set dropdown value from loaded dataset type
        if (this._typeDropdown && this._datasetType) {
            this._typeDropdown.set(this._datasetType);
        }

        // Load initial data
        await this._loadFolderTree();
        await this._loadImages();
    },

    /**
     * Show no project selected message
     */
    _showNoProjectMessage: function() {
        const btn = Q('<button>', { 
            class: 'btn btn-primary',
            text: lang('dataset_page.no_project.button')
        }).on('click', () => {
            if (typeof Navigation !== 'undefined') {
                Navigation.navigateTo('projects');
            }
        }).get(0);
        
        Q('<div>', { class: 'no-project-message' })
            .append(Q('<div>', { class: 'no-project-icon', text: '!' }).get(0))
            .append(Q('<h2>', { text: lang('dataset_page.no_project.title') }).get(0))
            .append(Q('<p>', { text: lang('dataset_page.no_project.description') }).get(0))
            .append(btn)
            .appendTo(this._container);
    },

    /**
     * Load project info and dataset type
     */
    _loadProjectInfo: async function() {
        try {
            // Get dataset type from DB (user-set, not auto-detected)
            const typeData = await API.datasetEditor.getDatasetType(this._projectName);
            this._datasetType = typeData.type || 'unknown';
            
            // Update dropdown if it exists
            if (this._typeDropdown) {
                this._typeDropdown.set(this._datasetType);
            }
        } catch (err) {
            console.error('Failed to load project info:', err);
            this._datasetType = 'unknown';
        }
    },

    /**
     * Build main layout
     */
    _buildLayout: function() {
        // Toolbar (includes pagination and status)
        this._toolbar = this._buildToolbar();
        
        // Folder browser (left sidebar)
        this._folderBrowser = this._buildFolderBrowser();
        
        // Image grid (main area)
        this._imageGrid = this._buildImageGrid();
        
        // Bulk actions bar (hidden by default)
        this._bulkActionsBar = this._buildBulkActionsBar();

        // Build layout with Q chaining
        Q('<div>', { class: 'dataset-page' })
            .append(this._toolbar)
            .append(
                Q('<div>', { class: 'dataset-content' })
                    .append(this._folderBrowser)
                    .append(this._imageGrid)
                    .get(0)
            )
            .append(this._bulkActionsBar)
            .appendTo(this._container);

        // Load all tags for suggestions
        this._loadAllTags();

        // Setup keyboard shortcuts
        this._setupKeyboardShortcuts();

        // Setup context menu for image cards
        this._setupContextMenu();

        // Setup global zoom/pan event handlers
        this._setupGlobalZoomPanHandlers();

        // Load images without auto-discovery (user must click Sync)
        this._loadFolderTree();
        this._loadImages();
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
            
            // Root folder: only show "New Folder" option
            if (!folderPath) {
                return [
                    {
                        label: lang('dataset_page.new_folder'),
                        icon: 'folder_closed.svg',
                        action: () => this._createFolder()
                    }
                ];
            }
            
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
     * Apply zoom/pan transform - MUST match backend compute_crop_box exactly
     * 
     * Backend formula:
     *   crop_size = min(imgWidth, imgHeight) / zoom
     *   crop_center = (centerX * imgWidth, centerY * imgHeight)
     * 
     * Frontend goal:
     *   Show exactly what will be cropped - the crop area fills the container
     *   Container is the "window", image moves/scales behind it
     * 
     * IMPORTANT: imgW/imgH come from DATABASE, not from naturalWidth/naturalHeight
     */
    _applyZoomTransform: function(img, container, state) {
        const imgW = state.imgWidth;
        const imgH = state.imgHeight;
        
        if (!imgW || !imgH) {
            img.style.transform = 'none';
            return;
        }
        
        const rect = container.getBoundingClientRect();
        const containerSize = rect.width || 300; // Container is square
        
        const minDim = Math.min(imgW, imgH);
        
        // Backend: crop_size = min(w,h) / zoom (in original image pixels)
        const cropSize = minDim / state.zoom;
        
        // Scale factor: how many container pixels per original image pixel
        // The crop area (cropSize x cropSize original pixels) fills the container
        const scale = containerSize / cropSize;
        
        // Crop center in original image pixels
        const cropCenterX = state.centerX * imgW;
        const cropCenterY = state.centerY * imgH;
        
        // After scaling, crop center will be at: cropCenterX * scale, cropCenterY * scale
        // We want this point to be at container center (containerSize/2, containerSize/2)
        // Image top-left should be at: containerCenter - scaledCropCenter
        const imgLeft = containerSize / 2 - cropCenterX * scale;
        const imgTop = containerSize / 2 - cropCenterY * scale;
        
        // Apply: translate positions image top-left, scale enlarges from top-left (transform-origin: 0 0)
        img.style.transform = `translate(${imgLeft}px, ${imgTop}px) scale(${scale})`;
    },

    /**
     * Calculate minimum zoom
     * At zoom=1, crop_size = min(w,h) which is the largest square that fits
     * This is always valid, so minZoom = 1
     */
    _calculateMinZoom: function() {
        return 1;
    },

    /**
     * Constrain zoom to valid range
     */
    _constrainZoom: function(zoom, minZoom, maxZoom = 10) {
        return Math.max(minZoom, Math.min(maxZoom, zoom));
    },

    /**
     * Constrain center so crop area stays within image bounds
     * 
     * Backend: crop is a square of size (min(w,h) / zoom) centered at (centerX*w, centerY*h)
     * Center must be at least halfCrop from each edge
     * 
     * IMPORTANT: Uses imgW/imgH from state (database), not naturalWidth/naturalHeight
     */
    _constrainCenter: function(centerX, centerY, zoom, state) {
        const imgW = state.imgWidth;
        const imgH = state.imgHeight;
        
        if (!imgW || !imgH) return { x: 0.5, y: 0.5 };
        
        const minDim = Math.min(imgW, imgH);
        
        // Crop size in original pixels
        const cropSize = minDim / zoom;
        const halfCrop = cropSize / 2;
        
        // Center (in pixels) must be at least halfCrop from edges
        // centerX * imgW >= halfCrop  =>  centerX >= halfCrop / imgW
        // centerX * imgW <= imgW - halfCrop  =>  centerX <= 1 - halfCrop / imgW
        const minCenterX = halfCrop / imgW;
        const maxCenterX = 1 - halfCrop / imgW;
        const minCenterY = halfCrop / imgH;
        const maxCenterY = 1 - halfCrop / imgH;
        
        return {
            x: Math.max(minCenterX, Math.min(maxCenterX, centerX)),
            y: Math.max(minCenterY, Math.min(maxCenterY, centerY))
        };
    },

    /**
     * Setup global zoom/pan event handlers (for mouse move/up during pan)
     */
    _setupGlobalZoomPanHandlers: function() {
        // Mouse move handler for panning
        this._panMoveHandler = (e) => {
            if (!this._activeZoom || !this._activeZoom.isPanning) return;
            
            const { startX, startY, startCenterX, startCenterY, img, state, container } = this._activeZoom;
            
            const rect = container.getBoundingClientRect();
            const containerSize = rect.width || 300;
            
            // Use database dimensions from state
            const imgW = state.imgWidth;
            const imgH = state.imgHeight;
            const minDim = Math.min(imgW, imgH);
            
            // Same scale calculation as _applyZoomTransform
            const cropSize = minDim / state.zoom;
            const scale = containerSize / cropSize;
            
            // Mouse delta in screen pixels
            const deltaScreenX = e.clientX - startX;
            const deltaScreenY = e.clientY - startY;
            
            // Convert to original image pixels
            const deltaImgX = deltaScreenX / scale;
            const deltaImgY = deltaScreenY / scale;
            
            // Convert to normalized coords (0-1)
            const deltaNormX = deltaImgX / imgW;
            const deltaNormY = deltaImgY / imgH;
            
            // Apply delta (negative: dragging image right = crop center moves left)
            let centerX = startCenterX - deltaNormX;
            let centerY = startCenterY - deltaNormY;
            
            // Constrain to valid range
            const constrained = this._constrainCenter(centerX, centerY, state.zoom, state);
            state.centerX = constrained.x;
            state.centerY = constrained.y;
            
            this._applyZoomTransform(img, container, state);
        };

        // Mouse up handler to end panning
        this._panUpHandler = () => {
            if (!this._activeZoom) return;
            this._activeZoom.isPanning = false;
            if (this._activeZoom.container) {
                this._activeZoom.container.style.cursor = '';
            }
            if (this._activeZoom.saveCallback) {
                this._activeZoom.saveCallback();
            }
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
        const headerActions = Q('#header-actions').get(0);
        if (!headerActions) return;

        // Sync button using ActionButton widget (icon only)
        this._syncBtnWidget = new ActionButton('dataset-sync', {
            label: '',
            className: 'btn btn-secondary btn-icon',
            onClick: () => this._startSync()
        });
        this._syncBtn = this._syncBtnWidget.getElement();
        this._syncBtn.title = lang('dataset_page.sync_button');
        Q(this._syncBtn).append(
            Q('<img>', { src: '/static/icons/sync.svg', alt: 'Sync', class: 'btn-icon-img' }).get(0)
        );

        // Build button using ActionButton widget
        this._buildBtnWidget = new ActionButton('dataset-build', {
            label: lang('dataset_page.build_button'),
            className: 'btn btn-primary',
            onClick: () => this._startBuild()
        });
        this._buildBtn = this._buildBtnWidget.getElement();
        
        // Add buttons to header
        Q(headerActions)
            .append(this._syncBtn)
            .append(this._buildBtn);
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
        // Left section - dataset type selector
        const leftSection = Q('<div>', { class: 'toolbar-left' })
            .append(this._buildDatasetTypeSelector())
            .get(0);

        // Center section - search
        const centerSection = Q('<div>', { class: 'toolbar-center' })
            .append(this._buildSearchBox())
            .get(0);

        // Right section - pagination and status
        this._statusCount = Q('<span>', { class: 'status-count' }).get(0);
        this._totalPagesLabel = Q('<span>', { class: 'total-pages', text: '/ 1' }).get(0);
        
        // Pagination buttons
        this._prevBtn = new ActionButton('page-prev', {
            label: '<',
            className: 'btn btn-sm btn-secondary',
            onClick: () => this._goToPage(this._currentPage - 1)
        });
        
        this._nextBtn = new ActionButton('page-next', {
            label: '>',
            className: 'btn btn-sm btn-secondary',
            onClick: () => this._goToPage(this._currentPage + 1)
        });

        // Page dropdown
        this._pageDropdown = new Dropdown('page-select', {
            options: ['1'],
            optionLabels: { '1': '1' },
            default: '1'
        });
        this._pageDropdown.onChange((value) => this._goToPage(parseInt(value, 10)));
        const pageDropdownEl = this._pageDropdown.getElement();
        pageDropdownEl.classList.add('pagination-dropdown');

        const paginationControls = Q('<div>', { class: 'pagination-controls' })
            .append(this._prevBtn.getElement())
            .append(pageDropdownEl)
            .append(this._totalPagesLabel)
            .append(this._nextBtn.getElement())
            .get(0);

        // Page size dropdown
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

        const pageSizeGroup = Q('<div>', { class: 'page-size-group' })
            .append(Q('<span>', { class: 'page-size-label', text: lang('dataset_page.per_page') }).get(0))
            .append(pageSizeEl)
            .get(0);

        const rightSection = Q('<div>', { class: 'toolbar-right' })
            .append(this._statusCount)
            .append(paginationControls)
            .append(pageSizeGroup)
            .get(0);

        // Build toolbar
        return Q('<div>', { class: 'dataset-toolbar' })
            .append(leftSection)
            .append(centerSection)
            .append(rightSection)
            .get(0);
    },

    /**
     * Build dataset type selector using Dropdown widget
     */
    _buildDatasetTypeSelector: function() {
        const types = ['unknown', 'multi_label', 'folder_classification', 'annotation'];
        const optionLabels = {
            'unknown': lang('dataset_page.types.unknown'),
            'multi_label': lang('dataset_page.types.multi_label'),
            'folder_classification': lang('dataset_page.types.folder_classification'),
            'annotation': lang('dataset_page.types.annotation')
        };

        this._typeDropdown = new Dropdown('dataset-type', {
            label: lang('dataset_page.type_label'),
            options: types,
            optionLabels: optionLabels,
            default: 'unknown'
        });

        this._typeDropdown.onChange((value) => {
            // Save dataset type to database
            API.datasetEditor.setDatasetType(this._projectName, value).catch(err => {
                console.error('Failed to save dataset type:', err);
            });
            
            this._datasetType = value;
            this._refreshImageGrid();
        });

        return Q('<div>', { class: 'dataset-type-selector' })
            .append(this._typeDropdown.getElement())
            .get(0);
    },

    /**
     * Build search box using TextInput and Dropdown widgets
     */
    _buildSearchBox: function() {
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

        const dropdownEl = this._searchModeDropdown.getElement();
        dropdownEl.classList.add('search-mode-widget');

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

        // Show Duplicates toggle button
        this._dupToggleBtn = Q('<button>', { 
            class: 'btn btn-secondary btn-icon duplicate-toggle',
            title: lang('dataset_page.show_duplicates')
        }).on('click', async () => {
            if (this._showDuplicatesOnly) {
                // Turn off duplicate filter
                this._showDuplicatesOnly = false;
                this._duplicateIds.clear();
                this._dupToggleBtn.classList.remove('active');
                this._currentPage = 1;
                await this._loadImages();
            } else {
                // Scan for duplicates and show them
                await this._loadDuplicatesFilter();
            }
        }).get(0);
        this._dupToggleBtn.innerHTML = '<img src="/static/icons/copy.svg" alt="Duplicates" class="btn-icon-img">';

        return Q('<div>', { class: 'search-container' })
            .append(dropdownEl)
            .append(inputEl)
            .append(this._dupToggleBtn)
            .get(0);
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
        this._folderTreeContainer = Q('<div>', { class: 'folder-tree' }).get(0);
        
        return Q('<div>', { class: 'folder-browser' })
            .append(
                Q('<div>', { class: 'folder-header' })
                    .append(Q('<span>', { class: 'folder-title', text: lang('dataset_page.folders') }).get(0))
                    .get(0)
            )
            .append(this._folderTreeContainer)
            .get(0);
    },

    /**
     * Build image grid
     */
    _buildImageGrid: function() {
        // Upload zone (drag & drop overlay)
        this._uploadZone = Q('<div>', { class: 'upload-zone' }).get(0);
        this._uploadZone.innerHTML = `
            <div class="upload-zone-content">
                <img src="/static/icons/upload.svg" alt="Upload" class="upload-icon">
                <p>${lang('dataset_page.drop_images')}</p>
            </div>
        `;

        // Image cards container
        this._imageCardsContainer = Q('<div>', { class: 'image-cards' }).get(0);

        const grid = Q('<div>', { class: 'image-grid-container' })
            .append(this._uploadZone)
            .append(this._imageCardsContainer)
            .get(0);

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
            const isActive = node.path === this._currentFolder;
            
            const item = Q('<div>', { 
                class: 'folder-item' + (isActive ? ' active' : ''),
                'data-path': node.path
            })
            .css('paddingLeft', (depth * 16 + 8) + 'px')
            .append(
                Q('<img>', {
                    src: `/static/icons/${hasChildren ? 'folder_open' : 'folder_closed'}.svg`,
                    class: 'folder-icon'
                }).get(0)
            )
            .append(
                Q('<span>', { 
                    class: 'folder-name',
                    text: node.name || lang('dataset_page.root_folder')
                }).get(0)
            )
            .append(
                Q('<span>', { 
                    class: 'folder-count',
                    text: `(${node.image_count})`
                }).get(0)
            )
            .on('click', () => this._selectFolder(node.path))
            .appendTo(this._folderTreeContainer);

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
        if (this._imageCardsContainer) {
            this._imageCardsContainer.innerHTML = '';
        }
        
        this._lastSelectedIndex = -1;

        this._images.forEach((image, index) => {
            const card = this._createImageCard(image, index);
            this._imageCardsContainer.appendChild(card);

            const thumbContainer = card.querySelector('.image-thumb-container');
            const thumb = thumbContainer?.querySelector('.image-thumb');
            if (thumbContainer && thumb) {
                this._setupImageZoom(thumbContainer, thumb, image);
            }
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
        
        // Thumbnail
        const thumb = Q('<img>', {
            src: image.image_url,
            class: 'image-thumb',
            alt: image.filename,
            loading: 'lazy'
        }).get(0);

        // Delete button
        const deleteBtn = Q('<button>', { 
            class: 'image-delete-btn',
            title: lang('dataset_page.delete_image')
        }).on('click', (e) => {
            e.stopPropagation();
            this._deleteImage(image);
        }).get(0);
        deleteBtn.innerHTML = '&times;';

        // Thumbnail container
        const thumbContainer = Q('<div>', { class: 'image-thumb-container' })
            .append(thumb)
            .append(deleteBtn)
            .get(0);

        // Duplicate indicator - show if this image is in any duplicate group
        if (this._showDuplicatesOnly && this._duplicateIds.has(image.relative_path)) {
            let dupCount = 0;
            for (const [hash, paths] of this._duplicates) {
                if (paths.includes(image.relative_path)) {
                    dupCount = paths.length;
                    break;
                }
            }
            if (dupCount > 1) {
                Q(thumbContainer).append(
                    Q('<span>', { 
                        class: 'duplicate-indicator',
                        title: lang('dataset_page.duplicate_count', { count: dupCount }),
                        text: dupCount
                    }).get(0)
                );
            }
        }

        // Filename
        const filename = Q('<div>', { 
            class: 'image-filename',
            text: image.filename,
            title: image.relative_path
        }).get(0);

        // Check if bounds have been changed from defaults (zoom != 1 or center != 0.5)
        const bounds = image.bounds || {};
        const hasChangedBounds = (bounds.zoom && bounds.zoom !== 1) || 
                                 (bounds.center_x && bounds.center_x !== 0.5) || 
                                 (bounds.center_y && bounds.center_y !== 0.5);

        // Build card
        const card = Q('<div>', { 
            class: 'image-card' + (isSelected ? ' selected' : '') + (hasChangedBounds ? ' bounds-changed' : ''),
            'data-id': image.relative_path,
            'data-index': index
        })
        .append(thumbContainer)
        .append(filename)
        .on('click', (e) => {
            if (e.target.closest('button, input, .tag-remove, .tag-add-input')) return;
            this._handleImageSelect(image, index, e);
        })
        .get(0);

        // Annotation editor based on dataset type
        const editor = this._createAnnotationEditor(image);
        if (editor) {
            Q(card).append(editor);
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
     * Setup image zoom/pan - IMAGE EDITOR STYLE
     * User sees FULL image and zooms/pans to select what area goes into dataset
     * Container shows a square crop preview - what you see is what you get in training
     * 
     * IMPORTANT: Image dimensions come from DATABASE (image.width, image.height),
     * NOT from img.naturalWidth/naturalHeight
     */
    _setupImageZoom: function(container, img, image) {
        const savedBounds = image.bounds || {};
        const savedZoom = typeof savedBounds.zoom === 'number' ? savedBounds.zoom : 1;
        const savedCenterX = typeof savedBounds.center_x === 'number' ? savedBounds.center_x : 0.5;
        const savedCenterY = typeof savedBounds.center_y === 'number' ? savedBounds.center_y : 0.5;

        // Image dimensions FROM DATABASE
        const imgWidth = image.width;
        const imgHeight = image.height;

        const state = {
            zoom: savedZoom,
            centerX: savedCenterX,
            centerY: savedCenterY,
            imageId: image.relative_path,
            imgWidth: imgWidth,   // From database
            imgHeight: imgHeight, // From database
            minZoom: 1,
            initialized: false
        };
        container._zoomState = state;

        const initialize = () => {
            // Wait for image to load in DOM (for display), but use DB dimensions for calculations
            if (!img.complete) return;
            
            const rect = container.getBoundingClientRect();
            if (!rect.width || !rect.height) {
                requestAnimationFrame(initialize);
                return;
            }

            // minZoom is always 1 (crop = min dimension)
            state.minZoom = 1;
            
            // Restore saved zoom (constrained to min)
            state.zoom = Math.max(state.minZoom, savedZoom);
            
            // Constrain center coordinates using DB dimensions
            const constrained = this._constrainCenter(state.centerX, state.centerY, state.zoom, state);
            state.centerX = constrained.x;
            state.centerY = constrained.y;
            
            this._applyZoomTransform(img, container, state);
            state.initialized = true;
        };

        if (img.complete) {
            requestAnimationFrame(initialize);
        } else {
            img.addEventListener('load', () => requestAnimationFrame(initialize), { once: true });
        }

        // Save function
        const saveZoomState = () => {
            if (!state.initialized) return;

            const bounds = {
                zoom: Math.round(state.zoom * 100) / 100,
                center_x: Math.round(state.centerX * 1000) / 1000,
                center_y: Math.round(state.centerY * 1000) / 1000
            };

            API.datasetEditor.updateBounds(this._projectName, state.imageId, bounds)
                .catch(err => console.error('Failed to save bounds:', err));

            image.bounds = { ...bounds };
            
            // Update bounds-changed class on card
            const card = container.closest('.image-card');
            if (card) {
                const hasChanged = bounds.zoom !== 1 || bounds.center_x !== 0.5 || bounds.center_y !== 0.5;
                card.classList.toggle('bounds-changed', hasChanged);
            }
        };

        let saveTimeout = null;
        const debouncedSave = () => {
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(saveZoomState, 500);
        };

        // CTRL + Scroll = Zoom
        Q(container).on('wheel', (e) => {
            if (!e.ctrlKey || !state.initialized) return;
            e.preventDefault();
            e.stopPropagation();

            // minZoom is always 1
            state.minZoom = 1;
            
            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            state.zoom = this._constrainZoom(state.zoom + delta, state.minZoom, 10);
            
            // Constrain center after zoom change (using state with DB dimensions)
            const constrained = this._constrainCenter(state.centerX, state.centerY, state.zoom, state);
            state.centerX = constrained.x;
            state.centerY = constrained.y;
            
            this._applyZoomTransform(img, container, state);
            debouncedSave();
        });

        // CTRL + Drag = Pan
        Q(container).on('mousedown', (e) => {
            if (!e.ctrlKey || !state.initialized) return;
            e.preventDefault();
            e.stopPropagation();

            this._activeZoom = {
                isPanning: true,
                startX: e.clientX,
                startY: e.clientY,
                startCenterX: state.centerX,
                startCenterY: state.centerY,
                img: img,
                container: container,
                state: state,
                saveCallback: debouncedSave
            };

            container.style.cursor = 'grabbing';
        });

        // Double-click = Reset to center
        Q(container).on('dblclick', (e) => {
            e.preventDefault();
            if (!state.initialized) return;

            state.minZoom = 1;
            state.zoom = state.minZoom;
            state.centerX = 0.5;
            state.centerY = 0.5;
            
            this._applyZoomTransform(img, container, state);
            saveZoomState();
        });

        // Prevent native drag
        Q(container).on('dragstart', (e) => e.preventDefault());
    },

    /**
     * Create annotation editor based on dataset type
     */
    _createAnnotationEditor: function(image) {
        if (this._datasetType === 'folder_classification') {
            // No editor for folder classification
            return null;
        }

        const container = Q('<div>', { class: 'annotation-editor' });

        if (this._datasetType === 'multi_label') {
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
            
            container.append(tagInput.getElement());
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

            container.append(textarea.getElement());
        }

        return container.get(0);
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
        // Selection count
        this._bulkSelectionCount = Q('<span>', { class: 'selection-count' }).get(0);
        
        // Selection action buttons
        this._clearSelectionBtn = new ActionButton('bulk-clear', {
            label: lang('dataset_page.clear_selection'),
            className: 'btn btn-secondary btn-sm',
            onClick: () => this._clearSelection()
        });
        
        this._selectAllBtn = new ActionButton('bulk-select-all', {
            label: lang('dataset_page.select_all'),
            className: 'btn btn-secondary btn-sm',
            onClick: () => this._selectAll()
        });
        
        const selectionInfo = Q('<div>', { class: 'bulk-selection-info' })
            .append(this._bulkSelectionCount)
            .append(this._clearSelectionBtn.getElement())
            .append(this._selectAllBtn.getElement())
            .get(0);

        // Add tags group
        this._bulkAddTagsWidget = new TagInput('bulk-add-tags', {
            placeholder: lang('dataset_page.bulk_add_tags_placeholder'),
            suggestions: this._allTags,
            recentTags: this._recentTags,
            allowCustom: true
        });
        
        this._bulkAddBtn = new ActionButton('bulk-add-btn', {
            label: lang('dataset_page.bulk_add'),
            className: 'btn btn-primary btn-sm',
            onClick: () => this._bulkAddTags()
        });
        
        const addTagsGroup = Q('<div>', { class: 'bulk-tag-group' })
            .append(this._bulkAddTagsWidget.getElement())
            .append(this._bulkAddBtn.getElement())
            .get(0);

        // Remove tags group
        this._bulkRemoveTagsWidget = new TagInput('bulk-remove-tags', {
            placeholder: lang('dataset_page.bulk_remove_tags_placeholder'),
            suggestions: this._allTags,
            recentTags: this._recentTags,
            allowCustom: true
        });
        
        this._bulkRemoveBtn = new ActionButton('bulk-remove-btn', {
            label: lang('dataset_page.bulk_remove'),
            className: 'btn btn-secondary btn-sm',
            onClick: () => this._bulkRemoveTags()
        });
        
        const removeTagsGroup = Q('<div>', { class: 'bulk-tag-group' })
            .append(this._bulkRemoveTagsWidget.getElement())
            .append(this._bulkRemoveBtn.getElement())
            .get(0);

        const tagActions = Q('<div>', { class: 'bulk-tag-actions' })
            .append(addTagsGroup)
            .append(removeTagsGroup)
            .get(0);

        // Bulk delete button
        this._bulkDeleteBtn = new ActionButton('bulk-delete-btn', {
            label: lang('dataset_page.bulk_delete'),
            className: 'btn btn-secondary btn-sm',
            onClick: () => this._bulkDelete()
        });

        return Q('<div>', { class: 'bulk-actions-bar hidden' })
            .append(selectionInfo)
            .append(tagActions)
            .append(this._bulkDeleteBtn.getElement())
            .get(0);
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
