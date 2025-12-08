/**
 * HootSight - Heatmap Page
 * Grad-CAM visualization and model inference testing
 */

const HeatmapPage = {
    /**
     * Page identifier
     */
    name: 'heatmap',

    /**
     * Container reference
     */
    _container: null,

    /**
     * Current image path - full path for refresh functionality
     */
    _currentImagePath: null,

    /**
     * Is a custom uploaded image
     */
    _isUploadedImage: false,

    /**
     * Use live model (skip cache) setting
     */
    _useLiveModel: false,

    /**
     * Multi-label mode - show all class activations combined
     */
    _multiLabel: true,

    /**
     * UI element references
     */
    _elements: {
        imageContainer: null,
        resultsContainer: null,
        dropZone: null,
        heatmapImage: null,
        predictionsPanel: null,
        imageInfo: null,
        alphaSlider: null,
        refreshBtn: null,
        randomBtn: null,
        liveModelSwitch: null,
        multiLabelSwitch: null
    },

    /**
     * Build the Heatmap page
     * @param {HTMLElement} container - Container element
     */
    build: async function(container) {
        Q(container).empty();
        this._container = container;
        this._currentImagePath = null;
        this._isUploadedImage = false;
        this._useLiveModel = false;
        this._multiLabel = true;

        // Check if project is selected (use global active project)
        const projectName = Config.getActiveProject();
        if (!projectName) {
            this._buildNoProjectView(container);
            return;
        }

        // Build main layout
        this._buildLayout(container);

        // Load initial random image
        await this._loadRandomImage();
    },

    /**
     * Build "no project selected" view
     */
    _buildNoProjectView: function(container) {
        const wrapper = Q('<div>', { class: 'heatmap-no-project' }).get(0);
        
        const content = Q('<div>', { class: 'no-project-content' }).get(0);
        
        const title = Q('<h2>', { text: lang('heatmap_page.no_project.title') }).get(0);
        title.setAttribute('data-lang-key', 'heatmap_page.no_project.title');
        
        const desc = Q('<p>', { text: lang('heatmap_page.no_project.description') }).get(0);
        desc.setAttribute('data-lang-key', 'heatmap_page.no_project.description');
        
        const btn = Q('<button>', { 
            class: 'btn btn-primary',
            id: 'heatmap-go-projects',
            text: lang('heatmap_page.no_project.button')
        }).get(0);
        btn.setAttribute('data-lang-key', 'heatmap_page.no_project.button');
        
        Q(content).append(title).append(desc).append(btn);
        Q(wrapper).append(content);
        Q(container).append(wrapper);

        Q('#heatmap-go-projects').on('click', () => {
            Navigation.navigateTo('projects');
        });
    },

    /**
     * Build main page layout
     */
    _buildLayout: function(container) {
        // Page wrapper
        const wrapper = Q('<div>', { class: 'heatmap-page' }).get(0);

        // Main content - split view
        const content = Q('<div>', { class: 'heatmap-content' }).get(0);

        // Left panel - Image with drop zone
        const leftPanel = Q('<div>', { class: 'heatmap-panel heatmap-panel-left' }).get(0);
        
        const dropZone = Q('<div>', { 
            class: 'heatmap-drop-zone',
            id: 'heatmap-drop-zone'
        }).get(0);
        
        const placeholder = Q('<div>', { 
            class: 'heatmap-drop-placeholder',
            id: 'heatmap-placeholder'
        }).get(0);
        
        const dropIcon = Q('<div>', { 
            class: 'drop-icon',
            text: '\u{1F4F7}'
        }).get(0);
        
        const dropText = Q('<div>', { 
            class: 'drop-text',
            text: lang('heatmap_page.drop_zone.text')
        }).get(0);
        dropText.setAttribute('data-lang-key', 'heatmap_page.drop_zone.text');
        
        const dropHint = Q('<div>', { 
            class: 'drop-hint',
            text: lang('heatmap_page.drop_zone.hint')
        }).get(0);
        dropHint.setAttribute('data-lang-key', 'heatmap_page.drop_zone.hint');
        
        const heatmapImg = Q('<img>', { 
            class: 'heatmap-image',
            id: 'heatmap-image',
            alt: 'Heatmap'
        }).get(0);
        heatmapImg.style.display = 'none';
        
        Q(placeholder).append(dropIcon).append(dropText).append(dropHint);
        Q(dropZone).append(placeholder).append(heatmapImg);
        Q(leftPanel).append(dropZone);

        // Alpha slider
        const alphaControl = Q('<div>', { class: 'heatmap-alpha-control' }).get(0);
        
        const alphaLabel = Q('<label>', { text: lang('heatmap_page.alpha_label') }).get(0);
        alphaLabel.setAttribute('data-lang-key', 'heatmap_page.alpha_label');
        
        const alphaInput = Q('<input>', { 
            class: 'heatmap-slider',
            id: 'heatmap-alpha',
            type: 'range',
            min: '0',
            max: '100',
            value: '50'
        }).get(0);
        
        const alphaValue = Q('<span>', { 
            id: 'heatmap-alpha-value',
            text: '50%'
        }).get(0);
        
        Q(alphaControl).append(alphaLabel).append(alphaInput).append(alphaValue);
        Q(leftPanel).append(alphaControl);

        // Image info
        const imageInfo = Q('<div>', { 
            class: 'heatmap-image-info',
            id: 'heatmap-image-info'
        }).get(0);
        Q(leftPanel).append(imageInfo);

        Q(content).append(leftPanel);

        // Right panel - Results
        const rightPanel = Q('<div>', { class: 'heatmap-panel heatmap-panel-right' }).get(0);
        
        const resultsHeader = Q('<div>', { class: 'heatmap-results-header' }).get(0);
        const resultsTitle = Q('<h3>', { text: lang('heatmap_page.results.title') }).get(0);
        resultsTitle.setAttribute('data-lang-key', 'heatmap_page.results.title');
        Q(resultsHeader).append(resultsTitle);
        Q(rightPanel).append(resultsHeader);

        const predictionsPanel = Q('<div>', { 
            class: 'heatmap-predictions',
            id: 'heatmap-predictions'
        }).get(0);
        
        const predictionsPlaceholder = Q('<div>', { 
            class: 'predictions-placeholder',
            text: lang('heatmap_page.results.placeholder')
        }).get(0);
        predictionsPlaceholder.setAttribute('data-lang-key', 'heatmap_page.results.placeholder');
        Q(predictionsPanel).append(predictionsPlaceholder);
        Q(rightPanel).append(predictionsPanel);

        Q(content).append(rightPanel);
        Q(wrapper).append(content);
        Q(container).append(wrapper);

        // Store references
        this._elements.dropZone = dropZone;
        this._elements.heatmapImage = Q('#heatmap-image').get(0);
        this._elements.predictionsPanel = predictionsPanel;
        this._elements.imageInfo = imageInfo;
        this._elements.alphaSlider = Q('#heatmap-alpha').get(0);

        // Setup event listeners
        this._setupEventListeners();
    },

    /**
     * Setup header action buttons (called by app.js after page build)
     */
    setupHeaderActions: function() {
        // Clear existing buttons
        HeaderActions.clear();
        
        const headerActions = Q('#header-actions').get(0);
        if (!headerActions) return;

        const self = this;

        // Live model switch (custom element - not a standard button)
        const switchContainer = Q('<div>', { class: 'switch-container' }).get(0);
        
        const switchTrack = Q('<div>', { 
            class: 'switch-track',
            id: 'heatmap-live-model-switch',
            tabindex: '0'
        }).get(0);
        
        const switchThumb = Q('<div>', { class: 'switch-thumb' }).get(0);
        Q(switchTrack).append(switchThumb);
        
        const switchText = Q('<span>', { 
            class: 'switch-text',
            text: lang('heatmap_page.live_model_switch')
        }).get(0);
        switchText.setAttribute('data-lang-key', 'heatmap_page.live_model_switch');
        
        Q(switchContainer).append(switchTrack).append(switchText);
        Q(headerActions).append(switchContainer);

        // Multi-label switch
        const multiLabelContainer = Q('<div>', { class: 'switch-container' }).get(0);
        
        const multiLabelTrack = Q('<div>', { 
            class: 'switch-track active',
            id: 'heatmap-multi-label-switch',
            tabindex: '0'
        }).get(0);
        
        const multiLabelThumb = Q('<div>', { class: 'switch-thumb' }).get(0);
        Q(multiLabelTrack).append(multiLabelThumb);
        
        const multiLabelText = Q('<span>', { 
            class: 'switch-text',
            text: lang('heatmap_page.multi_label_switch')
        }).get(0);
        multiLabelText.setAttribute('data-lang-key', 'heatmap_page.multi_label_switch');
        
        Q(multiLabelContainer).append(multiLabelTrack).append(multiLabelText);
        Q(headerActions).append(multiLabelContainer);

        // Add buttons using HeaderActions
        HeaderActions.add([
            {
                id: 'heatmap-random',
                label: lang('heatmap_page.random_button'),
                labelLangKey: 'heatmap_page.random_button',
                type: 'secondary',
                onClick: () => self._loadRandomImage()
            },
            {
                id: 'heatmap-refresh',
                label: lang('heatmap_page.refresh_button'),
                labelLangKey: 'heatmap_page.refresh_button',
                type: 'primary',
                onClick: () => self._refreshCurrentImage()
            }
        ]);

        // Store button references
        this._elements.randomBtn = HeaderActions.get('heatmap-random')?.getElement();
        this._elements.refreshBtn = HeaderActions.get('heatmap-refresh')?.getElement();

        // Store switch reference and setup event
        this._elements.liveModelSwitch = Q('#heatmap-live-model-switch').get(0);
        if (this._elements.liveModelSwitch) {
            const switchEl = this._elements.liveModelSwitch;
            const toggle = () => {
                self._useLiveModel = !self._useLiveModel;
                Q(switchEl).toggleClass('active', self._useLiveModel);
            };
            Q(switchEl).on('click', toggle);
            Q(switchEl).on('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggle();
                }
            });
        }

        // Store multi-label switch reference and setup event
        this._elements.multiLabelSwitch = Q('#heatmap-multi-label-switch').get(0);
        if (this._elements.multiLabelSwitch) {
            const switchEl = this._elements.multiLabelSwitch;
            const toggle = () => {
                self._multiLabel = !self._multiLabel;
                Q(switchEl).toggleClass('active', self._multiLabel);
            };
            Q(switchEl).on('click', toggle);
            Q(switchEl).on('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggle();
                }
            });
        }
    },

    /**
     * Setup event listeners
     */
    _setupEventListeners: function() {
        const dropZone = this._elements.dropZone;
        const self = this;

        // Drag and drop
        Q(dropZone).on('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            Q(dropZone).addClass('drag-over');
        });

        Q(dropZone).on('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            Q(dropZone).removeClass('drag-over');
        });

        Q(dropZone).on('drop', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            Q(dropZone).removeClass('drag-over');

            const files = e.dataTransfer?.files;
            if (files && files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    await self._handleUploadedImage(file);
                }
            }
        });

        // Click to upload
        Q(dropZone).on('click', () => {
            const input = Q('<input>', { 
                type: 'file',
                accept: 'image/*'
            }).get(0);
            input.onchange = async (e) => {
                const file = e.target?.files?.[0];
                if (file) {
                    await self._handleUploadedImage(file);
                }
            };
            input.click();
        });

        // Alpha slider
        Q(this._elements.alphaSlider).on('input', (e) => {
            const value = e.target.value;
            Q('#heatmap-alpha-value').text(`${value}%`);
        });

        Q(this._elements.alphaSlider).on('change', async () => {
            if (self._currentImagePath && !self._isUploadedImage) {
                await self._refreshCurrentImage();
            }
        });
    },

    /**
     * Refresh/re-evaluate current image
     */
    _refreshCurrentImage: async function() {
        if (!this._currentImagePath) {
            // No image loaded, get a random one
            await this._loadRandomImage();
            return;
        }
        
        // Re-evaluate with the stored full image path
        await this._evaluateImage(this._currentImagePath);
    },

    /**
     * Load a random image from validation or data_source
     */
    _loadRandomImage: async function() {
        const projectName = Config.getActiveProject();
        if (!projectName) return;

        this._showLoading(true);
        this._isUploadedImage = false;
        this._currentImagePath = null;

        try {
            // Use evaluate endpoint without image_path to get random
            // Pass useLiveModel to skip cache if enabled
            const result = await API.heatmap.evaluate(projectName, null, null, this._useLiveModel, this._multiLabel);
            
            if (result.error) {
                this._showError(result.error);
                return;
            }

            // Store the full path for refresh functionality
            if (result.full_image_path) {
                this._currentImagePath = result.full_image_path;
            }

            this._displayResults(result);
        } catch (err) {
            this._showError(err.message);
        } finally {
            this._showLoading(false);
        }
    },

    /**
     * Evaluate image with given path
     */
    _evaluateImage: async function(imagePath) {
        const projectName = Config.getActiveProject();
        if (!projectName) return;

        this._showLoading(true);

        try {
            // Pass useLiveModel to skip cache if enabled
            const result = await API.heatmap.evaluate(projectName, imagePath, null, this._useLiveModel, this._multiLabel);
            
            if (result.error) {
                this._showError(result.error);
                return;
            }

            // Update stored path from response (in case it was normalized)
            if (result.full_image_path) {
                this._currentImagePath = result.full_image_path;
            }

            this._displayResults(result);
        } catch (err) {
            this._showError(err.message);
        } finally {
            this._showLoading(false);
        }
    },

    /**
     * Handle uploaded image file
     */
    _handleUploadedImage: async function(file) {
        const projectName = Config.getActiveProject();
        if (!projectName) return;

        this._showLoading(true);
        this._isUploadedImage = true;
        this._currentImagePath = null;

        try {
            const result = await API.heatmap.evaluateUpload(projectName, file, this._multiLabel);
            
            if (result.error) {
                this._showError(result.error);
                return;
            }

            // Store path for refresh (uploaded images get saved to validation folder)
            if (result.full_image_path) {
                this._currentImagePath = result.full_image_path;
            }

            this._displayResults(result);
        } catch (err) {
            this._showError(err.message);
        } finally {
            this._showLoading(false);
        }
    },

    /**
     * Display evaluation results
     */
    _displayResults: function(result) {
        // Display heatmap image
        const img = this._elements.heatmapImage;
        const placeholder = Q('#heatmap-placeholder');
        
        if (result.heatmap) {
            img.src = `data:image/png;base64,${result.heatmap}`;
            Q(img).css('display', 'block');
            placeholder.css('display', 'none');
        }

        // Update image info
        const imageInfo = this._elements.imageInfo;
        if (result.image_path) {
            this._currentImagePath = result.image_path;
            
            const filenameSpan = Q('<span>', { 
                class: 'image-filename',
                text: result.image_path
            }).get(0);
            
            Q(imageInfo).empty().append(filenameSpan);
            
            if (result.checkpoint) {
                const checkpointSpan = Q('<span>', { class: 'checkpoint-info' }).get(0);
                
                const checkpointLabel = Q('<span>', { 
                    text: lang('heatmap_page.checkpoint_label')
                }).get(0);
                checkpointLabel.setAttribute('data-lang-key', 'heatmap_page.checkpoint_label');
                
                Q(checkpointSpan).append(checkpointLabel).append(' ' + result.checkpoint);
                Q(imageInfo).append(checkpointSpan);
            }
        }

        // Display predictions
        this._displayPredictions(result.predictions);
    },

    /**
     * Display prediction results
     */
    _displayPredictions: function(predictions) {
        const panel = this._elements.predictionsPanel;
        if (!predictions || !predictions.predicted_classes || predictions.predicted_classes.length === 0) {
            const emptyDiv = Q('<div>', { 
                class: 'predictions-empty',
                text: lang('heatmap_page.results.no_predictions')
            }).get(0);
            emptyDiv.setAttribute('data-lang-key', 'heatmap_page.results.no_predictions');
            Q(panel).empty().append(emptyDiv);
            return;
        }

        const classes = predictions.predicted_classes;
        const confidences = predictions.confidence_values || [];

        let html = '<div class="predictions-list">';
        
        classes.forEach((className, index) => {
            const confidence = confidences[index] || 0;
            const percent = (confidence * 100).toFixed(1);
            const barWidth = Math.min(confidence * 100, 100);
            
            html += `
                <div class="prediction-item">
                    <div class="prediction-header">
                        <span class="prediction-class">${className}</span>
                        <span class="prediction-confidence">${percent}%</span>
                    </div>
                    <div class="prediction-bar-bg">
                        <div class="prediction-bar" style="width: ${barWidth}%"></div>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        Q(panel).html(html);
    },

    /**
     * Show loading state
     */
    _showLoading: function(loading) {
        const dropZone = this._elements.dropZone;
        Q(dropZone).toggleClass('loading', loading);
    },

    /**
     * Show error message
     */
    _showError: function(message) {
        const panel = this._elements.predictionsPanel;
        Q(panel).html(`
            <div class="predictions-error">
                <div class="error-icon">&#9888;</div>
                <div class="error-message">${message}</div>
            </div>
        `);
    },

    /**
     * Cleanup when leaving page
     */
    cleanup: function() {
        this._currentImagePath = null;
        this._isUploadedImage = false;
        this._useLiveModel = false;
        this._elements = {
            imageContainer: null,
            resultsContainer: null,
            dropZone: null,
            heatmapImage: null,
            predictionsPanel: null,
            imageInfo: null,
            alphaSlider: null,
            refreshBtn: null,
            randomBtn: null,
            liveModelSwitch: null
        };
    }
};

// Register with Pages system
Pages.register('heatmap', HeatmapPage);
