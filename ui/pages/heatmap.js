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
        liveModelSwitch: null
    },

    /**
     * Build the Heatmap page
     * @param {HTMLElement} container - Container element
     */
    build: async function(container) {
        container.innerHTML = '';
        this._container = container;
        this._currentImagePath = null;
        this._isUploadedImage = false;
        this._useLiveModel = false;

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
        const wrapper = document.createElement('div');
        wrapper.className = 'heatmap-no-project';
        wrapper.innerHTML = `
            <div class="no-project-content">
                <h2>${lang('heatmap_page.no_project.title')}</h2>
                <p>${lang('heatmap_page.no_project.description')}</p>
                <button class="btn btn-primary" id="heatmap-go-projects">
                    ${lang('heatmap_page.no_project.button')}
                </button>
            </div>
        `;
        container.appendChild(wrapper);

        document.getElementById('heatmap-go-projects')?.addEventListener('click', () => {
            Navigation.navigateTo('projects');
        });
    },

    /**
     * Build main page layout
     */
    _buildLayout: function(container) {
        const projectName = Config.getActiveProject();

        // Page wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'heatmap-page';

        // Header
        const header = document.createElement('div');
        header.className = 'heatmap-header';
        header.innerHTML = `
            <div class="heatmap-header-left">
                <span class="heatmap-project-label">${lang('common.project_label')}</span>
                <span class="heatmap-project-name">${projectName}</span>
            </div>
            <div class="heatmap-header-center">
                <div class="switch-container">
                    <div class="switch-track" id="heatmap-live-model-switch" tabindex="0">
                        <div class="switch-thumb"></div>
                    </div>
                    <span class="switch-text">${lang('heatmap_page.live_model_switch')}</span>
                </div>
            </div>
            <div class="heatmap-header-actions">
                <button class="btn btn-secondary" id="heatmap-random-btn">
                    ${lang('heatmap_page.random_button')}
                </button>
                <button class="btn btn-primary" id="heatmap-refresh-btn">
                    ${lang('heatmap_page.refresh_button')}
                </button>
            </div>
        `;
        wrapper.appendChild(header);

        // Main content - split view
        const content = document.createElement('div');
        content.className = 'heatmap-content';

        // Left panel - Image with drop zone
        const leftPanel = document.createElement('div');
        leftPanel.className = 'heatmap-panel heatmap-panel-left';
        
        const dropZone = document.createElement('div');
        dropZone.className = 'heatmap-drop-zone';
        dropZone.id = 'heatmap-drop-zone';
        dropZone.innerHTML = `
            <div class="heatmap-drop-placeholder" id="heatmap-placeholder">
                <div class="drop-icon">&#128247;</div>
                <div class="drop-text">${lang('heatmap_page.drop_zone.text')}</div>
                <div class="drop-hint">${lang('heatmap_page.drop_zone.hint')}</div>
            </div>
            <img class="heatmap-image" id="heatmap-image" style="display: none;" alt="Heatmap">
        `;
        leftPanel.appendChild(dropZone);

        // Alpha slider
        const alphaControl = document.createElement('div');
        alphaControl.className = 'heatmap-alpha-control';
        alphaControl.innerHTML = `
            <label>${lang('heatmap_page.alpha_label')}</label>
            <input type="range" id="heatmap-alpha" min="0" max="100" value="50" class="heatmap-slider">
            <span id="heatmap-alpha-value">50%</span>
        `;
        leftPanel.appendChild(alphaControl);

        // Image info
        const imageInfo = document.createElement('div');
        imageInfo.className = 'heatmap-image-info';
        imageInfo.id = 'heatmap-image-info';
        leftPanel.appendChild(imageInfo);

        content.appendChild(leftPanel);

        // Right panel - Results
        const rightPanel = document.createElement('div');
        rightPanel.className = 'heatmap-panel heatmap-panel-right';
        
        const resultsHeader = document.createElement('div');
        resultsHeader.className = 'heatmap-results-header';
        resultsHeader.innerHTML = `<h3>${lang('heatmap_page.results.title')}</h3>`;
        rightPanel.appendChild(resultsHeader);

        const predictionsPanel = document.createElement('div');
        predictionsPanel.className = 'heatmap-predictions';
        predictionsPanel.id = 'heatmap-predictions';
        predictionsPanel.innerHTML = `
            <div class="predictions-placeholder">
                ${lang('heatmap_page.results.placeholder')}
            </div>
        `;
        rightPanel.appendChild(predictionsPanel);

        content.appendChild(rightPanel);
        wrapper.appendChild(content);
        container.appendChild(wrapper);

        // Store references
        this._elements.dropZone = dropZone;
        this._elements.heatmapImage = document.getElementById('heatmap-image');
        this._elements.predictionsPanel = predictionsPanel;
        this._elements.imageInfo = imageInfo;
        this._elements.alphaSlider = document.getElementById('heatmap-alpha');
        this._elements.refreshBtn = document.getElementById('heatmap-refresh-btn');
        this._elements.randomBtn = document.getElementById('heatmap-random-btn');
        this._elements.liveModelSwitch = document.getElementById('heatmap-live-model-switch');

        // Setup event listeners
        this._setupEventListeners();
    },

    /**
     * Setup event listeners
     */
    _setupEventListeners: function() {
        const dropZone = this._elements.dropZone;

        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');

            const files = e.dataTransfer?.files;
            if (files && files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    await this._handleUploadedImage(file);
                }
            }
        });

        // Click to upload
        dropZone.addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.onchange = async (e) => {
                const file = e.target?.files?.[0];
                if (file) {
                    await this._handleUploadedImage(file);
                }
            };
            input.click();
        });

        // Alpha slider
        this._elements.alphaSlider?.addEventListener('input', (e) => {
            const value = e.target.value;
            document.getElementById('heatmap-alpha-value').textContent = `${value}%`;
        });

        this._elements.alphaSlider?.addEventListener('change', async () => {
            if (this._currentImagePath && !this._isUploadedImage) {
                await this._refreshCurrentImage();
            }
        });

        // Live model switch (click toggles)
        if (this._elements.liveModelSwitch) {
            const switchEl = this._elements.liveModelSwitch;
            const toggle = () => {
                this._useLiveModel = !this._useLiveModel;
                switchEl.classList.toggle('active', this._useLiveModel);
            };
            switchEl.addEventListener('click', toggle);
            switchEl.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggle();
                }
            });
        }

        // Random button - loads new random image
        this._elements.randomBtn?.addEventListener('click', async () => {
            await this._loadRandomImage();
        });

        // Refresh button - re-evaluates current image
        this._elements.refreshBtn?.addEventListener('click', async () => {
            await this._refreshCurrentImage();
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
            const result = await API.heatmap.evaluate(projectName, null, null, this._useLiveModel);
            
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
            const result = await API.heatmap.evaluate(projectName, imagePath, null, this._useLiveModel);
            
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
            const result = await API.heatmap.evaluateUpload(projectName, file);
            
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
        const placeholder = document.getElementById('heatmap-placeholder');
        
        if (result.heatmap) {
            img.src = `data:image/png;base64,${result.heatmap}`;
            img.style.display = 'block';
            placeholder.style.display = 'none';
        }

        // Update image info
        const imageInfo = this._elements.imageInfo;
        if (result.image_path) {
            this._currentImagePath = result.image_path;
            imageInfo.innerHTML = `
                <span class="image-filename">${result.image_path}</span>
                ${result.checkpoint ? `<span class="checkpoint-info">${lang('heatmap_page.checkpoint_label')} ${result.checkpoint}</span>` : ''}
            `;
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
            panel.innerHTML = `
                <div class="predictions-empty">
                    ${lang('heatmap_page.results.no_predictions')}
                </div>
            `;
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
        panel.innerHTML = html;
    },

    /**
     * Show loading state
     */
    _showLoading: function(loading) {
        const dropZone = this._elements.dropZone;
        if (loading) {
            dropZone.classList.add('loading');
        } else {
            dropZone.classList.remove('loading');
        }
    },

    /**
     * Show error message
     */
    _showError: function(message) {
        const panel = this._elements.predictionsPanel;
        panel.innerHTML = `
            <div class="predictions-error">
                <div class="error-icon">&#9888;</div>
                <div class="error-message">${message}</div>
            </div>
        `;
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
Pages.register('heatmap', {
    build: (container) => HeatmapPage.build(container),
    cleanup: () => HeatmapPage.cleanup()
});
