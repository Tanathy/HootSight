/**
 * HootSight - Training Controller
 * Manages training sessions independently of project selection
 * Displays training progress in the header
 */

const TrainingController = {
    /**
     * Currently running training ID
     */
    _activeTrainingId: null,

    /**
     * Training project name (may differ from selected project)
     */
    _trainingProject: null,

    /**
     * Header progress element reference
     */
    _headerElement: null,

    /**
     * Unsubscribe function for TrainingMonitor
     */
    _unsubscribe: null,

    /**
     * Initialize the controller
     * @returns {Promise<void>}
     */
    init: async function() {
        this._createHeaderElement();
        await this._checkExistingTraining();
    },

    /**
     * Create the header training progress element
     */
    _createHeaderElement: function() {
        const headerProgress = Q('#header-progress').get();
        if (!headerProgress) return;

        // Create training progress container
        this._headerElement = Q('<div>', { 
            class: 'header-training-progress', 
            id: 'header-training-progress' 
        }).get();
        Q(this._headerElement).css('display', 'none');

        // Progress info
        const infoContainer = Q('<div>', { class: 'training-progress-info' }).get();

        const projectLabel = Q('<span>', { 
            class: 'training-project-label', 
            id: 'training-project-label' 
        }).get();
        Q(infoContainer).append(projectLabel);

        const progressText = Q('<span>', { 
            class: 'training-progress-text', 
            id: 'training-progress-text' 
        }).get();
        Q(infoContainer).append(progressText);

        Q(this._headerElement).append(infoContainer);

        // Progress bar
        const progressBarContainer = Q('<div>', { class: 'training-progress-bar-container' }).get();

        const progressBar = Q('<div>', { 
            class: 'training-progress-bar', 
            id: 'training-progress-bar' 
        }).get();
        Q(progressBar).css('width', '0%');
        Q(progressBarContainer).append(progressBar);

        Q(this._headerElement).append(progressBarContainer);

        // Stop button
        const stopBtn = Q('<button>', { 
            class: 'btn btn-secondary btn-sm training-stop-btn', 
            id: 'training-stop-btn',
            text: lang('training_controller.stop')
        }).get();
        Q(stopBtn).on('click', () => this.stopTraining());
        Q(this._headerElement).append(stopBtn);

        Q(headerProgress).append(this._headerElement);
    },

    /**
     * Check if there's already a running training on app load
     */
    _checkExistingTraining: async function() {
        try {
            const status = await API.training.getStatus();
            if (status.active_trainings && status.active_trainings.length > 0) {
                // Resume monitoring the first active training
                const trainingId = status.active_trainings[0];
                this._activeTrainingId = trainingId;
                
                // Get details
                const details = await API.training.getStatus(trainingId);
                this._trainingProject = details.project;
                
                // Load full history first so Performance page can access it
                await TrainingMonitor.loadHistory(trainingId);
                
                // Start monitoring for live updates
                this._startMonitoring(trainingId);
                this._showProgress();
            }
        } catch (e) {
            console.warn('Could not check for existing training:', e);
        }
    },

    /**
     * Start training for a project
     * @param {string} projectName - Project to train
     * @param {string} modelType - Model type (e.g., 'resnet')
     * @param {string} modelName - Model name (e.g., 'resnet50')
     * @param {number} [epochs] - Optional epoch override
     * @returns {Promise<Object>} - Result
     */
    startTraining: async function(projectName, modelType = null, modelName = null, epochs = null) {
        if (this._activeTrainingId) {
            return { 
                started: false, 
                error: lang('training_controller.already_running') 
            };
        }

        try {
            // Get model info from config if not provided
            if (!modelType || !modelName) {
                const config = Config.get('model') || {};
                modelType = modelType || config.type || 'resnet';
                modelName = modelName || config.name || 'resnet50';
            }

            const result = await API.training.start(projectName, modelType, modelName, epochs);

            if (result.started) {
                this._activeTrainingId = result.training_id;
                this._trainingProject = projectName;
                
                // Clear any previous history
                TrainingMonitor.clearHistory();
                
                // Start monitoring
                this._startMonitoring(result.training_id);
                this._showProgress();
            }

            return result;

        } catch (e) {
            console.error('Failed to start training:', e);
            return { started: false, error: e.message };
        }
    },

    /**
     * Stop the current training
     * @returns {Promise<Object>} - Result
     */
    stopTraining: async function() {
        if (!this._activeTrainingId) {
            return { stopped: false, error: 'No active training' };
        }

        try {
            const result = await API.training.stop(this._activeTrainingId);
            
            // Don't immediately clear - let the monitor handle completion
            return result;

        } catch (e) {
            console.error('Failed to stop training:', e);
            return { stopped: false, error: e.message };
        }
    },

    /**
     * Start monitoring training progress
     */
    _startMonitoring: function(trainingId) {
        // Subscribe to TrainingMonitor updates
        this._unsubscribe = TrainingMonitor.subscribe((data) => {
            this._onTrainingUpdate(data);
        });

        // Start the monitor
        TrainingMonitor.start(trainingId);
    },

    /**
     * Handle training update
     */
    _onTrainingUpdate: function(data) {
        const { state, metrics } = data;

        // Update header progress
        this._updateProgress(state, metrics);

        // Check if training ended
        if (state.status === 'completed' || state.status === 'stopped' || state.status === 'error') {
            this._onTrainingEnded(state.status);
        }
    },

    /**
     * Update header progress display
     */
    _updateProgress: function(state, metrics) {
        // Project label
        const projectLabel = Q('#training-project-label');
        if (projectLabel.get()) {
            projectLabel.text(this._trainingProject || state.project || '');
        }

        // Progress text
        const progressText = Q('#training-progress-text');
        if (progressText.get()) {
            const epoch = state.currentEpoch || 0;
            const totalEpochs = state.totalEpochs || 0;
            const step = state.currentStep || 0;
            const totalSteps = state.totalSteps || 0;
            const phase = state.phase === 'val' ? 'Val' : 'Train';
            
            let text = '';
            if (totalEpochs > 0) {
                text = `Epoch ${epoch}/${totalEpochs}`;
                if (totalSteps > 0) {
                    text += ` - ${phase} ${step}/${totalSteps}`;
                }
            }
            
            if (metrics && metrics.loss !== null && metrics.loss !== undefined) {
                text += ` - Loss: ${Format.loss(metrics.loss)}`;
            }
            
            progressText.text(text);
        }

        // Progress bar
        const progressBar = Q('#training-progress-bar');
        if (progressBar.get()) {
            const epoch = state.currentEpoch || 0;
            const totalEpochs = state.totalEpochs || 1;
            const step = state.currentStep || 0;
            const totalSteps = state.totalSteps || 1;
            
            // Calculate overall progress
            const epochProgress = (epoch - 1) / totalEpochs;
            const stepProgress = step / totalSteps / totalEpochs;
            const overallProgress = Math.min((epochProgress + stepProgress) * 100, 100);
            
            progressBar.css('width', `${overallProgress}%`);
        }
    },

    /**
     * Handle training end
     */
    _onTrainingEnded: function(status) {
        // Unsubscribe from monitor
        if (this._unsubscribe) {
            this._unsubscribe();
            this._unsubscribe = null;
        }

        // Update UI to show completion
        const progressText = Q('#training-progress-text');
        if (progressText.get()) {
            if (status === 'completed') {
                progressText.text(lang('training_controller.completed'));
            } else if (status === 'stopped') {
                progressText.text(lang('training_controller.stopped'));
            } else {
                progressText.text(lang('training_controller.error'));
            }
        }

        // Change stop button to "Clear"
        const stopBtn = Q('#training-stop-btn');
        if (stopBtn.get()) {
            stopBtn.text(lang('training_controller.clear'));
            stopBtn.off('click').on('click', () => this._clearProgress());
        }

        // Clear active training ID
        this._activeTrainingId = null;
    },

    /**
     * Clear progress display
     */
    _clearProgress: function() {
        this._hideProgress();
        this._trainingProject = null;
        
        // Reset stop button
        const stopBtn = Q('#training-stop-btn');
        if (stopBtn.get()) {
            stopBtn.text(lang('training_controller.stop'));
            stopBtn.off('click').on('click', () => this.stopTraining());
        }
    },

    /**
     * Show header progress
     */
    _showProgress: function() {
        if (this._headerElement) {
            Q(this._headerElement).css('display', 'flex');
        }
    },

    /**
     * Hide header progress
     */
    _hideProgress: function() {
        if (this._headerElement) {
            Q(this._headerElement).css('display', 'none');
        }
    },

    /**
     * Check if training is active
     */
    isTraining: function() {
        return this._activeTrainingId !== null;
    },

    /**
     * Get current training project
     */
    getTrainingProject: function() {
        return this._trainingProject;
    },

    /**
     * Get active training ID
     */
    getActiveTrainingId: function() {
        return this._activeTrainingId;
    }
};
