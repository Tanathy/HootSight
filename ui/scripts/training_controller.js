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
        const headerProgress = document.getElementById('header-progress');
        if (!headerProgress) return;

        // Create training progress container
        this._headerElement = document.createElement('div');
        this._headerElement.className = 'header-training-progress';
        this._headerElement.id = 'header-training-progress';
        this._headerElement.style.display = 'none';

        // Progress info
        const infoContainer = document.createElement('div');
        infoContainer.className = 'training-progress-info';

        const projectLabel = document.createElement('span');
        projectLabel.className = 'training-project-label';
        projectLabel.id = 'training-project-label';
        projectLabel.textContent = '';
        infoContainer.appendChild(projectLabel);

        const progressText = document.createElement('span');
        progressText.className = 'training-progress-text';
        progressText.id = 'training-progress-text';
        progressText.textContent = '';
        infoContainer.appendChild(progressText);

        this._headerElement.appendChild(infoContainer);

        // Progress bar
        const progressBarContainer = document.createElement('div');
        progressBarContainer.className = 'training-progress-bar-container';

        const progressBar = document.createElement('div');
        progressBar.className = 'training-progress-bar';
        progressBar.id = 'training-progress-bar';
        progressBar.style.width = '0%';
        progressBarContainer.appendChild(progressBar);

        this._headerElement.appendChild(progressBarContainer);

        // Stop button
        const stopBtn = document.createElement('button');
        stopBtn.className = 'btn btn-secondary btn-sm training-stop-btn';
        stopBtn.id = 'training-stop-btn';
        stopBtn.textContent = lang('training_controller.stop');
        stopBtn.onclick = () => this.stopTraining();
        this._headerElement.appendChild(stopBtn);

        headerProgress.appendChild(this._headerElement);
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
        const projectLabel = document.getElementById('training-project-label');
        if (projectLabel) {
            projectLabel.textContent = this._trainingProject || state.project || '';
        }

        // Progress text
        const progressText = document.getElementById('training-progress-text');
        if (progressText) {
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
            
            progressText.textContent = text;
        }

        // Progress bar
        const progressBar = document.getElementById('training-progress-bar');
        if (progressBar) {
            const epoch = state.currentEpoch || 0;
            const totalEpochs = state.totalEpochs || 1;
            const step = state.currentStep || 0;
            const totalSteps = state.totalSteps || 1;
            
            // Calculate overall progress
            const epochProgress = (epoch - 1) / totalEpochs;
            const stepProgress = step / totalSteps / totalEpochs;
            const overallProgress = Math.min((epochProgress + stepProgress) * 100, 100);
            
            progressBar.style.width = `${overallProgress}%`;
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
        const progressText = document.getElementById('training-progress-text');
        if (progressText) {
            if (status === 'completed') {
                progressText.textContent = lang('training_controller.completed');
            } else if (status === 'stopped') {
                progressText.textContent = lang('training_controller.stopped');
            } else {
                progressText.textContent = lang('training_controller.error');
            }
        }

        // Change stop button to "Clear"
        const stopBtn = document.getElementById('training-stop-btn');
        if (stopBtn) {
            stopBtn.textContent = lang('training_controller.clear');
            stopBtn.className = 'btn btn-secondary btn-sm training-stop-btn';
            stopBtn.onclick = () => this._clearProgress();
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
        const stopBtn = document.getElementById('training-stop-btn');
        if (stopBtn) {
            stopBtn.textContent = lang('training_controller.stop');
            stopBtn.className = 'btn btn-secondary btn-sm training-stop-btn';
            stopBtn.onclick = () => this.stopTraining();
        }
    },

    /**
     * Show header progress
     */
    _showProgress: function() {
        if (this._headerElement) {
            this._headerElement.style.display = 'flex';
        }
    },

    /**
     * Hide header progress
     */
    _hideProgress: function() {
        if (this._headerElement) {
            this._headerElement.style.display = 'none';
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
