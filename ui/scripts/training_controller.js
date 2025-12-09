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
     * Current training mode: 'new' | 'resume' | 'finetune'
     */
    _trainingMode: null,

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
     * ETA calculation data
     */
    _etaData: {
        samples: [],          // { timestamp, totalSteps } - last 60 seconds of samples
        lastUpdate: 0,
        windowSize: 60000,    // 60 second window in ms (1 minute)
    },

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

        // Project label
        const projectLabel = Q('<span>', { 
            class: 'training-project-label', 
            id: 'training-project-label' 
        }).get();
        Q(this._headerElement).append(projectLabel);

        // Mode pill
        const modePill = Q('<span>', {
            class: 'header-action-pill training-mode-pill',
            id: 'training-mode-pill'
        }).get();
        Q(modePill).hide();
        Q(this._headerElement).append(modePill);

        // Progress bar
        const progressBarContainer = Q('<div>', { class: 'training-progress-bar-container' }).get();
        const progressBar = Q('<div>', { 
            class: 'training-progress-bar', 
            id: 'training-progress-bar' 
        }).get();
        Q(progressBar).css('width', '0%');
        Q(progressBarContainer).append(progressBar);
        Q(this._headerElement).append(progressBarContainer);

        // ETA display
        const etaDisplay = Q('<span>', { 
            class: 'training-eta', 
            id: 'training-eta' 
        }).get();
        Q(this._headerElement).append(etaDisplay);

        // Stop/Clear action styled as secondary header pill
        const stopBtn = Q('<div>', {
            class: 'header-action-pill training-stop-btn',
            id: 'training-stop-btn',
            text: lang('training_controller.stop')
        }).get();
        stopBtn.setAttribute('role', 'button');
        stopBtn.setAttribute('tabindex', '0');
        stopBtn.setAttribute('data-lang-key', 'training_controller.stop');
        Q(stopBtn).on('click', () => this.stopTraining());
        Q(stopBtn).on('keypress', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.stopTraining();
            }
        });
        Q(this._headerElement).append(stopBtn);

        Q(headerProgress).append(this._headerElement);
    },

    /**
     * Reset ETA calculation data
     */
    _resetEtaData: function() {
        this._etaData = {
            samples: [],
            lastUpdate: 0,
            windowSize: 60000,
        };
    },

    /**
     * Calculate ETA based on step rate over last 10 seconds
     * @param {number} currentStep - Current step in current epoch
     * @param {number} totalSteps - Total steps per epoch
     * @param {number} currentEpoch - Current epoch
     * @param {number} totalEpochs - Total epochs
     * @returns {string} - Formatted ETA string
     */
    _calculateEta: function(currentStep, totalSteps, currentEpoch, totalEpochs) {
        const now = Date.now();
        
        // Calculate total completed steps across all epochs
        const completedEpochs = currentEpoch - 1;
        const totalCompletedSteps = (completedEpochs * totalSteps) + currentStep;
        const grandTotalSteps = totalEpochs * totalSteps;
        const remainingSteps = grandTotalSteps - totalCompletedSteps;

        // Add current sample
        this._etaData.samples.push({
            timestamp: now,
            totalSteps: totalCompletedSteps
        });

        // Remove samples older than window
        const cutoff = now - this._etaData.windowSize;
        this._etaData.samples = this._etaData.samples.filter(s => s.timestamp >= cutoff);

        // Need at least 2 samples to calculate rate
        if (this._etaData.samples.length < 2) {
            return '--:--';
        }

        // Calculate steps per second over the window
        const oldest = this._etaData.samples[0];
        const newest = this._etaData.samples[this._etaData.samples.length - 1];
        const timeDiff = (newest.timestamp - oldest.timestamp) / 1000; // seconds
        const stepsDiff = newest.totalSteps - oldest.totalSteps;

        if (timeDiff <= 0 || stepsDiff <= 0) {
            return '--:--';
        }

        const stepsPerSecond = stepsDiff / timeDiff;
        const remainingSeconds = Math.ceil(remainingSteps / stepsPerSecond);

        return this._formatDuration(remainingSeconds);
    },

    /**
     * Format duration in smart format: DD:HH:MM:SS or HH:MM:SS or MM:SS
     * @param {number} totalSeconds - Total seconds
     * @returns {string} - Formatted string
     */
    _formatDuration: function(totalSeconds) {
        if (totalSeconds <= 0 || !isFinite(totalSeconds)) {
            return '--:--';
        }

        const days = Math.floor(totalSeconds / 86400);
        const hours = Math.floor((totalSeconds % 86400) / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = Math.floor(totalSeconds % 60);

        const pad = (n) => n.toString().padStart(2, '0');

        if (days > 0) {
            // DD:HH:MM:SS
            return `${pad(days)}:${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
        } else if (hours > 0) {
            // HH:MM:SS
            return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
        } else {
            // MM:SS
            return `${pad(minutes)}:${pad(seconds)}`;
        }
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
                this._trainingMode = 'resume';
                
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
     * @param {boolean} [resume] - Resume from last checkpoint
     * @returns {Promise<Object>} - Result
     */
    startTraining: async function(projectName, modelType = null, modelName = null, epochs = null, resume = false, mode = null) {
        if (this._activeTrainingId) {
            return { 
                started: false, 
                error: lang('training_controller.already_running') 
            };
        }

        try {
            // Get model info from config if not provided
            if (!modelType || !modelName) {
                const trainingConfig = Config.get('training') || {};
                modelType = modelType || trainingConfig.model_type || 'resnet';
                modelName = modelName || trainingConfig.model_name || 'resnet50';
            }

            // Determine effective mode
            const effectiveMode = mode || (resume ? 'resume' : 'new');

            const result = await API.training.start(projectName, modelType, modelName, epochs, resume, effectiveMode);

            if (result.started) {
                this._activeTrainingId = result.training_id;
                this._trainingProject = projectName;
                this._trainingMode = result.mode || effectiveMode;
                
                // Clear any previous history and reset ETA
                TrainingMonitor.clearHistory();
                this._resetEtaData();
                
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

        // Mode pill (resume / fine-tune / new)
        const modePill = Q('#training-mode-pill');
        if (modePill.get()) {
            const mode = this._trainingMode || 'new';
            let textKey = 'training_controller.mode.new';
            if (mode === 'resume') textKey = 'training_controller.mode.resume';
            if (mode === 'finetune') textKey = 'training_controller.mode.finetune';
            modePill.text(lang(textKey));
            modePill.get().setAttribute('data-lang-key', textKey);
            modePill.show();
        }

        const epoch = state.currentEpoch || 0;
        const totalEpochs = state.totalEpochs || 0;
        const step = state.currentStep || 0;
        const totalSteps = state.totalSteps || 1;

        // Progress bar
        const progressBar = Q('#training-progress-bar');
        if (progressBar.get()) {
            // Calculate overall progress
            const epochProgress = (epoch - 1) / (totalEpochs || 1);
            const stepProgress = step / totalSteps / (totalEpochs || 1);
            const overallProgress = Math.min((epochProgress + stepProgress) * 100, 100);
            progressBar.css('width', `${overallProgress}%`);
        }

        // ETA display
        const etaDisplay = Q('#training-eta');
        if (etaDisplay.get() && totalEpochs > 0 && totalSteps > 0) {
            etaDisplay.removeClass('training-status');
            const eta = this._calculateEta(step, totalSteps, epoch, totalEpochs);
            etaDisplay.text(eta);
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

        // Reset ETA data
        this._resetEtaData();

        // Update ETA display to show status
        const etaDisplay = Q('#training-eta');
        if (etaDisplay.get()) {
            let langKey;
            if (status === 'completed') {
                langKey = 'training_controller.completed';
            } else if (status === 'stopped') {
                langKey = 'training_controller.stopped';
            } else {
                langKey = 'training_controller.error';
            }
            etaDisplay.text(lang(langKey));
            etaDisplay.get().setAttribute('data-lang-key', langKey);
            etaDisplay.addClass('training-status');
        }

        // Change stop button to "Clear"
        const stopBtn = Q('#training-stop-btn');
        if (stopBtn.get()) {
            stopBtn.text(lang('training_controller.clear'));
            stopBtn.get().setAttribute('data-lang-key', 'training_controller.clear');
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
        this._resetEtaData();
        this._trainingMode = null;
        
        // Reset stop button
        const stopBtn = Q('#training-stop-btn');
        if (stopBtn.get()) {
            stopBtn.text(lang('training_controller.stop'));
            stopBtn.get().setAttribute('data-lang-key', 'training_controller.stop');
            stopBtn.off('click').on('click', () => this.stopTraining());
        }

        // Clear mode pill
        const modePill = Q('#training-mode-pill');
        if (modePill.get()) {
            modePill.text('');
            modePill.hide();
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
        return TrainingMonitor.isRunning() || this._activeTrainingId !== null;
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
    },

    getTrainingMode: function() {
        return this._trainingMode;
    }
};
