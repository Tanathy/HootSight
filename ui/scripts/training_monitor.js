/**
 * HootSight - Training Monitor Service
 * Background service that tracks training progress and maintains history
 */

const TrainingMonitor = {
    /**
     * Is monitoring currently active
     */
    _isRunning: false,

    /**
     * Polling interval ID
     */
    _pollInterval: null,

    /**
     * Polling rate in ms
     */
    _pollRate: 500,

    /**
     * Currently tracked training ID
     */
    _activeTrainingId: null,

    /**
     * Training state
     */
    _state: {
        status: null,           // 'idle', 'running', 'completed', 'stopped', 'error'
        trainingId: null,
        project: null,
        modelType: null,
        modelName: null,
        currentEpoch: 0,
        totalEpochs: 0,
        currentStep: 0,
        totalSteps: 0,
        phase: null,            // 'train' or 'val'
        batchSize: null,
        trainSamples: null,
        valSamples: null
    },

    /**
     * Historical data storage - no limit
     */
    _history: {
        // Step-level metrics
        stepLoss: [],           // { epoch, step, value, phase }
        stepAccuracy: [],       // { epoch, step, value, phase }
        learningRate: [],       // { epoch, step, value }
        
        // Epoch-level metrics
        epochTrainLoss: [],     // { epoch, value }
        epochTrainAccuracy: [], // { epoch, value }
        epochValLoss: [],       // { epoch, value }
        epochValAccuracy: [],   // { epoch, value }
        
        // Raw events for detailed analysis
        events: []
    },

    /**
     * Latest metrics snapshot
     */
    _latestMetrics: {
        loss: null,
        accuracy: null,
        learningRate: null,
        epochLoss: null,
        epochAccuracy: null,
        valLoss: null,
        valAccuracy: null
    },

    /**
     * Callbacks for live updates
     */
    _listeners: [],

    /**
     * Start monitoring a specific training
     * @param {string} trainingId - Training ID to monitor
     */
    start: function(trainingId) {
        if (this._isRunning && this._activeTrainingId === trainingId) return;
        
        // Stop any existing monitoring
        this.stop();
        
        this._activeTrainingId = trainingId;
        this._isRunning = true;
        this._state.status = 'running';
        this._state.trainingId = trainingId;
        
        this._poll();
        this._pollInterval = setInterval(() => this._poll(), this._pollRate);
    },

    /**
     * Stop monitoring
     */
    stop: function() {
        this._isRunning = false;
        
        if (this._pollInterval) {
            clearInterval(this._pollInterval);
            this._pollInterval = null;
        }
    },

    /**
     * Check if monitoring is active
     */
    isRunning: function() {
        return this._isRunning;
    },

    /**
     * Get current training ID
     */
    getActiveTrainingId: function() {
        return this._activeTrainingId;
    },

    /**
     * Get current state
     */
    getState: function() {
        return { ...this._state };
    },

    /**
     * Get history data
     */
    getHistory: function() {
        return {
            stepLoss: [...this._history.stepLoss],
            stepAccuracy: [...this._history.stepAccuracy],
            learningRate: [...this._history.learningRate],
            epochTrainLoss: [...this._history.epochTrainLoss],
            epochTrainAccuracy: [...this._history.epochTrainAccuracy],
            epochValLoss: [...this._history.epochValLoss],
            epochValAccuracy: [...this._history.epochValAccuracy]
        };
    },

    /**
     * Get latest metrics - reconstructs from history if needed
     */
    getLatestMetrics: function() {
        const metrics = { ...this._latestMetrics };
        
        // If any metric is null but we have history, use last history value
        if (metrics.loss === null && this._history.stepLoss.length > 0) {
            metrics.loss = this._history.stepLoss[this._history.stepLoss.length - 1].value;
        }
        if (metrics.learningRate === null && this._history.learningRate.length > 0) {
            metrics.learningRate = this._history.learningRate[this._history.learningRate.length - 1].value;
        }
        if (metrics.epochLoss === null && this._history.epochTrainLoss.length > 0) {
            metrics.epochLoss = this._history.epochTrainLoss[this._history.epochTrainLoss.length - 1].value;
        }
        if (metrics.epochAccuracy === null && this._history.epochTrainAccuracy.length > 0) {
            metrics.epochAccuracy = this._history.epochTrainAccuracy[this._history.epochTrainAccuracy.length - 1].value;
        }
        if (metrics.valLoss === null && this._history.epochValLoss.length > 0) {
            metrics.valLoss = this._history.epochValLoss[this._history.epochValLoss.length - 1].value;
        }
        if (metrics.valAccuracy === null && this._history.epochValAccuracy.length > 0) {
            metrics.valAccuracy = this._history.epochValAccuracy[this._history.epochValAccuracy.length - 1].value;
        }
        
        return metrics;
    },

    /**
     * Clear all history
     */
    clearHistory: function() {
        this._history = {
            stepLoss: [],
            stepAccuracy: [],
            learningRate: [],
            epochTrainLoss: [],
            epochTrainAccuracy: [],
            epochValLoss: [],
            epochValAccuracy: [],
            events: []
        };
        this._latestMetrics = {
            loss: null,
            accuracy: null,
            learningRate: null,
            epochLoss: null,
            epochAccuracy: null,
            valLoss: null,
            valAccuracy: null
        };
        this._state = {
            status: 'idle',
            trainingId: null,
            project: null,
            modelType: null,
            modelName: null,
            currentEpoch: 0,
            totalEpochs: 0,
            currentStep: 0,
            totalSteps: 0,
            phase: null,
            batchSize: null,
            trainSamples: null,
            valSamples: null
        };
    },

    /**
     * Register a listener for live updates
     * @param {Function} callback - Called with { state, metrics, newEvents }
     * @returns {Function} Unsubscribe function
     */
    subscribe: function(callback) {
        this._listeners.push(callback);
        
        return () => {
            const idx = this._listeners.indexOf(callback);
            if (idx > -1) {
                this._listeners.splice(idx, 1);
            }
        };
    },

    /**
     * Notify all listeners
     */
    _notifyListeners: function(data) {
        this._listeners.forEach(cb => {
            try {
                cb(data);
            } catch (e) {
                console.error('TrainingMonitor listener error:', e);
            }
        });
    },

    /**
     * Process incoming updates
     */
    _processUpdates: function(updates) {
        if (!updates || !Array.isArray(updates)) return [];
        
        const newEvents = [];
        
        updates.forEach(event => {
            // Store raw event
            this._history.events.push(event);
            newEvents.push(event);
            
            const metrics = event.metrics || {};
            const epoch = event.epoch;
            const step = event.step;
            const phase = event.phase;
            
            if (event.type === 'step') {
                // Step-level metrics
                if (metrics.step_loss !== undefined) {
                    this._history.stepLoss.push({
                        epoch,
                        step,
                        phase,
                        value: metrics.step_loss
                    });
                    this._latestMetrics.loss = metrics.step_loss;
                }
                
                if (metrics.step_accuracy !== undefined) {
                    this._history.stepAccuracy.push({
                        epoch,
                        step,
                        phase,
                        value: metrics.step_accuracy
                    });
                    this._latestMetrics.accuracy = metrics.step_accuracy;
                }
                
                if (metrics.learning_rate !== undefined) {
                    this._history.learningRate.push({
                        epoch,
                        step,
                        value: metrics.learning_rate
                    });
                    this._latestMetrics.learningRate = metrics.learning_rate;
                }
                
            } else if (event.type === 'epoch') {
                // Epoch-level metrics
                if (phase === 'train') {
                    if (metrics.epoch_loss !== undefined || metrics.train_loss !== undefined) {
                        const lossVal = metrics.epoch_loss ?? metrics.train_loss;
                        this._history.epochTrainLoss.push({ epoch, value: lossVal });
                        this._latestMetrics.epochLoss = lossVal;
                    }
                    if (metrics.epoch_accuracy !== undefined || metrics.train_accuracy !== undefined) {
                        const accVal = metrics.epoch_accuracy ?? metrics.train_accuracy;
                        this._history.epochTrainAccuracy.push({ epoch, value: accVal });
                        this._latestMetrics.epochAccuracy = accVal;
                    }
                    // Learning rate from epoch summary (train phase only)
                    if (metrics.learning_rate !== undefined) {
                        this._history.learningRate.push({ epoch, step: 0, value: metrics.learning_rate });
                        this._latestMetrics.learningRate = metrics.learning_rate;
                    }
                } else if (phase === 'val') {
                    if (metrics.epoch_loss !== undefined || metrics.val_loss !== undefined) {
                        const lossVal = metrics.epoch_loss ?? metrics.val_loss;
                        this._history.epochValLoss.push({ epoch, value: lossVal });
                        this._latestMetrics.valLoss = lossVal;
                    }
                    if (metrics.epoch_accuracy !== undefined || metrics.val_accuracy !== undefined) {
                        const accVal = metrics.epoch_accuracy ?? metrics.val_accuracy;
                        this._history.epochValAccuracy.push({ epoch, value: accVal });
                        this._latestMetrics.valAccuracy = accVal;
                    }
                }
            }
        });
        
        return newEvents;
    },

    /**
     * Poll training status
     */
    _poll: async function() {
        if (!this._isRunning || !this._activeTrainingId) return;

        try {
            const status = await API.training.getStatus(this._activeTrainingId);
            
            if (status.error) {
                // Training not found or error
                this._state.status = 'error';
                this.stop();
                this._notifyListeners({
                    state: this._state,
                    metrics: this._latestMetrics,
                    newEvents: [],
                    error: status.error
                });
                return;
            }
            
            // Update state
            this._state.status = status.status;
            this._state.project = status.project;
            this._state.modelType = status.model_type;
            this._state.modelName = status.model_name;
            this._state.currentEpoch = status.current_epoch || 0;
            this._state.totalEpochs = status.total_epochs || 0;
            this._state.currentStep = status.current_step || 0;
            this._state.totalSteps = status.total_steps || 0;
            this._state.phase = status.phase;
            this._state.batchSize = status.batch_size;
            this._state.trainSamples = status.train_samples;
            this._state.valSamples = status.val_samples;
            
            // Process incremental updates
            const newEvents = this._processUpdates(status.updates);
            
            // Forward system stats to SystemMonitor (unified polling)
            if (status.system_stats && typeof SystemMonitor !== 'undefined') {
                SystemMonitor.updateFromStatus(status.system_stats);
            }
            
            // Notify listeners
            this._notifyListeners({
                state: this._state,
                metrics: this._latestMetrics,
                newEvents
            });
            
            // Stop monitoring if training completed/stopped
            if (status.status === 'completed' || status.status === 'stopped') {
                this.stop();
            }
            
        } catch (e) {
            console.error('TrainingMonitor poll error:', e);
        }
    },

    /**
     * Load full history for a training (useful when resuming monitoring)
     * @param {string} trainingId - Training ID
     */
    loadHistory: async function(trainingId) {
        try {
            // Get current status
            const status = await API.training.getStatus(trainingId);
            
            if (status.error) {
                console.warn('Training not found:', trainingId);
                return false;
            }
            
            // Load full history
            const history = await API.training.getHistory(trainingId);
            
            // Clear history first
            this.clearHistory();
            
            // Then set state from status (after clearHistory reset it)
            this._activeTrainingId = trainingId;
            this._state.status = status.status || 'running';
            this._state.trainingId = trainingId;
            this._state.project = status.project;
            this._state.modelType = status.model_type;
            this._state.modelName = status.model_name;
            this._state.currentEpoch = status.current_epoch || 0;
            this._state.totalEpochs = status.total_epochs || 0;
            this._state.currentStep = status.current_step || 0;
            this._state.totalSteps = status.total_steps || 0;
            this._state.phase = status.phase;
            this._state.batchSize = status.batch_size;
            this._state.trainSamples = status.train_samples;
            this._state.valSamples = status.val_samples;
            
            // Process history events
            if (history.events && history.events.length > 0) {
                this._processUpdates(history.events);
            }
            
            return true;
        } catch (e) {
            console.error('Failed to load training history:', e);
            return false;
        }
    }
};
