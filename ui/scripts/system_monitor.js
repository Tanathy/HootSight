/**
 * HootSight - System Monitor Service
 * Background monitoring service that persists across page navigation
 * Can receive data from TrainingMonitor during training (unified polling)
 */

const SystemMonitor = {
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
     * Max data points to keep in history
     */
    _maxPoints: 50,

    /**
     * Whether we're receiving data from TrainingMonitor
     */
    _receivingFromTraining: false,

    /**
     * Historical data storage
     */
    _history: {
        cpu: [],
        memory: [],
        gpus: {}  // Keyed by GPU index
    },

    /**
     * Latest stats snapshot
     */
    _latestStats: null,

    /**
     * Callbacks for live updates
     */
    _listeners: [],

    /**
     * Start background monitoring (only polls if not receiving from training)
     */
    start: function() {
        if (this._isRunning) return;
        this._isRunning = true;
        
        // Only start own polling if not receiving from TrainingMonitor
        if (!this._receivingFromTraining) {
            this._poll();
            this._pollInterval = setInterval(() => this._poll(), this._pollRate);
        }
    },

    /**
     * Stop background monitoring
     */
    stop: function() {
        this._isRunning = false;
        
        if (this._pollInterval) {
            clearInterval(this._pollInterval);
            this._pollInterval = null;
        }
    },

    /**
     * Update stats from TrainingMonitor (unified polling)
     * Called by TrainingMonitor when it receives system_stats in status response
     */
    updateFromStatus: function(stats) {
        if (!stats) return;
        
        this._receivingFromTraining = true;
        
        // Stop own polling since we're getting data from TrainingMonitor
        if (this._pollInterval) {
            clearInterval(this._pollInterval);
            this._pollInterval = null;
        }
        
        this._processStats(stats);
    },

    /**
     * Called when TrainingMonitor stops - resume own polling if needed
     */
    resumeOwnPolling: function() {
        this._receivingFromTraining = false;
        
        if (this._isRunning && !this._pollInterval) {
            this._poll();
            this._pollInterval = setInterval(() => this._poll(), this._pollRate);
        }
    },

    /**
     * Check if monitoring is active
     */
    isRunning: function() {
        return this._isRunning;
    },

    /**
     * Get current history data
     */
    getHistory: function() {
        return {
            cpu: [...this._history.cpu],
            memory: [...this._history.memory],
            gpus: JSON.parse(JSON.stringify(this._history.gpus))
        };
    },

    /**
     * Get latest stats
     */
    getLatestStats: function() {
        return this._latestStats;
    },

    /**
     * Get max points setting
     */
    getMaxPoints: function() {
        return this._maxPoints;
    },

    /**
     * Register a listener for live updates
     * @param {Function} callback - Called with stats on each poll
     * @returns {Function} Unsubscribe function
     */
    subscribe: function(callback) {
        this._listeners.push(callback);
        
        // Return unsubscribe function
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
    _notifyListeners: function(stats) {
        this._listeners.forEach(cb => {
            try {
                cb(stats);
            } catch (e) {
                console.error('SystemMonitor listener error:', e);
            }
        });
    },

    /**
     * Process stats and store in history
     */
    _processStats: function(stats) {
        this._latestStats = stats;

        // Store CPU history
        if (stats.cpu !== undefined) {
            this._history.cpu.push(stats.cpu);
            if (this._history.cpu.length > this._maxPoints) {
                this._history.cpu.shift();
            }
        }

        // Store Memory history
        if (stats.memory !== undefined) {
            this._history.memory.push(stats.memory);
            if (this._history.memory.length > this._maxPoints) {
                this._history.memory.shift();
            }
        }

        // Store GPU history
        if (stats.gpus && stats.gpus.length > 0) {
            stats.gpus.forEach((gpu, index) => {
                if (!this._history.gpus[index]) {
                    this._history.gpus[index] = { usage: [], memory: [] };
                }
                
                if (gpu.usage !== undefined) {
                    this._history.gpus[index].usage.push(gpu.usage);
                    if (this._history.gpus[index].usage.length > this._maxPoints) {
                        this._history.gpus[index].usage.shift();
                    }
                }
                
                if (gpu.memory_percent !== undefined) {
                    this._history.gpus[index].memory.push(gpu.memory_percent);
                    if (this._history.gpus[index].memory.length > this._maxPoints) {
                        this._history.gpus[index].memory.shift();
                    }
                }
            });
        }

        // Notify listeners
        this._notifyListeners(stats);
    },

    /**
     * Poll system stats (only used when not receiving from TrainingMonitor)
     */
    _poll: async function() {
        if (!this._isRunning || this._receivingFromTraining) return;

        try {
            const stats = await API.system.getStats();
            this._processStats(stats);
        } catch (e) {
            console.error('SystemMonitor poll error:', e);
        }
    },

    /**
     * Clear all history
     */
    clearHistory: function() {
        this._history = {
            cpu: [],
            memory: [],
            gpus: {}
        };
        this._latestStats = null;
    }
};
