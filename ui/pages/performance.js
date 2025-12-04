/**
 * HootSight - Performance Page
 * System resource monitoring and training performance metrics
 */

const PerformancePage = {
    /**
     * Page identifier
     */
    name: 'performance',

    /**
     * Tab container instance
     */
    _tabs: null,

    /**
     * Container reference
     */
    _container: null,

    /**
     * System monitoring graphs
     */
    _systemGraphs: {
        cpu: null,
        memory: null,
        gpus: []
    },

    /**
     * Training monitoring graphs
     */
    _trainingGraphs: {
        trainStepLoss: null,
        valStepLoss: null,
        stepAccuracy: null,
        learningRate: null,
        epochLoss: null,
        epochAccuracy: null
    },

    /**
     * Info panel elements
     */
    _infoPanels: {
        cpu: null,
        memory: null,
        gpus: [],
        training: null
    },

    /**
     * Unsubscribe functions
     */
    _systemUnsubscribe: null,
    _trainingUnsubscribe: null,

    /**
     * Build the Performance page
     * @param {HTMLElement} container - Container element
     */
    build: async function(container) {
        container.innerHTML = '';
        this._container = container;
        
        // Cleanup previous subscriptions
        this._cleanupSubscriptions();

        // Create tabs container
        this._tabs = new Tabs('performance-tabs');

        // Build tabs
        const trainingTab = this._buildTrainingTab();
        const systemTab = this._buildSystemTab();

        this._tabs.addTab('training', lang('performance_page.tabs.training'), trainingTab);
        this._tabs.addTab('system', lang('performance_page.tabs.system'), systemTab);

        container.appendChild(this._tabs.getElement());

        // Handle tab changes
        this._tabs.onChange((tabId) => {
            if (tabId === 'system') {
                this._initSystemTab();
            } else if (tabId === 'training') {
                this._initTrainingTab();
            }
        });

        // Default to training tab and explicitly initialize it
        // (addTab auto-activates first tab, so onChange won't fire)
        this._tabs.activate('training');
        await this._initTrainingTab();
    },

    // ========================================
    // TRAINING TAB
    // ========================================

    /**
     * Build Training Performance tab content
     */
    _buildTrainingTab: function() {
        const content = document.createElement('div');
        content.className = 'performance-training-tab';
        content.id = 'training-tab-content';

        // Header
        const header = document.createElement('div');
        header.className = 'performance-header';
        
        const titleContainer = document.createElement('div');
        titleContainer.className = 'performance-title-container';
        
        const title = document.createElement('h3');
        title.className = 'performance-section-title';
        title.textContent = lang('performance_page.training.title');
        titleContainer.appendChild(title);
        
        // Training info subtitle
        const trainingInfo = document.createElement('div');
        trainingInfo.className = 'performance-hw-info';
        trainingInfo.id = 'training-info';
        trainingInfo.textContent = '';
        titleContainer.appendChild(trainingInfo);
        
        header.appendChild(titleContainer);

        // Status indicator
        const status = document.createElement('span');
        status.className = 'performance-status';
        status.id = 'training-status';
        status.textContent = lang('performance_page.training.no_active');
        header.appendChild(status);

        content.appendChild(header);

        // Graphs container
        const graphsContainer = document.createElement('div');
        graphsContainer.className = 'performance-graphs';
        graphsContainer.id = 'training-graphs';
        content.appendChild(graphsContainer);

        return content;
    },

    /**
     * Initialize Training tab
     */
    _initTrainingTab: async function() {
        const container = document.getElementById('training-graphs');
        if (!container) return;

        container.innerHTML = '';
        this._trainingGraphs = {
            trainStepLoss: null,
            valStepLoss: null,
            stepAccuracy: null,
            learningRate: null,
            epochLoss: null,
            epochAccuracy: null
        };

        // TrainingMonitor should already be running if there's active training
        // (started by TrainingController on app init)
        const isRunning = TrainingMonitor.isRunning();
        const history = TrainingMonitor.getHistory();
        const state = TrainingMonitor.getState();
        const hasData = history.stepLoss.length > 0 || history.epochTrainLoss.length > 0;

        // Update header info
        this._updateTrainingInfo(state);

        if (!hasData && !isRunning) {
            // Show placeholder
            const placeholder = document.createElement('div');
            placeholder.className = 'performance-placeholder';
            
            const title = document.createElement('h3');
            title.textContent = lang('performance_page.training.no_data');
            
            const desc = document.createElement('p');
            desc.textContent = lang('performance_page.training.description');
            
            placeholder.appendChild(title);
            placeholder.appendChild(desc);
            container.appendChild(placeholder);
            return;
        }

        // Create training graphs
        this._createTrainingGraphs(container, history, state);

        // Initialize info panels with current/historical data
        const metrics = TrainingMonitor.getLatestMetrics();
        this._updateTrainingInfoPanels(metrics, state);

        // Subscribe to live updates
        this._trainingUnsubscribe = TrainingMonitor.subscribe((data) => {
            this._onTrainingUpdate(data);
        });
    },

    /**
     * Create training graphs
     */
    _createTrainingGraphs: function(container, history, state) {
        // Train Step Loss graph - separate from validation
        this._trainingGraphs.trainStepLoss = new Graph('train-step-loss', {
            title: lang('performance_page.training.train_step_loss'),
            height: 180,
            unit: '',
            yMin: null,
            yMax: null,
            maxPoints: 2000,
            colors: ['#ef4444'],
            showLegend: false,
            smooth: false,
            animationDuration: 0
        });
        this._trainingGraphs.trainStepLoss.addSeries('train', { label: lang('performance_page.training.train'), color: '#ef4444' });
        
        history.stepLoss.filter(item => item.phase !== 'val').forEach(item => {
            this._trainingGraphs.trainStepLoss.append('train', item.value);
        });
        
        const trainStepLossCard = this._createGraphCard(this._trainingGraphs.trainStepLoss, 'train-step-loss-info');
        container.appendChild(trainStepLossCard.card);

        // Validation Step Loss graph - separate from training
        this._trainingGraphs.valStepLoss = new Graph('val-step-loss', {
            title: lang('performance_page.training.val_step_loss'),
            height: 180,
            unit: '',
            yMin: null,
            yMax: null,
            maxPoints: 2000,
            colors: ['#3b82f6'],
            showLegend: false,
            smooth: false,
            animationDuration: 0
        });
        this._trainingGraphs.valStepLoss.addSeries('val', { label: lang('performance_page.training.validation'), color: '#3b82f6' });
        
        history.stepLoss.filter(item => item.phase === 'val').forEach(item => {
            this._trainingGraphs.valStepLoss.append('val', item.value);
        });
        
        const valStepLossCard = this._createGraphCard(this._trainingGraphs.valStepLoss, 'val-step-loss-info');
        container.appendChild(valStepLossCard.card);

        // Step Accuracy graph - mirrors TensorBoard scalars
        this._trainingGraphs.stepAccuracy = new Graph('step-accuracy', {
            title: lang('performance_page.training.step_accuracy'),
            height: 160,
            unit: '%',
            yMin: 0,
            yMax: 100,
            maxPoints: 2000,
            colors: ['#14b8a6', '#8b5cf6'],
            showLegend: true,
            smooth: false,
            animationDuration: 0
        });
        this._trainingGraphs.stepAccuracy.addSeries('train', { label: lang('performance_page.training.train'), color: '#14b8a6' });
        this._trainingGraphs.stepAccuracy.addSeries('val', { label: lang('performance_page.training.validation'), color: '#8b5cf6' });

        history.stepAccuracy.forEach(item => {
            const series = item.phase === 'val' ? 'val' : 'train';
            this._trainingGraphs.stepAccuracy.append(series, item.value);
        });

        const stepAccuracyCard = this._createGraphCard(this._trainingGraphs.stepAccuracy, 'step-accuracy-info');
        container.appendChild(stepAccuracyCard.card);

        // Learning Rate graph - one point per epoch (TensorBoard standard)
        this._trainingGraphs.learningRate = new Graph('learning-rate', {
            title: lang('performance_page.training.learning_rate'),
            height: 140,
            unit: '',
            yMin: null,
            yMax: null,
            maxPoints: 500,  // One per epoch, plenty of room
            colors: ['#22c55e'],
            showLegend: false,
            smooth: false,
            animationDuration: 0
        });
        this._trainingGraphs.learningRate.addSeries('lr', { label: 'LR', color: '#22c55e' });
        
        // Restore LR history - only epoch-level entries (step === 0)
        const epochLrEntries = history.learningRate.filter(item => item.step === 0);
        epochLrEntries.forEach(item => {
            this._trainingGraphs.learningRate.append('lr', item.value);
        });
        
        const lrCard = this._createGraphCard(this._trainingGraphs.learningRate, 'lr-info');
        container.appendChild(lrCard.card);

        // Epoch Loss graph - keep more points (one per epoch)
        this._trainingGraphs.epochLoss = new Graph('epoch-loss', {
            title: lang('performance_page.training.epoch_loss'),
            height: 180,
            unit: '',
            yMin: null,
            yMax: null,
            maxPoints: 500,  // Epochs are few, can keep more
            colors: ['#ef4444', '#3b82f6'],
            showLegend: true,
            smooth: false,  // Disable animation
            animationDuration: 0
        });
        this._trainingGraphs.epochLoss.addSeries('train', { label: lang('performance_page.training.train'), color: '#ef4444' });
        this._trainingGraphs.epochLoss.addSeries('val', { label: lang('performance_page.training.validation'), color: '#3b82f6' });
        
        // Restore epoch loss history
        history.epochTrainLoss.forEach(item => {
            this._trainingGraphs.epochLoss.append('train', item.value);
        });
        history.epochValLoss.forEach(item => {
            this._trainingGraphs.epochLoss.append('val', item.value);
        });
        
        const epochLossCard = this._createGraphCard(this._trainingGraphs.epochLoss, 'epoch-loss-info');
        container.appendChild(epochLossCard.card);

        // Epoch Accuracy graph
        this._trainingGraphs.epochAccuracy = new Graph('epoch-accuracy', {
            title: lang('performance_page.training.epoch_accuracy'),
            height: 180,
            unit: '%',
            yMin: 0,
            yMax: 100,
            maxPoints: 500,  // Epochs are few, can keep more
            colors: ['#f59e0b', '#8b5cf6'],
            showLegend: true,
            smooth: false,  // Disable animation
            animationDuration: 0
        });
        this._trainingGraphs.epochAccuracy.addSeries('train', { label: lang('performance_page.training.train'), color: '#f59e0b' });
        this._trainingGraphs.epochAccuracy.addSeries('val', { label: lang('performance_page.training.validation'), color: '#8b5cf6' });
        
        // Restore epoch accuracy history
        history.epochTrainAccuracy.forEach(item => {
            this._trainingGraphs.epochAccuracy.append('train', item.value);
        });
        history.epochValAccuracy.forEach(item => {
            this._trainingGraphs.epochAccuracy.append('val', item.value);
        });
        
        const epochAccCard = this._createGraphCard(this._trainingGraphs.epochAccuracy, 'epoch-accuracy-info');
        container.appendChild(epochAccCard.card);

        // Update info panels
        this._updateTrainingInfoPanels(TrainingMonitor.getLatestMetrics(), state);
    },

    /**
     * Update training header info
     */
    _updateTrainingInfo: function(state) {
        const infoEl = document.getElementById('training-info');
        const statusEl = document.getElementById('training-status');
        
        if (infoEl && state.project) {
            const parts = [];
            parts.push(state.project);
            if (state.modelType && state.modelName) {
                parts.push(`${state.modelType}/${state.modelName}`);
            }
            if (state.totalEpochs) {
                parts.push(`Epoch ${state.currentEpoch}/${state.totalEpochs}`);
            }
            infoEl.textContent = parts.join(' | ');
        }
        
        if (statusEl) {
            if (state.status === 'running') {
                statusEl.textContent = lang('performance_page.training.running');
                statusEl.classList.add('active');
            } else if (state.status === 'completed') {
                statusEl.textContent = lang('performance_page.training.completed');
                statusEl.classList.remove('active');
            } else if (state.status === 'stopped') {
                statusEl.textContent = lang('performance_page.training.stopped');
                statusEl.classList.remove('active');
            } else {
                statusEl.textContent = lang('performance_page.training.no_active');
                statusEl.classList.remove('active');
            }
        }
    },

    /**
     * Update training info panels
     */
    _updateTrainingInfoPanels: function(metrics, state) {
        // Train Step Loss info
        const trainStepLossInfo = document.getElementById('train-step-loss-info');
        if (trainStepLossInfo && metrics) {
            const trainLossText = this._formatValue(metrics.trainStepLoss, '', v => Format.loss(v));
            let stepText = 'N/A';
            if (state.currentStep && state.totalSteps && state.phase !== 'val') {
                stepText = `${state.currentStep}/${state.totalSteps}`;
            }
            
            trainStepLossInfo.innerHTML = `
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.training.current')}:</span>
                    <span class="info-value">${trainLossText}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.training.step')}:</span>
                    <span class="info-value">${stepText}</span>
                </div>
            `;
        }

        // Validation Step Loss info
        const valStepLossInfo = document.getElementById('val-step-loss-info');
        if (valStepLossInfo && metrics) {
            const valLossText = this._formatValue(metrics.valStepLoss, '', v => Format.loss(v));
            let stepText = 'N/A';
            if (state.currentStep && state.totalSteps && state.phase === 'val') {
                stepText = `${state.currentStep}/${state.totalSteps}`;
            }
            
            valStepLossInfo.innerHTML = `
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.training.current')}:</span>
                    <span class="info-value">${valLossText}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.training.step')}:</span>
                    <span class="info-value">${stepText}</span>
                </div>
            `;
        }

        // Step Accuracy info
        const stepAccuracyInfo = document.getElementById('step-accuracy-info');
        if (stepAccuracyInfo && metrics) {
            const trainAccText = this._formatValue(metrics.trainStepAccuracy, '%', v => v.toFixed(2));
            const valAccText = this._formatValue(metrics.valStepAccuracy, '%', v => v.toFixed(2));

            stepAccuracyInfo.innerHTML = `
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.training.train_step_accuracy')}:</span>
                    <span class="info-value">${trainAccText}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.training.val_step_accuracy')}:</span>
                    <span class="info-value">${valAccText}</span>
                </div>
            `;
        }

        // LR info
        const lrInfo = document.getElementById('lr-info');
        if (lrInfo && metrics) {
            const lrText = this._formatValue(metrics.learningRate, '', v => v.toExponential(2));
            
            lrInfo.innerHTML = `
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.training.current_lr')}:</span>
                    <span class="info-value">${lrText}</span>
                </div>
            `;
        }

        // Epoch Loss info
        const epochLossInfo = document.getElementById('epoch-loss-info');
        if (epochLossInfo && metrics) {
            const trainLoss = this._formatValue(metrics.epochLoss, '', v => Format.loss(v));
            const valLoss = this._formatValue(metrics.valLoss, '', v => Format.loss(v));
            
            epochLossInfo.innerHTML = `
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.training.train_loss')}:</span>
                    <span class="info-value">${trainLoss}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.training.val_loss')}:</span>
                    <span class="info-value">${valLoss}</span>
                </div>
            `;
        }

        // Epoch Accuracy info
        const epochAccInfo = document.getElementById('epoch-accuracy-info');
        if (epochAccInfo && metrics) {
            const trainAcc = this._formatValue(metrics.epochAccuracy, '%', v => Format.percent(v, 2));
            const valAcc = this._formatValue(metrics.valAccuracy, '%', v => Format.percent(v, 2));
            
            epochAccInfo.innerHTML = `
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.training.train_accuracy')}:</span>
                    <span class="info-value">${trainAcc}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.training.val_accuracy')}:</span>
                    <span class="info-value">${valAcc}</span>
                </div>
            `;
        }
    },

    /**
     * Handle training update from TrainingMonitor
     */
    _onTrainingUpdate: function(data) {
        const { state, metrics, newEvents } = data;
        
        // Update header
        this._updateTrainingInfo(state);
        
        // Process new events for graphs
        newEvents.forEach(event => {
            const eventMetrics = event.metrics || {};
            
            if (event.type === 'step') {
                // Step loss - route to separate graphs based on phase
                if (eventMetrics.step_loss !== undefined) {
                    if (event.phase === 'val' && this._trainingGraphs.valStepLoss) {
                        this._trainingGraphs.valStepLoss.append('val', eventMetrics.step_loss);
                    } else if (this._trainingGraphs.trainStepLoss) {
                        this._trainingGraphs.trainStepLoss.append('train', eventMetrics.step_loss);
                    }
                }
                if (eventMetrics.step_accuracy !== undefined && this._trainingGraphs.stepAccuracy) {
                    const series = event.phase === 'val' ? 'val' : 'train';
                    this._trainingGraphs.stepAccuracy.append(series, eventMetrics.step_accuracy);
                }
                
            } else if (event.type === 'epoch') {
                // Epoch loss
                if (this._trainingGraphs.epochLoss) {
                    if (event.phase === 'train' && (eventMetrics.epoch_loss !== undefined || eventMetrics.train_loss !== undefined)) {
                        this._trainingGraphs.epochLoss.append('train', eventMetrics.epoch_loss ?? eventMetrics.train_loss);
                    } else if (event.phase === 'val' && (eventMetrics.epoch_loss !== undefined || eventMetrics.val_loss !== undefined)) {
                        this._trainingGraphs.epochLoss.append('val', eventMetrics.epoch_loss ?? eventMetrics.val_loss);
                    }
                }
                
                // Epoch accuracy
                if (this._trainingGraphs.epochAccuracy) {
                    if (event.phase === 'train' && (eventMetrics.epoch_accuracy !== undefined || eventMetrics.train_accuracy !== undefined)) {
                        this._trainingGraphs.epochAccuracy.append('train', eventMetrics.epoch_accuracy ?? eventMetrics.train_accuracy);
                    } else if (event.phase === 'val' && (eventMetrics.epoch_accuracy !== undefined || eventMetrics.val_accuracy !== undefined)) {
                        this._trainingGraphs.epochAccuracy.append('val', eventMetrics.epoch_accuracy ?? eventMetrics.val_accuracy);
                    }
                }
                
                // Learning rate - one point per epoch at train phase end (TensorBoard standard)
                if (event.phase === 'train' && eventMetrics.learning_rate !== undefined && this._trainingGraphs.learningRate) {
                    this._trainingGraphs.learningRate.append('lr', eventMetrics.learning_rate);
                }
            }
        });
        
        // Update info panels
        this._updateTrainingInfoPanels(metrics, state);
    },

    // ========================================
    // SYSTEM TAB
    // ========================================

    /**
     * Build System Usage tab content
     */
    _buildSystemTab: function() {
        const content = document.createElement('div');
        content.className = 'performance-system-tab';

        // Header
        const header = document.createElement('div');
        header.className = 'performance-header';
        
        const titleContainer = document.createElement('div');
        titleContainer.className = 'performance-title-container';
        
        const title = document.createElement('h3');
        title.className = 'performance-section-title';
        title.textContent = lang('performance_page.system.title');
        titleContainer.appendChild(title);
        
        // Hardware info subtitle
        const hwInfo = document.createElement('div');
        hwInfo.className = 'performance-hw-info';
        hwInfo.id = 'performance-hw-info';
        hwInfo.textContent = '';
        titleContainer.appendChild(hwInfo);
        
        header.appendChild(titleContainer);

        // Status indicator
        const status = document.createElement('span');
        status.className = 'performance-status';
        status.id = 'performance-status';
        status.textContent = SystemMonitor.isRunning() 
            ? lang('performance_page.system.monitoring_active')
            : lang('performance_page.system.monitoring_paused');
        if (SystemMonitor.isRunning()) {
            status.classList.add('active');
        }
        header.appendChild(status);

        content.appendChild(header);

        // Graphs container
        const graphsContainer = document.createElement('div');
        graphsContainer.className = 'performance-graphs';
        graphsContainer.id = 'performance-graphs';
        content.appendChild(graphsContainer);

        return content;
    },

    /**
     * Initialize System tab - create graphs and restore history
     */
    _initSystemTab: async function() {
        // Initialize graphs
        await this._initSystemGraphs();
        
        // Subscribe to live updates
        this._systemUnsubscribe = SystemMonitor.subscribe((stats) => {
            this._onSystemStatsUpdate(stats);
        });

        // Start monitoring if not running
        if (!SystemMonitor.isRunning()) {
            SystemMonitor.start();
        }

        // Update status indicator
        const status = document.getElementById('performance-status');
        if (status) {
            status.textContent = lang('performance_page.system.monitoring_active');
            status.classList.add('active');
        }
    },

    /**
     * Create a graph card with info panel below
     */
    _createGraphCard: function(graphInstance, infoId) {
        const card = document.createElement('div');
        card.className = 'performance-graph-card';
        
        // Graph
        card.appendChild(graphInstance.getElement());
        
        // Info panel
        const infoPanel = document.createElement('div');
        infoPanel.className = 'performance-info-panel';
        infoPanel.id = infoId;
        card.appendChild(infoPanel);
        
        return { card, infoPanel };
    },

    /**
     * Format bytes to human readable
     */
    _formatMemory: function(mb) {
        if (mb >= 1024) {
            return (mb / 1024).toFixed(1) + ' GB';
        }
        return Math.round(mb) + ' MB';
    },

    /**
     * Format value with N/A fallback
     */
    _formatValue: function(value, suffix = '', transform = null) {
        if (value === null || value === undefined) {
            return 'N/A';
        }
        const formatted = transform ? transform(value) : value;
        return suffix ? `${formatted}${suffix}` : String(formatted);
    },

    /**
     * Initialize system monitoring graphs
     */
    _initSystemGraphs: async function() {
        const container = document.getElementById('performance-graphs');
        if (!container) return;

        container.innerHTML = '';
        this._systemGraphs = { cpu: null, memory: null, gpus: [] };
        this._infoPanels = { cpu: null, memory: null, gpus: [] };

        const maxPoints = SystemMonitor.getMaxPoints();
        const history = SystemMonitor.getHistory();

        // CPU Usage graph - max 50 points (from SystemMonitor)
        this._systemGraphs.cpu = new Graph('cpu-usage', {
            title: lang('performance_page.system.cpu_usage'),
            height: 160,
            unit: '%',
            yMin: 0,
            yMax: 100,
            maxPoints: maxPoints,
            colors: ['#3b82f6'],
            showLegend: false,
            smooth: false,  // Disable animation for performance
            animationDuration: 0
        });
        this._systemGraphs.cpu.addSeries('usage', { label: 'CPU', color: '#3b82f6' });
        
        // Restore CPU history
        history.cpu.forEach(value => {
            this._systemGraphs.cpu.append('usage', value);
        });
        
        const cpuCard = this._createGraphCard(this._systemGraphs.cpu, 'cpu-info');
        this._infoPanels.cpu = cpuCard.infoPanel;
        container.appendChild(cpuCard.card);

        // System Memory graph
        this._systemGraphs.memory = new Graph('system-memory', {
            title: lang('performance_page.system.system_memory'),
            height: 160,
            unit: '%',
            yMin: 0,
            yMax: 100,
            maxPoints: maxPoints,
            colors: ['#22c55e'],
            showLegend: false,
            smooth: false,  // Disable animation for performance
            animationDuration: 0
        });
        this._systemGraphs.memory.addSeries('usage', { label: 'RAM', color: '#22c55e' });
        
        // Restore Memory history
        history.memory.forEach(value => {
            this._systemGraphs.memory.append('usage', value);
        });
        
        const memCard = this._createGraphCard(this._systemGraphs.memory, 'memory-info');
        this._infoPanels.memory = memCard.infoPanel;
        container.appendChild(memCard.card);

        // Get initial stats or use latest
        let stats = SystemMonitor.getLatestStats();
        if (!stats) {
            try {
                stats = await API.system.getStats();
            } catch (e) {
                console.warn('Could not get initial stats:', e);
            }
        }

        if (stats) {
            // Update hardware info header
            this._updateHwInfo(stats);
            
            // Create GPU graphs
            if (stats.gpus && stats.gpus.length > 0) {
                stats.gpus.forEach((gpu, index) => {
                    // GPU Usage graph
                    const gpuUsageGraph = new Graph(`gpu-${index}-usage`, {
                        title: lang('performance_page.system.gpu_usage').replace('{index}', index) + ': ' + (gpu.name || ''),
                        height: 160,
                        unit: '%',
                        yMin: 0,
                        yMax: 100,
                        maxPoints: maxPoints,
                        colors: ['#f59e0b'],
                        showLegend: false,
                        smooth: false,  // Disable animation for performance
                        animationDuration: 0
                    });
                    gpuUsageGraph.addSeries('usage', { label: gpu.name || `GPU ${index}`, color: '#f59e0b' });
                    
                    // Restore GPU usage history
                    if (history.gpus[index] && history.gpus[index].usage) {
                        history.gpus[index].usage.forEach(value => {
                            gpuUsageGraph.append('usage', value);
                        });
                    }
                    
                    const gpuUsageCard = this._createGraphCard(gpuUsageGraph, `gpu-${index}-usage-info`);
                    container.appendChild(gpuUsageCard.card);

                    // GPU Memory graph
                    const gpuMemGraph = new Graph(`gpu-${index}-memory`, {
                        title: lang('performance_page.system.gpu_memory').replace('{index}', index),
                        height: 160,
                        unit: '%',
                        yMin: 0,
                        yMax: 100,
                        maxPoints: maxPoints,
                        colors: ['#ef4444'],
                        showLegend: false,
                        smooth: false,  // Disable animation for performance
                        animationDuration: 0
                    });
                    gpuMemGraph.addSeries('usage', { label: 'VRAM', color: '#ef4444' });
                    
                    // Restore GPU memory history
                    if (history.gpus[index] && history.gpus[index].memory) {
                        history.gpus[index].memory.forEach(value => {
                            gpuMemGraph.append('usage', value);
                        });
                    }
                    
                    const gpuMemCard = this._createGraphCard(gpuMemGraph, `gpu-${index}-memory-info`);
                    container.appendChild(gpuMemCard.card);

                    this._systemGraphs.gpus.push({ usage: gpuUsageGraph, memory: gpuMemGraph });
                    this._infoPanels.gpus.push({ 
                        usage: gpuUsageCard.infoPanel, 
                        memory: gpuMemCard.infoPanel 
                    });
                });
            }
            
            // Update info panels with latest stats
            this._updateSystemInfoPanels(stats);
        }
    },

    /**
     * Update hardware info header
     */
    _updateHwInfo: function(stats) {
        const hwInfoEl = document.getElementById('performance-hw-info');
        if (!hwInfoEl) return;
        
        const parts = [];
        
        // Platform info
        if (stats.platform) {
            parts.push(`${stats.platform.system} ${stats.platform.release}`);
        }
        
        // CPU name
        if (stats.cpu_name && stats.cpu_name !== 'Unknown CPU') {
            parts.push(stats.cpu_name);
        }
        
        // GPU driver
        if (stats.gpus && stats.gpus.length > 0 && stats.gpus[0].driver_version) {
            parts.push(`Driver: ${stats.gpus[0].driver_version}`);
        }
        
        hwInfoEl.textContent = parts.join(' | ');
    },

    /**
     * Handle stats update from SystemMonitor
     */
    _onSystemStatsUpdate: function(stats) {
        // Update CPU graph
        if (this._systemGraphs.cpu && stats.cpu !== undefined) {
            this._systemGraphs.cpu.append('usage', stats.cpu);
        }

        // Update Memory graph
        if (this._systemGraphs.memory && stats.memory !== undefined) {
            this._systemGraphs.memory.append('usage', stats.memory);
        }

        // Update GPU graphs
        if (stats.gpus && this._systemGraphs.gpus.length > 0) {
            stats.gpus.forEach((gpu, index) => {
                if (this._systemGraphs.gpus[index]) {
                    if (gpu.usage !== undefined) {
                        this._systemGraphs.gpus[index].usage.append('usage', gpu.usage);
                    }
                    if (gpu.memory_percent !== undefined) {
                        this._systemGraphs.gpus[index].memory.append('usage', gpu.memory_percent);
                    }
                }
            });
        }
        
        // Update info panels
        this._updateSystemInfoPanels(stats);
    },

    /**
     * Update system info panels with latest stats
     */
    _updateSystemInfoPanels: function(stats) {
        // CPU info
        if (this._infoPanels.cpu) {
            const speedText = this._formatValue(stats.cpu_speed_mhz, ' MHz', Math.round);
            const tempText = this._formatValue(stats.cpu_temp, ' C', v => v.toFixed(1));
            const coresText = stats.cpu_cores && stats.cpu_threads 
                ? `${stats.cpu_cores} / ${stats.cpu_threads}` 
                : 'N/A';
            
            this._infoPanels.cpu.innerHTML = `
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.system.speed')}:</span>
                    <span class="info-value">${speedText}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.system.cores_threads')}:</span>
                    <span class="info-value">${coresText}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.system.temperature')}:</span>
                    <span class="info-value">${tempText}</span>
                </div>
            `;
        }
        
        // Memory info
        if (this._infoPanels.memory) {
            const totalText = this._formatMemory(stats.memory_total_mb);
            const availableText = this._formatMemory(stats.memory_available_mb);
            
            this._infoPanels.memory.innerHTML = `
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.system.max_memory')}:</span>
                    <span class="info-value">${totalText}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">${lang('performance_page.system.available')}:</span>
                    <span class="info-value">${availableText}</span>
                </div>
            `;
        }
        
        // GPU info
        if (stats.gpus && this._infoPanels.gpus.length > 0) {
            stats.gpus.forEach((gpu, index) => {
                if (this._infoPanels.gpus[index]) {
                    // GPU Usage panel
                    const tempText = this._formatValue(gpu.temperature, ' C');
                    const powerText = gpu.power_draw_w !== null && gpu.power_draw_w !== undefined 
                        ? `${Math.round(gpu.power_draw_w)}W / ${this._formatValue(gpu.power_limit_w, 'W', Math.round)}` 
                        : 'N/A';
                    const clockGpu = this._formatValue(gpu.clock_graphics_mhz, ' MHz');
                    const clockMem = this._formatValue(gpu.clock_memory_mhz, ' MHz');
                    const fanText = this._formatValue(gpu.fan_speed, '%');
                    
                    this._infoPanels.gpus[index].usage.innerHTML = `
                        <div class="info-row">
                            <span class="info-label">${lang('performance_page.system.temperature')}:</span>
                            <span class="info-value">${tempText}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">${lang('performance_page.system.power')}:</span>
                            <span class="info-value">${powerText}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">${lang('performance_page.system.gpu_clock')}:</span>
                            <span class="info-value">${clockGpu}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">${lang('performance_page.system.mem_clock')}:</span>
                            <span class="info-value">${clockMem}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">${lang('performance_page.system.fan')}:</span>
                            <span class="info-value">${fanText}</span>
                        </div>
                    `;
                    
                    // GPU Memory panel
                    const totalText = this._formatMemory(gpu.memory_total_mb);
                    const freeText = this._formatMemory(gpu.memory_free_mb);
                    const usedText = this._formatMemory(gpu.memory_used_mb);
                    
                    this._infoPanels.gpus[index].memory.innerHTML = `
                        <div class="info-row">
                            <span class="info-label">${lang('performance_page.system.dedicated_memory')}:</span>
                            <span class="info-value">${totalText}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">${lang('performance_page.system.used_memory')}:</span>
                            <span class="info-value">${usedText}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">${lang('performance_page.system.available')}:</span>
                            <span class="info-value">${freeText}</span>
                        </div>
                    `;
                }
            });
        }
    },

    // ========================================
    // CLEANUP
    // ========================================

    /**
     * Cleanup subscriptions
     */
    _cleanupSubscriptions: function() {
        if (this._systemUnsubscribe) {
            this._systemUnsubscribe();
            this._systemUnsubscribe = null;
        }
        if (this._trainingUnsubscribe) {
            this._trainingUnsubscribe();
            this._trainingUnsubscribe = null;
        }
    },

    /**
     * Cleanup when navigating away
     */
    cleanup: function() {
        this._cleanupSubscriptions();
        // Note: We DON'T stop monitors here - they continue in background
    }
};

// Register page
Pages.register('performance', PerformancePage);
