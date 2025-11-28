/**
 * Progress Manager Widget
 * Manages multiple progress bars in the header
 */

const ProgressManager = {
    _container: null,
    _progressBars: new Map(), // id -> { element, label, progress, status }

    /**
     * Initialize the progress manager
     */
    init: function() {
        // Find or create the container
        const headerProgress = document.querySelector('.header-progress');
        if (headerProgress) {
            // Clear existing static content
            headerProgress.innerHTML = '';
            this._container = headerProgress;
        }
    },

    /**
     * Create or update a progress bar
     * @param {string} id - Unique identifier for this progress
     * @param {Object} options - { label, progress (0-100), status }
     */
    show: function(id, options = {}) {
        if (!this._container) this.init();
        if (!this._container) return;

        let bar = this._progressBars.get(id);
        
        if (!bar) {
            // Create new progress bar
            bar = this._createProgressBar(id, options);
            this._progressBars.set(id, bar);
            this._container.appendChild(bar.element);
        }

        // Update values
        this._updateProgressBar(bar, options);
    },

    /**
     * Update progress value
     * @param {string} id - Progress bar ID
     * @param {number} progress - Progress percentage (0-100)
     * @param {string} status - Optional status text
     */
    update: function(id, progress, status = null) {
        const bar = this._progressBars.get(id);
        if (bar) {
            this._updateProgressBar(bar, { progress, status });
        }
    },

    /**
     * Hide/remove a progress bar
     * @param {string} id - Progress bar ID
     */
    hide: function(id) {
        const bar = this._progressBars.get(id);
        if (bar) {
            bar.element.classList.add('hiding');
            setTimeout(() => {
                bar.element.remove();
                this._progressBars.delete(id);
            }, 300);
        }
    },

    /**
     * Hide all progress bars
     */
    hideAll: function() {
        for (const id of this._progressBars.keys()) {
            this.hide(id);
        }
    },

    /**
     * Create a new progress bar element
     */
    _createProgressBar: function(id, options) {
        const wrapper = document.createElement('div');
        wrapper.className = 'header-progress-item';
        wrapper.dataset.progressId = id;

        const labelEl = document.createElement('span');
        labelEl.className = 'progress-label';
        labelEl.textContent = options.label || id;

        const barContainer = document.createElement('div');
        barContainer.className = 'progress-bar';

        const fill = document.createElement('div');
        fill.className = 'progress-fill';
        fill.style.width = '0%';
        barContainer.appendChild(fill);

        const statusEl = document.createElement('span');
        statusEl.className = 'progress-status';
        statusEl.textContent = options.status || '';

        wrapper.appendChild(labelEl);
        wrapper.appendChild(barContainer);
        wrapper.appendChild(statusEl);

        return {
            element: wrapper,
            label: labelEl,
            fill: fill,
            statusEl: statusEl,
            progress: 0,
            status: ''
        };
    },

    /**
     * Update progress bar values
     */
    _updateProgressBar: function(bar, options) {
        if (options.label !== undefined) {
            bar.label.textContent = options.label;
        }
        if (options.progress !== undefined) {
            bar.progress = Math.min(100, Math.max(0, options.progress));
            bar.fill.style.width = bar.progress + '%';
            
            // Add completion class
            if (bar.progress >= 100) {
                bar.element.classList.add('complete');
            } else {
                bar.element.classList.remove('complete');
            }
        }
        if (options.status !== undefined) {
            bar.status = options.status;
            bar.statusEl.textContent = options.status;
        }
    },

    /**
     * Check if a progress bar exists
     */
    has: function(id) {
        return this._progressBars.has(id);
    },

    /**
     * Get progress value
     */
    getProgress: function(id) {
        const bar = this._progressBars.get(id);
        return bar ? bar.progress : 0;
    }
};

// Auto-init when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => ProgressManager.init());
} else {
    ProgressManager.init();
}
