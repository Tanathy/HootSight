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
        const headerProgress = Q('.header-progress').get();
        if (headerProgress) {
            // Clear existing static content
            Q(headerProgress).empty();
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
            Q(this._container).append(bar.element);
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
            Q(bar.element).addClass('hiding');
            setTimeout(() => {
                Q(bar.element).remove();
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
        const wrapper = Q('<div>', { class: 'header-progress-item', 'data-progress-id': id });

        const labelEl = Q('<span>', { class: 'progress-label', text: options.label || id }).get();

        const barContainer = Q('<div>', { class: 'progress-bar' });

        const fill = Q('<div>', { class: 'progress-fill' }).get();
        fill.style.width = '0%';
        barContainer.append(fill);

        const statusEl = Q('<span>', { class: 'progress-status', text: options.status || '' }).get();

        wrapper.append(labelEl).append(barContainer.get()).append(statusEl);

        return {
            element: wrapper.get(),
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
            Q(bar.label).text(options.label);
        }
        if (options.progress !== undefined) {
            bar.progress = Math.min(100, Math.max(0, options.progress));
            bar.fill.style.width = bar.progress + '%';
            
            // Add completion class
            if (bar.progress >= 100) {
                Q(bar.element).addClass('complete');
            } else {
                Q(bar.element).removeClass('complete');
            }
        }
        if (options.status !== undefined) {
            bar.status = options.status;
            Q(bar.statusEl).text(options.status);
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
    Q(document).on('DOMContentLoaded', () => ProgressManager.init());
} else {
    ProgressManager.init();
}
