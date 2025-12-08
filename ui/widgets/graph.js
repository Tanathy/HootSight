/**
 * Graph Widget
 * Canvas-based line graph with auto-scaling Y-axis
 * 
 * Features:
 *   - Auto-scaling Y-axis based on data min/max
 *   - Configurable max data points (default 500)
 *   - Canvas rendering for high performance with many points
 *   - Multiple series support with different colors
 *   - Dark theme styling
 *   - Real-time data appending
 *   - Optional value smoothing animation
 */
class Graph {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            title: options.title || '',
            titleLangKey: options.titleLangKey || null,
            maxPoints: options.maxPoints || 500,
            height: options.height || 200,
            showGrid: options.showGrid !== false,
            showLegend: options.showLegend !== false,
            showArea: options.showArea !== false,  // Show gradient area under lines
            unit: options.unit || '',
            yMin: options.yMin ?? null,  // null = auto
            yMax: options.yMax ?? null,  // null = auto
            colors: options.colors || ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'],
            gridLines: options.gridLines || 5,
            animationDuration: options.animationDuration || 300,
            smooth: options.smooth !== false,  // Enable value smoothing animation
            smoothDuration: options.smoothDuration || 250,  // Animation duration in ms
            lineWidth: options.lineWidth || 2,
            ...options
        };
        
        this._series = {};  // { seriesName: { data: [], displayData: [], color: string, label: string } }
        this._element = null;
        this._canvas = null;
        this._ctx = null;
        this._legendContainer = null;
        this._valueDisplay = null;
        this._colorIndex = 0;
        this._animationFrame = null;
        this._lastRenderTime = 0;
        
        this._width = 0;
        this._height = this.options.height;
        this._padding = { top: 20, right: 20, bottom: 30, left: 50 };
        this._dpr = window.devicePixelRatio || 1;
        
        this._build();
        this._setupResize();
    }
    
    _build() {
        // Main container
        this._element = Q('<div>', { class: 'graph-widget', id: `graph-${this.id}` }).get();
        
        // Header with title and current values
        if (this.options.title || this.options.showLegend) {
            const header = Q('<div>', { class: 'graph-header' }).get();
            
            if (this.options.title) {
                const title = Q('<div>', { class: 'graph-title', text: this.options.title }).get();
                if (this.options.titleLangKey) {
                    title.setAttribute('data-lang-key', this.options.titleLangKey);
                }
                Q(header).append(title);
            }
            
            // Value display area
            this._valueDisplay = Q('<div>', { class: 'graph-values' }).get();
            Q(header).append(this._valueDisplay);
            
            Q(this._element).append(header);
        }
        
        // Canvas container
        const canvasContainer = Q('<div>', { class: 'graph-canvas-container' }).get();
        canvasContainer.style.position = 'relative';
        canvasContainer.style.width = '100%';
        canvasContainer.style.minHeight = '100px';
        
        this._canvas = Q('<canvas>', { class: 'graph-canvas' }).get();
        this._canvas.style.display = 'block';
        this._canvas.style.width = '100%';
        this._canvas.style.height = `${this._height}px`;
        this._ctx = this._canvas.getContext('2d');
        
        Q(canvasContainer).append(this._canvas);
        Q(this._element).append(canvasContainer);
        
        // Legend
        if (this.options.showLegend) {
            this._legendContainer = Q('<div>', { class: 'graph-legend' }).get();
            Q(this._element).append(this._legendContainer);
        }
    }
    
    _setupResize() {
        const resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const rect = entry.contentRect;
                if (rect.width > 0) {
                    this._width = rect.width;
                    this._updateCanvasSize();
                    this._render();
                }
            }
        });
        
        // Observe after element is in DOM
        requestAnimationFrame(() => {
            const container = this._canvas?.parentElement;
            if (container) {
                resizeObserver.observe(container);
                this._width = container.clientWidth || 400;
                this._updateCanvasSize();
            }
        });
    }
    
    _updateCanvasSize() {
        if (!this._canvas) return;
        
        // Set canvas size accounting for device pixel ratio for crisp rendering
        this._canvas.width = this._width * this._dpr;
        this._canvas.height = this._height * this._dpr;
        this._canvas.style.width = `${this._width}px`;
        this._canvas.style.height = `${this._height}px`;
        
        // Scale context for DPR
        this._ctx.setTransform(this._dpr, 0, 0, this._dpr, 0, 0);
    }
    
    /**
     * Add or update a data series
     * @param {string} name - Series identifier
     * @param {Object} config - { label, color, data }
     */
    addSeries(name, config = {}) {
        const color = config.color || this.options.colors[this._colorIndex % this.options.colors.length];
        this._colorIndex++;
        
        this._series[name] = {
            data: config.data || [],
            displayData: config.data ? [...config.data] : [],  // Animated display values
            color: color,
            label: config.label || name
        };
        
        this._updateLegend();
        this._render();
    }
    
    /**
     * Append a single value to a series
     * @param {string} name - Series name
     * @param {number} value - Value to append
     */
    append(name, value) {
        if (!this._series[name]) {
            this.addSeries(name);
        }
        
        const series = this._series[name];
        series.data.push(value);
        
        // Trim to max points
        if (series.data.length > this.options.maxPoints) {
            series.data.shift();
        }
        
        // For smoothing: add to displayData with animation
        if (this.options.smooth) {
            // Start from previous last value or current value
            const startValue = series.displayData.length > 0 
                ? series.displayData[series.displayData.length - 1] 
                : value;
            series.displayData.push(startValue);
            
            // Trim displayData to match
            if (series.displayData.length > this.options.maxPoints) {
                series.displayData.shift();
            }
            
            // Start animation
            this._startSmoothAnimation();
        } else {
            // No smoothing - displayData matches data
            series.displayData = [...series.data];
            this._render();
        }
        
        this._updateValueDisplay();
    }
    
    /**
     * Start smooth animation loop
     */
    _startSmoothAnimation() {
        if (this._animationFrame) return;  // Already running
        
        this._lastRenderTime = performance.now();
        this._animateSmoothValues();
    }
    
    /**
     * Animate display values toward actual values
     */
    _animateSmoothValues() {
        const now = performance.now();
        const deltaTime = now - this._lastRenderTime;
        this._lastRenderTime = now;
        
        const duration = this.options.smoothDuration;
        const progress = Math.min(deltaTime / duration, 1);
        
        let needsMoreAnimation = false;
        
        Object.values(this._series).forEach(series => {
            for (let i = 0; i < series.data.length; i++) {
                const target = series.data[i];
                const current = series.displayData[i] ?? target;
                
                if (current !== target) {
                    // Ease toward target
                    const diff = target - current;
                    const step = diff * Math.min(progress * 3, 1);  // Faster convergence
                    
                    if (Math.abs(diff) < 0.01) {
                        series.displayData[i] = target;
                    } else {
                        series.displayData[i] = current + step;
                        needsMoreAnimation = true;
                    }
                }
            }
        });
        
        this._render();
        
        if (needsMoreAnimation) {
            this._animationFrame = requestAnimationFrame(() => this._animateSmoothValues());
        } else {
            this._animationFrame = null;
        }
    }
    
    /**
     * Set all data for a series
     * @param {string} name - Series name
     * @param {number[]} data - Array of values
     */
    setData(name, data) {
        if (!this._series[name]) {
            this.addSeries(name);
        }
        
        const series = this._series[name];
        series.data = data.slice(-this.options.maxPoints);
        series.displayData = [...series.data];  // No animation for bulk set
        
        this._render();
        this._updateValueDisplay();
    }
    
    /**
     * Clear all data from a series or all series
     * @param {string} [name] - Series name (omit to clear all)
     */
    clear(name) {
        if (name) {
            if (this._series[name]) {
                this._series[name].data = [];
                this._series[name].displayData = [];
            }
        } else {
            Object.keys(this._series).forEach(n => {
                this._series[n].data = [];
                this._series[n].displayData = [];
            });
        }
        this._render();
        this._updateValueDisplay();
    }
    
    /**
     * Calculate Y-axis bounds from all data
     */
    _calculateBounds() {
        let min = this.options.yMin;
        let max = this.options.yMax;
        
        if (min === null || max === null) {
            let dataMin = Infinity;
            let dataMax = -Infinity;
            
            Object.values(this._series).forEach(series => {
                series.data.forEach(v => {
                    if (v < dataMin) dataMin = v;
                    if (v > dataMax) dataMax = v;
                });
            });
            
            // Handle empty data
            if (dataMin === Infinity) dataMin = 0;
            if (dataMax === -Infinity) dataMax = 100;
            
            // Add padding
            const range = dataMax - dataMin || 1;
            const padding = range * 0.1;
            
            if (min === null) min = Math.max(0, dataMin - padding);
            if (max === null) max = dataMax + padding;
        }
        
        // Ensure we have a range
        if (max <= min) max = min + 1;
        
        return { min, max };
    }
    
    /**
     * Render all series and grid on canvas
     */
    _render() {
        if (!this._width || !this._ctx) return;
        
        const ctx = this._ctx;
        const bounds = this._calculateBounds();
        const chartWidth = this._width - this._padding.left - this._padding.right;
        const chartHeight = this._height - this._padding.top - this._padding.bottom;
        
        // Clear canvas
        ctx.clearRect(0, 0, this._width, this._height);
        
        // Render grid
        this._renderGrid(ctx, bounds, chartWidth, chartHeight);
        
        // Render each series (area first, then lines on top)
        Object.values(this._series).forEach(series => {
            this._renderSeriesArea(ctx, series, bounds, chartWidth, chartHeight);
        });
        Object.values(this._series).forEach(series => {
            this._renderSeriesLine(ctx, series, bounds, chartWidth, chartHeight);
        });
    }
    
    _renderGrid(ctx, bounds, chartWidth, chartHeight) {
        if (!this.options.showGrid) return;
        
        const { min, max } = bounds;
        const range = max - min;
        
        // Get computed styles for colors
        const style = getComputedStyle(document.documentElement);
        const gridColor = style.getPropertyValue('--border-light').trim() || 'rgba(255,255,255,0.1)';
        const textColor = style.getPropertyValue('--text-muted').trim() || 'rgba(255,255,255,0.5)';
        
        ctx.save();
        ctx.strokeStyle = gridColor;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.font = '12px system-ui, -apple-system, sans-serif';
        ctx.fillStyle = textColor;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        
        // Horizontal grid lines
        for (let i = 0; i <= this.options.gridLines; i++) {
            const ratio = i / this.options.gridLines;
            const y = this._padding.top + chartHeight * (1 - ratio);
            const value = min + range * ratio;
            
            // Grid line
            ctx.beginPath();
            ctx.moveTo(this._padding.left, y);
            ctx.lineTo(this._width - this._padding.right, y);
            ctx.stroke();
            
            // Y-axis label
            ctx.fillText(this._formatValue(value), this._padding.left - 8, y);
        }
        
        ctx.restore();
    }
    
    _renderSeriesArea(ctx, series, bounds, chartWidth, chartHeight) {
        if (!this.options.showArea) return;
        
        const renderData = this.options.smooth ? series.displayData : series.data;
        if (!renderData.length) return;
        
        const points = this._getSeriesPoints(renderData, bounds, chartWidth, chartHeight);
        if (points.length < 2) return;
        
        const baseY = this._padding.top + chartHeight;
        
        ctx.save();
        ctx.beginPath();
        this._tracePathToCtx(ctx, points);
        ctx.lineTo(points[points.length - 1].x, baseY);
        ctx.lineTo(points[0].x, baseY);
        ctx.closePath();
        
        // Create gradient for area fill
        const gradient = ctx.createLinearGradient(0, this._padding.top, 0, baseY);
        const color = series.color || '#3498db';
        gradient.addColorStop(0, this._hexToRgba(color, 0.3));
        gradient.addColorStop(1, this._hexToRgba(color, 0));
        ctx.fillStyle = gradient;
        ctx.fill();
        ctx.restore();
    }
    
    _renderSeriesLine(ctx, series, bounds, chartWidth, chartHeight) {
        const renderData = this.options.smooth ? series.displayData : series.data;
        if (!renderData.length) return;
        
        const points = this._getSeriesPoints(renderData, bounds, chartWidth, chartHeight);
        if (points.length < 2) return;
        
        ctx.save();
        ctx.strokeStyle = series.color || '#3498db';
        ctx.lineWidth = this.options.lineWidth || 2;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        
        ctx.beginPath();
        this._tracePathToCtx(ctx, points);
        ctx.stroke();
        ctx.restore();
    }
    
    _getSeriesPoints(data, bounds, chartWidth, chartHeight) {
        const { min, max } = bounds;
        const range = max - min || 1;
        const points = [];
        
        data.forEach((value, i) => {
            const x = this._padding.left + (i / Math.max(1, data.length - 1)) * chartWidth;
            const y = this._padding.top + chartHeight * (1 - (value - min) / range);
            points.push({ x, y });
        });
        
        // Handle single point
        if (points.length === 1) {
            points.push({ ...points[0], x: points[0].x + 1 });
        }
        
        return points;
    }
    
    /**
     * Trace smooth Catmull-Rom path to canvas context
     */
    _tracePathToCtx(ctx, points) {
        if (points.length < 2) return;
        
        ctx.moveTo(points[0].x, points[0].y);
        
        for (let i = 0; i < points.length - 1; i++) {
            const p0 = points[Math.max(0, i - 1)];
            const p1 = points[i];
            const p2 = points[i + 1];
            const p3 = points[Math.min(points.length - 1, i + 2)];
            
            // Catmull-Rom to Bezier conversion
            const tension = 0.5;
            const cp1x = p1.x + (p2.x - p0.x) * tension / 3;
            const cp1y = p1.y + (p2.y - p0.y) * tension / 3;
            const cp2x = p2.x - (p3.x - p1.x) * tension / 3;
            const cp2y = p2.y - (p3.y - p1.y) * tension / 3;
            
            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
        }
    }
    
    _hexToRgba(hex, alpha) {
        // Handle rgb/rgba pass-through
        if (hex.startsWith('rgb')) return hex;
        
        // Convert shorthand hex
        if (hex.length === 4) {
            hex = '#' + hex[1] + hex[1] + hex[2] + hex[2] + hex[3] + hex[3];
        }
        
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    
    _formatValue(value) {
        // Use Format utility if available, otherwise fallback
        if (typeof Format !== 'undefined' && Format.graphAxis) {
            return Format.graphAxis(value);
        }
        
        // Fallback implementation
        if (Math.abs(value) >= 1000000) {
            return (value / 1000000).toFixed(1) + 'M';
        } else if (Math.abs(value) >= 1000) {
            return (value / 1000).toFixed(1) + 'K';
        } else if (Math.abs(value) < 0.001 && value !== 0) {
            const exp = Math.floor(Math.log10(Math.abs(value)));
            const mantissa = value / Math.pow(10, exp);
            return `${mantissa.toFixed(1)}e${exp}`;
        } else if (Math.abs(value) < 1 && value !== 0) {
            return value.toFixed(3);
        } else {
            return Math.round(value).toString();
        }
    }
    
    _updateLegend() {
        if (!this._legendContainer) return;
        
        Q(this._legendContainer).empty();
        
        Object.entries(this._series).forEach(([name, series]) => {
            const item = Q('<div>', { class: 'graph-legend-item' }).get();
            
            const dot = Q('<span>', { class: 'graph-legend-dot' }).get();
            dot.style.backgroundColor = series.color;
            
            const label = Q('<span>', { class: 'graph-legend-label', text: series.label }).get();
            
            Q(item).append(dot);
            Q(item).append(label);
            Q(this._legendContainer).append(item);
        });
    }
    
    _updateValueDisplay() {
        if (!this._valueDisplay) return;
        
        Q(this._valueDisplay).empty();
        
        Object.entries(this._series).forEach(([name, series]) => {
            const lastValue = series.data[series.data.length - 1];
            if (lastValue === undefined) return;
            
            const item = Q('<span>', { 
                class: 'graph-value-item',
                text: `${this._formatValue(lastValue)}${this.options.unit}`
            }).get();
            item.style.color = series.color;
            Q(this._valueDisplay).append(item);
        });
    }
    
    /**
     * Get the DOM element
     * @returns {HTMLElement}
     */
    getElement() {
        return this._element;
    }
    
    /**
     * Get current data for a series
     * @param {string} name - Series name
     * @returns {number[]}
     */
    getData(name) {
        return this._series[name]?.data.slice() || [];
    }
    
    /**
     * Destroy the widget
     */
    destroy() {
        // Cancel any pending animation
        if (this._animationFrame) {
            cancelAnimationFrame(this._animationFrame);
            this._animationFrame = null;
        }
        
        if (this._element && this._element.parentNode) {
            this._element.parentNode.removeChild(this._element);
        }
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Graph;
}
