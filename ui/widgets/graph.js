/**
 * Graph Widget
 * SVG path-based line graph with auto-scaling Y-axis
 * 
 * Features:
 *   - Auto-scaling Y-axis based on data min/max
 *   - Configurable max data points (default 500)
 *   - SVG path rendering for smooth lines
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
            maxPoints: options.maxPoints || 500,
            height: options.height || 200,
            showGrid: options.showGrid !== false,
            showLegend: options.showLegend !== false,
            unit: options.unit || '',
            yMin: options.yMin ?? null,  // null = auto
            yMax: options.yMax ?? null,  // null = auto
            colors: options.colors || ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'],
            gridLines: options.gridLines || 5,
            animationDuration: options.animationDuration || 300,
            smooth: options.smooth !== false,  // Enable value smoothing animation
            smoothDuration: options.smoothDuration || 250,  // Animation duration in ms
            ...options
        };
        
        this._series = {};  // { seriesName: { data: [], displayData: [], color: string, label: string } }
        this._element = null;
        this._svg = null;
        this._pathGroup = null;
        this._gridGroup = null;
        this._yLabels = null;
        this._legendContainer = null;
        this._valueDisplay = null;
        this._colorIndex = 0;
        this._animationFrame = null;
        this._lastRenderTime = 0;
        
        this._width = 0;
        this._height = this.options.height;
        this._padding = { top: 20, right: 20, bottom: 30, left: 50 };
        
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
                Q(header).append(title);
            }
            
            // Value display area
            this._valueDisplay = Q('<div>', { class: 'graph-values' }).get();
            Q(header).append(this._valueDisplay);
            
            Q(this._element).append(header);
        }
        
        // SVG container
        const svgContainer = Q('<div>', { class: 'graph-svg-container' }).get();
        
        this._svg = Q('<svg>', { class: 'graph-svg', preserveAspectRatio: 'none' }).get();
        
        // Grid group (behind paths)
        this._gridGroup = Q('<g>', { class: 'graph-grid' }).get();
        Q(this._svg).append(this._gridGroup);
        
        // Path group
        this._pathGroup = Q('<g>', { class: 'graph-paths' }).get();
        Q(this._svg).append(this._pathGroup);
        
        // Y-axis labels group
        this._yLabels = Q('<g>', { class: 'graph-y-labels' }).get();
        Q(this._svg).append(this._yLabels);
        
        Q(svgContainer).append(this._svg);
        Q(this._element).append(svgContainer);
        
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
                    this._updateViewBox();
                    this._render();
                }
            }
        });
        
        // Observe after element is in DOM
        requestAnimationFrame(() => {
            const container = Q(this._element).find('.graph-svg-container').get(0);
            if (container) {
                resizeObserver.observe(container);
                this._width = container.clientWidth || 400;
                this._updateViewBox();
            }
        });
    }
    
    _updateViewBox() {
        this._svg.setAttribute('viewBox', `0 0 ${this._width} ${this._height}`);
        this._svg.style.width = '100%';
        this._svg.style.height = `${this._height}px`;
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
            label: config.label || name,
            path: null
        };
        
        // Create path element
        const path = Q('<path>', { 
            class: 'graph-line',
            stroke: color,
            fill: 'none',
            'stroke-width': '2',
            'stroke-linecap': 'round',
            'stroke-linejoin': 'round'
        }).get();
        Q(this._pathGroup).append(path);
        this._series[name].path = path;
        
        // Create area fill
        const area = Q('<path>', {
            class: 'graph-area',
            fill: color,
            opacity: '0.1'
        }).get();
        this._pathGroup.insertBefore(area, path);
        this._series[name].area = area;
        
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
     * Render all series paths and grid
     */
    _render() {
        if (!this._width) return;
        
        const bounds = this._calculateBounds();
        const chartWidth = this._width - this._padding.left - this._padding.right;
        const chartHeight = this._height - this._padding.top - this._padding.bottom;
        
        // Render grid
        this._renderGrid(bounds, chartWidth, chartHeight);
        
        // Render each series
        Object.values(this._series).forEach(series => {
            this._renderSeries(series, bounds, chartWidth, chartHeight);
        });
    }
    
    _renderGrid(bounds, chartWidth, chartHeight) {
        // Clear existing grid
        Q(this._gridGroup).empty();
        Q(this._yLabels).empty();
        
        if (!this.options.showGrid) return;
        
        const { min, max } = bounds;
        const range = max - min;
        
        // Horizontal grid lines
        for (let i = 0; i <= this.options.gridLines; i++) {
            const ratio = i / this.options.gridLines;
            const y = this._padding.top + chartHeight * (1 - ratio);
            const value = min + range * ratio;
            
            // Grid line
            const line = Q('<line>', {
                x1: this._padding.left,
                y1: y,
                x2: this._width - this._padding.right,
                y2: y,
                class: 'graph-grid-line'
            }).get();
            Q(this._gridGroup).append(line);
            
            // Y-axis label
            const label = Q('<text>', {
                x: this._padding.left - 8,
                y: y,
                class: 'graph-label',
                'text-anchor': 'end',
                'dominant-baseline': 'middle',
                text: this._formatValue(value)
            }).get();
            Q(this._yLabels).append(label);
        }
    }
    
    _renderSeries(series, bounds, chartWidth, chartHeight) {
        // Use displayData for rendering (animated values)
        const renderData = this.options.smooth ? series.displayData : series.data;
        
        if (!renderData.length) {
            series.path.setAttribute('d', '');
            series.area.setAttribute('d', '');
            return;
        }
        
        const { min, max } = bounds;
        const range = max - min;
        const points = [];
        
        renderData.forEach((value, i) => {
            const x = this._padding.left + (i / Math.max(1, renderData.length - 1)) * chartWidth;
            const y = this._padding.top + chartHeight * (1 - (value - min) / range);
            points.push({ x, y });
        });
        
        // Handle single point
        if (points.length === 1) {
            points.push({ ...points[0], x: points[0].x + 1 });
        }
        
        // Build smooth path using Catmull-Rom spline
        const pathD = this._buildSmoothPath(points);
        series.path.setAttribute('d', pathD);
        
        // Build area path
        const areaD = pathD + 
            ` L ${points[points.length - 1].x} ${this._padding.top + chartHeight}` +
            ` L ${points[0].x} ${this._padding.top + chartHeight} Z`;
        series.area.setAttribute('d', areaD);
    }
    
    /**
     * Build a smooth SVG path using simplified Catmull-Rom
     */
    _buildSmoothPath(points) {
        if (points.length < 2) return '';
        
        let d = `M ${points[0].x} ${points[0].y}`;
        
        for (let i = 0; i < points.length - 1; i++) {
            const p0 = points[Math.max(0, i - 1)];
            const p1 = points[i];
            const p2 = points[i + 1];
            const p3 = points[Math.min(points.length - 1, i + 2)];
            
            // Control points
            const cp1x = p1.x + (p2.x - p0.x) / 6;
            const cp1y = p1.y + (p2.y - p0.y) / 6;
            const cp2x = p2.x - (p3.x - p1.x) / 6;
            const cp2y = p2.y - (p3.y - p1.y) / 6;
            
            d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2.x} ${p2.y}`;
        }
        
        return d;
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
