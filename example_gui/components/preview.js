class Preview {
    constructor(identifier, title = "", description = "", options = {}) {
        this.identifier = identifier;
        this.title = title;
        this.description = description;
        this.options = {
            // Fixed height: number (px) or CSS string (e.g., '60vh'). If null/undefined, auto.
            height: null,
            // If true and a FloatingWindow parent exists, fit to its window height dynamically.
            dynamicwindow: false,
            ...options
        };
        this.canvasElement = null;
        this.ctx = null;
        this.image = null;
    // Stores the original data URL if the image was set via base64
    this.originalDataURL = null;
    this.originalMimeType = null;
        this.container = null;
        this.resizeObserver = null;
        this.resizeTimeout = null;
    this.windowResizeObserver = null;
    this.windowResizeTimeout = null;
    this.closestWindowContainer = null;
    this.closestWindowContent = null;
    this.windowReady = false;
        this.scale = 1;
        this.panX = 0;
        this.panY = 0;
        this.isPanning = false;
        this.lastPanX = 0;
        this.lastPanY = 0;
        this.imageWidth = 0;
        this.imageHeight = 0;
        this.displayWidth = 0;
        this.displayHeight = 0;
        this.displayX = 0;
        this.displayY = 0;

        this.init();
    }

    init() {
        this.inputWrapper = Q('<div>', { class: 'preview_wrapper' }).get(0);

        if (this.title) {
            const heading = Q('<h3>', { class: 'inputs_title', text: this.title }).get(0);
            Q(this.inputWrapper).append(heading);
        }

        if (this.description) {
            const descriptionHeading = Q('<h4>', { class: 'inputs_description', text: this.description }).get(0);
            Q(this.inputWrapper).append(descriptionHeading);
        }

        this.canvasElement = Q('<canvas>', { class: 'preview-image' }).get(0);
        this.ctx = this.canvasElement.getContext("2d");

        Q(this.inputWrapper).append(this.canvasElement);

        this.setupEventListeners();
        this.setupResizeObserver();
        this.applyInitialSizing();
    }

    loadImageFromPath(imagePath, preserveView = false) {
        this.image = document.createElement('img');
        
        this.image.onload = () => {
            this.imageWidth = this.image.width;
            this.imageHeight = this.image.height;
            if (!preserveView) {
                this.resetZoomAndPan();
            }
            this.debouncedResize();
            
            // Trigger onChange callback on successful load
            if (this.onChangeCallback && typeof this.onChangeCallback === 'function') {
                try {
                    this.onChangeCallback(this.get());
                } catch (error) {
                    console.error('Preview: Error in onChange callback after load:', error);
                }
            }
        };
        
        this.image.onerror = () => {
            console.error('Preview: Failed to load image from path:', imagePath);
            this.image = null;
        };
        
        this.image.src = imagePath;
    }

    loadImageFromBase64(base64, preserveView = false) {
        this.image = document.createElement('img');
        
        this.image.onload = () => {
            this.imageWidth = this.image.width;
            this.imageHeight = this.image.height;
            if (!preserveView) {
                this.resetZoomAndPan();
            }
            this.debouncedResize();
            
            // Trigger onChange callback on successful load
            if (this.onChangeCallback && typeof this.onChangeCallback === 'function') {
                try {
                    this.onChangeCallback(this.get());
                } catch (error) {
                    console.error('Preview: Error in onChange callback after base64 load:', error);
                }
            }
        };
        
        this.image.onerror = () => {
            console.error('Preview: Failed to load base64 image');
            this.image = null;
        };
        
        this.image.src = base64;
    }

    debouncedResize() {
        if (this.resizeTimeout) {
            clearTimeout(this.resizeTimeout);
        }
        this.resizeTimeout = setTimeout(() => {
            this.resizeCanvas();
        }, 100);
    }

    setupEventListeners() {
        Q(this.canvasElement).on('wheel', (e) => {
            e.preventDefault();
            this.handleZoom(e);
        });

        Q(this.canvasElement).on('mousedown', (e) => {
            if (e.button === 1 || e.button === 0) {
                e.preventDefault();
                this.startPanning(e);
            }
        });

        Q(this.canvasElement).on('mousemove', (e) => {
            if (this.isPanning) {
                this.handlePanning(e);
            }
        });

        Q(this.canvasElement).on('mouseup', (e) => {
            if (e.button === 1 || e.button === 0) {
                this.stopPanning();
            }
        });

        Q(this.canvasElement).on('contextmenu', (e) => {
            e.preventDefault();
        });

        Q(this.canvasElement).on('dblclick', () => {
            this.resetSize();
        });
    }

    setupResizeObserver() {
        if (window.ResizeObserver) {
            this.resizeObserver = new ResizeObserver(() => {
                this.debouncedResize();
            });
            this.resizeObserver.observe(this.inputWrapper);
        }
        
        this.resizeHandler = () => this.debouncedResize();
        Q(window).on('resize', this.resizeHandler);

        // If dynamicwindow is enabled, observe the closest floating window for size changes
        const inheritedDynamic = this.inheritDynamicWindowFlag();
        if (this.options.dynamicwindow || inheritedDynamic) {
            this.options.dynamicwindow = true;
            this.ensureWindowBindings(/*retry*/true);
        }
    }

    applyInitialSizing() {
        // If dynamicwindow is requested and a window parent exists, it takes precedence
        if (this.options.dynamicwindow) {
            // Height will be applied after window animation completes
            this.ensureWindowBindings(true);
            return;
        }
        // Fixed height if provided
        if (this.options.height !== null && this.options.height !== undefined) {
            const h = this.coerceCssSize(this.options.height);
            Q(this.inputWrapper).css('height', h);
            this.debouncedResize();
        }
    }

    coerceCssSize(v) {
        if (v === null || v === undefined) return '';
        if (typeof v === 'number') return `${v}px`;
        if (typeof v === 'string') return v; // allow '60vh', '80%', '400px'
        return '';
    }

    findClosestWindowContainer() {
        try {
            const el = this.inputWrapper?.closest?.('.window_container');
            return el || null;
        } catch (_e) { return null; }
    }

    findClosestWindowContent(container) {
        try {
            if (!container) return null;
            const content = container.querySelector?.('.window_content');
            return content || null;
        } catch (_e) { return null; }
    }

    applyDynamicWindowHeight() {
        // Do not measure until we know the window finished opening
        const container = this.closestWindowContainer || this.findClosestWindowContainer();
        if (!this.windowReady) return false;
        if (container) {
            const isFading = container.classList.contains('window_fade_out');
            const isShown = container.classList.contains('window_show');
            if (!isShown || isFading) {
                return false; // wait until shown and not fading
            }
        }

        const container2 = this.closestWindowContainer || this.findClosestWindowContainer();
        if (container2) {
            const titlebar = container2.querySelector?.('.window_titlebar');
            const footer = container2.querySelector?.('.window_footer');
            const tb = titlebar ? titlebar.offsetHeight : 0;
            const ft = footer ? footer.offsetHeight : 0;
            
            // Available window content height
            let availableHeight = container2.clientHeight - tb - ft - 20; // base padding
            
            // If we're in a splitbox, calculate our proportional share
            const splitboxContainer = this.inputWrapper.closest('.splitbox-container');
            if (splitboxContainer) {
                const ourPanel = this.inputWrapper.closest('.split-panel');
                if (ourPanel) {
                    const flexValue = parseFloat(ourPanel.style.flex) || 0.5;
                    const allPanels = splitboxContainer.querySelectorAll('.split-panel');
                    let totalFlex = 0;
                    allPanels.forEach(panel => {
                        totalFlex += parseFloat(panel.style.flex) || 1;
                    });
                    
                    // Our proportional share of the available height
                    const ourProportion = flexValue / totalFlex;
                    availableHeight = Math.floor(availableHeight * ourProportion);
                }
            }
            
            // Subtract box padding and margins
            const boxElement = this.inputWrapper.closest('.box-component');
            if (boxElement) {
                const boxStyle = window.getComputedStyle(boxElement);
                const boxPaddingTop = parseFloat(boxStyle.paddingTop) || 0;
                const boxPaddingBottom = parseFloat(boxStyle.paddingBottom) || 0;
                const boxMarginTop = parseFloat(boxStyle.marginTop) || 0;
                const boxMarginBottom = parseFloat(boxStyle.marginBottom) || 0;
                availableHeight -= (boxPaddingTop + boxPaddingBottom + boxMarginTop + boxMarginBottom + 30);
            }
            
            if (availableHeight > 100) {
                Q(this.inputWrapper).css('height', `${availableHeight}px`);
                Q(this.inputWrapper).css('max-height', `${availableHeight}px`);
                return true;
            }
        }
        return false;
    }

    inheritDynamicWindowFlag() {
        try {
            const el = this.inputWrapper?.closest?.('[data-dynamicwindow="true"]');
            return !!el;
        } catch (_e) { return false; }
    }

    ensureWindowBindings(retry = false, attempt = 0) {
        this.closestWindowContainer = this.findClosestWindowContainer();
        this.closestWindowContent = this.findClosestWindowContent(this.closestWindowContainer);

        if (!this.closestWindowContainer) {
            if (retry && attempt < 20) {
                setTimeout(() => this.ensureWindowBindings(true, attempt + 1), 50);
            }
            return;
        }

        // Apply initial height immediately if window is already shown
        const container = this.closestWindowContainer;
        if (container.classList.contains('window_show') && !container.classList.contains('window_fade_out')) {
            this.windowReady = true;
            this.applyDynamicWindowHeight();
            this.debouncedResize();
        }

        // Listen for the end of the open animation (opacity/transform transitions)
        const onTransitionEnd = (e) => {
            if (!e || (e.propertyName !== 'opacity' && e.propertyName !== 'transform')) return;
            // Only act when window is fully shown
            if (!container.classList.contains('window_show') || container.classList.contains('window_fade_out')) return;
            // Apply height now that animation finished
            this.windowReady = true;
            this.applyDynamicWindowHeight();
            this.debouncedResize();
        };
        // Attach one-time listener
        container.addEventListener('transitionend', onTransitionEnd, { once: true });

        // Fallback timer in case transitionend doesn't fire (e.g., animations disabled)
        setTimeout(() => {
            if (!container.classList.contains('window_show') || container.classList.contains('window_fade_out')) return;
            this.windowReady = true;
            this.applyDynamicWindowHeight();
            this.debouncedResize();
        }, 300);

        // Observe content/container resize after it's shown
        if (window.ResizeObserver) {
            const target = this.closestWindowContent || this.closestWindowContainer;
            this.windowResizeObserver = new ResizeObserver(() => {
                if (this.windowResizeTimeout) clearTimeout(this.windowResizeTimeout);
                this.windowResizeTimeout = setTimeout(() => {
                    const ok = this.applyDynamicWindowHeight();
                    if (ok) this.debouncedResize();
                }, 120);
            });
            this.windowResizeObserver.observe(target);
        }
    }

    resizeCanvas() {
        if (!this.image || !this.canvasElement) return;

        const containerRect = this.inputWrapper.getBoundingClientRect();
        const availableWidth = containerRect.width - 16;
        const availableHeight = containerRect.height - 16;

        this.canvasElement.width = availableWidth;
        this.canvasElement.height = availableHeight;
        Q(this.canvasElement).css('width', availableWidth + 'px');
        Q(this.canvasElement).css('height', availableHeight + 'px');

        this.fitImageToCanvas();
        this.draw();
    }

    fitImageToCanvas() {
        const canvasWidth = this.canvasElement.width;
        const canvasHeight = this.canvasElement.height;

        const scaleX = canvasWidth / this.imageWidth;
        const scaleY = canvasHeight / this.imageHeight;
        const scale = Math.min(scaleX, scaleY);

        this.displayWidth = this.imageWidth * scale;
        this.displayHeight = this.imageHeight * scale;

        this.displayX = (canvasWidth - this.displayWidth) / 2;
        this.displayY = (canvasHeight - this.displayHeight) / 2;
    }

    resetZoomAndPan() {
        this.scale = 1;
        this.panX = 0;
        this.panY = 0;
    }

    resetSize() {
        this.resetZoomAndPan();
        this.fitImageToCanvas();
        this.draw();
    }

    handleZoom(e) {
        const rect = this.canvasElement.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = this.scale * zoomFactor;

        if (newScale < 0.1 || newScale > 10) return;

        const scaleDiff = newScale - this.scale;

        this.panX -= (mouseX - this.displayX - this.panX) * scaleDiff / this.scale;
        this.panY -= (mouseY - this.displayY - this.panY) * scaleDiff / this.scale;

        this.scale = newScale;
        this.draw();
    }

    startPanning(e) {
        this.isPanning = true;
        this.lastPanX = e.clientX;
        this.lastPanY = e.clientY;
        Q(this.canvasElement).css('cursor', 'grabbing');
    }

    handlePanning(e) {
        if (!this.isPanning) return;

        const deltaX = e.clientX - this.lastPanX;
        const deltaY = e.clientY - this.lastPanY;

        this.panX += deltaX;
        this.panY += deltaY;

        this.lastPanX = e.clientX;
        this.lastPanY = e.clientY;

        this.draw();
    }

    stopPanning() {
        this.isPanning = false;
        Q(this.canvasElement).css('cursor', 'default');
    }

    draw() {
        if (!this.image || !this.ctx) return;
        
        // Check if image is in broken state
        if (this.image.complete && this.image.naturalWidth === 0) {
            console.warn('Preview: Skipping draw - image is in broken state');
            return;
        }

        this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

        const scaledWidth = this.displayWidth * this.scale;
        const scaledHeight = this.displayHeight * this.scale;
        const x = this.displayX + this.panX;
        const y = this.displayY + this.panY;

        try {
            this.ctx.drawImage(this.image, x, y, scaledWidth, scaledHeight);
        } catch (error) {
            console.error('Preview: Failed to draw image:', error);
            this.image = null; // Clear broken image
        }
    }

    set(data) {
        if (!data || data.length === 0) {
            return;
        }

        // Store for potential onChange callback
        const previousValue = this.get();

        // Check if data is a base64 data URL
        if (typeof data === 'string' && data.startsWith('data:image/')) {
            // It's a base64 data URL
            this.originalDataURL = data;
            // Try to capture original mime type from data URL
            const match = /^data:([^;]+);base64,/i.exec(data);
            this.originalMimeType = match ? match[1] : null;
            this.loadImageFromBase64(data, true);
        } else if (typeof data === 'string' && data.startsWith('blob:')) {
            // It's a blob URL (binary WebSocket data)
            this.originalDataURL = null;
            this.originalMimeType = 'image/jpeg';
            this.loadImageFromPath(data, true);
        } else if (typeof data === 'string') {
            // It's a regular URL/path
            this.originalDataURL = null;
            this.originalMimeType = null;
            this.loadImageFromPath(data, true);
        } else if (data instanceof Blob) {
            // It's a Blob object (binary data)
            const blobUrl = URL.createObjectURL(data);
            this.originalDataURL = null;
            this.originalMimeType = data.type || 'image/jpeg';
            this.loadImageFromPath(blobUrl, true);
            
            // Clean up blob URL after image loads
            setTimeout(() => {
                URL.revokeObjectURL(blobUrl);
            }, 30000);
        } else if (data instanceof ArrayBuffer) {
            // It's binary data as ArrayBuffer
            const blob = new Blob([data], { type: 'image/jpeg' });
            const blobUrl = URL.createObjectURL(blob);
            this.originalDataURL = null;
            this.originalMimeType = 'image/jpeg';
            this.loadImageFromPath(blobUrl, true);
            
            // Clean up blob URL after image loads
            setTimeout(() => {
                URL.revokeObjectURL(blobUrl);
            }, 30000);
        } else {
            console.warn('Preview: Unknown data type for preview:', typeof data, data);
        }

        // Trigger onChange callback if set and value actually changed
        if (this.onChangeCallback && typeof this.onChangeCallback === 'function') {
            const newValue = this.get();
            if (newValue !== previousValue) {
                try {
                    this.onChangeCallback(newValue);
                } catch (error) {
                    console.error('Preview: Error in onChange callback:', error);
                }
            }
        }
    }

    /**
     * Returns a Base64 data URL.
     * Default: ORIGINAL full image (independent of pan/zoom), to avoid accidental crop.
     * To get the current viewport instead, pass { viewport: true } (or { original: false }).
     *
     * @param {Object|boolean} [options] - Options or boolean shorthand.
     *   If boolean true, same as { original: true }.
     *   If object, supports:
     *     - original: boolean (default true)
     *     - viewport: boolean (alternative to original:false)
     *     - type: string (e.g. 'image/png', 'image/jpeg')
     *     - quality: number (0-1, JPEG/WebP)
     */
    get(options) {
        if (!this.canvasElement) return null;

        // New default: original. Explicit opt-in required for viewport export.
        const explicitViewport = options && (options.viewport === true || options.original === false || options.mode === 'viewport');
        const useOriginal = !explicitViewport && (options === undefined || options === null || options === true || options.original !== false);
        const type = options && options.type ? options.type : undefined;
        const quality = options && typeof options.quality === 'number' ? options.quality : undefined;

        if (useOriginal) {
            return this.getOriginalBase64(type, quality);
        }
        return this.getViewportBase64(type, quality);
    }

    /** Returns the current canvas (viewport) as Base64, including pan/zoom. */
    getViewportBase64(type, quality) {
        if (!this.canvasElement) return null;
        try {
            return type ? this.canvasElement.toDataURL(type, quality) : this.canvasElement.toDataURL();
        } catch (e) {
            return this.canvasElement.toDataURL();
        }
    }

    /** Returns the original full image as Base64, independent of pan/zoom. */
    getOriginalBase64(type, quality) {
        // If we were given a base64 originally, return that to preserve bytes/mime
        if (this.originalDataURL) {
            if (!type) return this.originalDataURL;
            // If caller requested a different type, convert using an offscreen canvas
        }
        if (!this.image) return null;

    const off = Q('<canvas>').get(0);
        off.width = this.imageWidth || this.image.naturalWidth || this.image.width || 0;
        off.height = this.imageHeight || this.image.naturalHeight || this.image.height || 0;
        const offCtx = off.getContext('2d');
        offCtx.drawImage(this.image, 0, 0, off.width, off.height);
        const mime = type || this.originalMimeType || 'image/png';
        try {
            return off.toDataURL(mime, quality);
        } catch (e) {
            return off.toDataURL();
        }
    }

    // Aliases for clarity
    getOriginal(options) {
        const type = options && options.type ? options.type : undefined;
        const quality = options && typeof options.quality === 'number' ? options.quality : undefined;
        return this.getOriginalBase64(type, quality);
    }
    getViewport(options) {
        const type = options && options.type ? options.type : undefined;
        const quality = options && typeof options.quality === 'number' ? options.quality : undefined;
        return this.getViewportBase64(type, quality);
    }

    getElement() {
        return this.inputWrapper;
    }

    // Components system expects these methods
    getValue() {
        // Return the current image data (base64 by default)
        return this.get();
    }

    setValue(value) {
        // Set new image data
        this.set(value);
    }

    onChange(callback) {
        // Preview doesn't really have "change" events in the traditional sense
        // since it's mainly a display component, but we can store the callback
        // for potential future use (e.g., when user manipulates the image)
        this.onChangeCallback = callback;
    }

    destroy() {
        if (this.resizeTimeout) {
            clearTimeout(this.resizeTimeout);
        }
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
        if (this.resizeHandler) {
            Q(window).off('resize', this.resizeHandler);
        }
        if (this.windowResizeTimeout) {
            clearTimeout(this.windowResizeTimeout);
        }
        if (this.windowResizeObserver) {
            try { this.windowResizeObserver.disconnect(); } catch (_e) {}
            this.windowResizeObserver = null;
        }
        
        if (this.inputWrapper && this.inputWrapper.parentNode) {
            this.inputWrapper.parentNode.removeChild(this.inputWrapper);
        }
    }
}
