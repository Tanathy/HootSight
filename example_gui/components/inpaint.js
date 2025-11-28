class Inpaint {
    constructor(identifier, title = "", description = "", mode = "paint") {
        this.identifier = identifier;
        this.mode = mode;
        this.title = title;
        this.description = description;
        this.canvas = null;
        this.ctx = null;
        this.maskCanvas = null;
        this.maskCtx = null;
        this.image = null;
        this.scale = 1;
        this.panX = 0;
        this.panY = 0;
        this.isPanning = false;
        this.isPainting = false;
        this.lastPanX = 0;
        this.lastPanY = 0;
        this.imageWidth = 0;
        this.imageHeight = 0;
        this.displayWidth = 0;
        this.displayHeight = 0;
        this.displayX = 0;
        this.displayY = 0;
        this.brushSize = 50;  // Default brush size
        this.isErasing = false;
        this.cursorDiv = null;
        this.imageCanvas = null;
        this.imageCtx = null;
        this.brushColor = "#ffffff";
        this.lastPaintX = undefined;
    this.lastPaintY = undefined;
    this.lastPaintPressure = 1;
        this.pendingImageData = null; // Store image data when tab is not visible
        this.visibilityObserver = null; // Observer to detect when component becomes visible
        this.resizeObserver = null; // Observer to detect container size changes
        this.pixelRatio = Math.max(1, window.devicePixelRatio || 1);
    // History (undo/redo) state
    this.undoStack = [];
    this.redoStack = [];
    this.historyLimit = 10;
    this.isHovered = false; // only react to shortcuts when over canvas

        this.init();
    }

    init() {
        this.inputWrapper = Q('<div>', { class: 'inpaint_wrapper' }).get(0);

        if (this.title) {
              const heading = Q('<h3>', { class: 'inputs_title', text: this.title }).get(0);
              Q(this.inputWrapper).append(heading);
        }

        if (this.description) {
              const descriptionHeading = Q('<h4>', { class: 'inputs_description', text: this.description }).get(0);
              Q(this.inputWrapper).append(descriptionHeading);
        }

        const inputContent = Q('<div>', { class: 'input_content' }).get(0);

        this.container = Q('<div>', { class: 'image-canvas-container' }).get(0);
        this.container.setAttribute("id", this.identifier);
        Q(inputContent).append(this.container);
        Q(this.inputWrapper).append(inputContent);

        this.canvas = Q('<canvas>').get(0);
        this.canvas.className = this.mode === "paint" ? "paint-canvas" : "mask-canvas";
        Q(this.container).append(this.canvas);

        this.maskCanvas = Q('<canvas>').get(0);
        this.maskCanvas.className = "mask-overlay";
        Q(this.container).append(this.maskCanvas);

        this.cursorDiv = Q('<div>').get(0);
        this.cursorDiv.className = "brush-cursor";
        Q(this.container).append(this.cursorDiv);

        this.ctx = this.canvas.getContext('2d');
        this.maskCtx = this.maskCanvas.getContext('2d');
    // Hard-disable touch gestures on element level (safety belt)
    this.canvas.style.touchAction = 'none';
    this.container.style.touchAction = 'none';
    this.canvas.style.msTouchAction = 'none';
        this.createSettingsPanel();
        this.setupEventListeners();
        this.setupVisibilityObserver();
        this.setupResizeObserver();

        // Initial sizing after DOM paint
        setTimeout(() => this.resizeCanvasToContainer(true), 50);
    }

    setupVisibilityObserver() {
        // Use IntersectionObserver to detect when the component becomes visible
        this.visibilityObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && entry.intersectionRatio > 0) {
                    console.log('Inpaint: Component became visible');
                    this.onBecameVisible();
                }
            });
        }, {
            threshold: 0.1 // Trigger when at least 10% visible
        });

        this.visibilityObserver.observe(this.container);
    }

    setupResizeObserver() {
        // Use ResizeObserver to detect container size changes
        // This handles split panel resizing, popup window changes, etc.
        this.resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    this.resizeCanvasToContainer();
                }
            }
        });

        this.resizeObserver.observe(this.container);
    }

    onBecameVisible() {
        // Check if we need to resize canvas
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        if (width > 0 && height > 0) {
            // Resize canvas if needed
            if (this.canvas.width !== width || this.canvas.height !== height) {
                console.log('Inpaint: Resizing canvas on visibility:', width, 'x', height);
                this.resizeCanvasToContainer();
            }

            // If we have pending image data, load it now
            if (this.pendingImageData) {
                console.log('Inpaint: Loading pending image data');
                const imageData = this.pendingImageData;
                this.pendingImageData = null;
                this.loadImageFromBase64(imageData);
            } else if (this.image) {
                // Redraw existing image
                this.fitImageToCanvas();
                this.draw();
            }
        }
    }

    resizeCanvasToContainer(force = false) {
        // Measure CSS pixel size of the container
        const cssWidth = Math.floor(this.container.clientWidth);
        const cssHeight = Math.floor(this.container.clientHeight);

        if (cssWidth <= 0 || cssHeight <= 0) return;

        // Track current CSS size to avoid redundant work
        const sameCssSize = (this.canvas.style.width === cssWidth + 'px' && this.canvas.style.height === cssHeight + 'px');
        if (!force && sameCssSize) return;

        // Set CSS size
        this.canvas.style.width = cssWidth + 'px';
        this.canvas.style.height = cssHeight + 'px';
        this.maskCanvas.style.width = cssWidth + 'px';
        this.maskCanvas.style.height = cssHeight + 'px';

        // Internal pixel buffer considering devicePixelRatio
        const dpr = Math.max(1, window.devicePixelRatio || 1);
        this.pixelRatio = dpr;
        this.canvas.width = Math.round(cssWidth * dpr);
        this.canvas.height = Math.round(cssHeight * dpr);
        this.maskCanvas.width = Math.round(cssWidth * dpr);
        this.maskCanvas.height = Math.round(cssHeight * dpr);

        // Scale contexts so drawing code uses CSS pixels
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.maskCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

        if (this.image) {
            this.fitImageToCanvas();
            this.draw();
        } else {
            // Clear visible area when there's no image yet
            this.ctx.clearRect(0, 0, cssWidth, cssHeight);
            this.maskCtx.clearRect(0, 0, cssWidth, cssHeight);
        }
    }

    // Removed duplicate init() definition that overwrote observers and broke resizing.

    createSettingsPanel() {
    this.settingsPanel = Q('<div>', { class: 'inpaint_settings_panel' }).get(0);

    // Color picker: wrapper + inner input so CSS can make a perfect circle
    this.colorPickerWrapper = Q('<div>', { class: 'color_picker' }).get(0);
    this.colorSwatch = Q('<div>', { class: 'color_swatch' }).get(0);
    this.colorPicker = Q('<input>', { type: 'color' }).get(0);
    this.colorPicker.value = this.brushColor;

    this.sizeSlider = new Slider("brush_size", 1, 200, 1, this.brushSize, "");
    // Icon toggle button (brush <-> eraser)
    this.toolToggleButton = Q('<button>', { class: 'inpaint-tool-toggle', html: UI_ICONS.brush, 'data-tooltip': 'ui.inpaint.brush_eraser_toggle'}).get(0);
    this.clearButton = Q('<button>', { class: 'clear_button', html: UI_ICONS.clear, 'data-tooltip': 'ui.clipboard.clear' }).get(0);
        // Source buttons (T2I / I2I)
        this.t2iButton = Q('<button>', { class: 'inpaint-source-btn', text: 'T2I', 'data-tooltip': 'ui.inpaint.use_text2image_preview' }).get(0);
        this.i2iButton = Q('<button>', { class: 'inpaint-source-btn', text: 'I2I', 'data-tooltip': 'ui.inpaint.use_image2image_preview' }).get(0);

        Q(this.settingsPanel).append(
            this.colorPickerWrapper,
            this.sizeSlider.getElement(),
            this.toolToggleButton,
            this.clearButton
        );
        const sourceGroup = Q('<div>', { class: 'inpaint-source-group' }).get(0);
        sourceGroup.append(this.t2iButton, this.i2iButton);

        Q(this.settingsPanel).append(
            sourceGroup
        );

        Q(this.container).append(this.settingsPanel);

        // mount the input inside the wrapper after panel is in DOM
    // Visual swatch + invisible input overlay
    this.colorPickerWrapper.appendChild(this.colorSwatch);
    this.colorPickerWrapper.appendChild(this.colorPicker);
    this.colorSwatch.style.backgroundColor = this.brushColor;

        Q(this.colorPicker).on("change", () => {
            this.brushColor = this.colorPicker.value;
            this.colorSwatch.style.backgroundColor = this.brushColor;
        });

        // Listen to the actual input field change, not the wrapper
        const sizeSliderInput = this.sizeSlider.getElement().querySelector('input');
        if (sizeSliderInput) {
            Q(sizeSliderInput).on("change", () => {
                this.brushSize = this.sizeSlider.get();
                console.log('Inpaint: Brush size changed to:', this.brushSize);
            });
        }

        // Toggle tool button click
        Q(this.toolToggleButton).on('click', (e) => {
            e.stopPropagation();
            this.toggleTool();
        });

        Q(this.clearButton).on("click", () => {
            this.clearMask();
        });

        // Source buttons actions
        Q(this.t2iButton).on('click', (e) => {
            e.stopPropagation();
            const data = this.getPreviewData('img-preview', 'tab-text-to-image');
            if (data) this.set(data);
        });
        Q(this.i2iButton).on('click', (e) => {
            e.stopPropagation();
            const data = this.getPreviewData('i2i-img-preview', 'tab-image-to-image');
            if (data) this.set(data);
        });

        // Periodically update button visibility based on availability
        this.updateSourceButtonsVisibility();
        this.sourceButtonsInterval = setInterval(() => this.updateSourceButtonsVisibility(), 1500);
    }

    setupEventListeners() {
    Q(this.canvas).on('wheel', (e) => {
            e.preventDefault();
            this.handleZoom(e);
        });

        // Prefer Pointer Events for pen pressure
        if (window.PointerEvent) {
            Q(this.canvas).on('pointerdown', (e) => {
                // Prevent scrollbars / OS gestures from stealing the pointer
                if (typeof e.preventDefault === 'function') e.preventDefault();
                this.pointerActive = true;
                if (e.button === 1) {
                    e.preventDefault();
                    this.startPanning(e);
                } else if (e.button === 0) {
                    if (this.canvas.setPointerCapture && e.pointerId != null) {
                        try { this.canvas.setPointerCapture(e.pointerId); } catch (_) {}
                    }
                    this.startPainting(e);
                }
            });
            Q(this.canvas).on('pointermove', (e) => {
                this.updateCursor(e);
                if (this.isPanning) this.handlePanning(e);
                else if (this.isPainting) this.handlePainting(e);
            });
            Q(this.canvas).on('pointerup', (e) => {
                if (e.button === 1) this.stopPanning();
                else if (e.button === 0) this.stopPainting();
                if (this.canvas.releasePointerCapture && e.pointerId != null) {
                    try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) {}
                }
                this.pointerActive = false;
            });
            Q(this.canvas).on('pointercancel', () => {
                this.pointerActive = false;
                this.stopPainting();
                this.stopPanning();
            });
        }

    Q(this.canvas).on('mousedown', (e) => {
            // Switching to mouse input; ensure pointerActive is cleared
            this.pointerActive = false;
            if (this.pointerActive) return;
            if (e.button === 1) {
                e.preventDefault();
                this.startPanning(e);
            } else if (e.button === 0) {
                this.startPainting(e);
            }
        });

    Q(this.canvas).on('mousemove', (e) => {
            if (this.pointerActive) return;
            this.updateCursor(e);
            if (this.isPanning) {
                this.handlePanning(e);
            } else if (this.isPainting) {
                this.handlePainting(e);
            }
        });

    Q(this.canvas).on('mouseup', (e) => {
            if (this.pointerActive) return;
            if (e.button === 1) {
                this.stopPanning();
            } else if (e.button === 0) {
                this.stopPainting();
            }
        });

        Q(this.canvas).on('mouseenter', () => {
            this.isHovered = true;
            if (this.image) {
                this.cursorDiv.style.display = 'block';
            }
        });

        Q(this.canvas).on('mouseleave', () => {
            this.isHovered = false;
            this.pointerActive = false;
            this.cursorDiv.style.display = 'none';
        });

        // If we ever leave pointer context explicitly
        if (window.PointerEvent) {
            Q(this.canvas).on('pointerleave', () => { this.pointerActive = false; });
            Q(this.canvas).on('pointerout', () => { this.pointerActive = false; });
        }

    Q(this.canvas).on('contextmenu', (e) => {
            e.preventDefault();
        });

    Q(document).on('keydown', (e) => {
            // Local tool toggle (E)
            if (this.isHovered && (e.key === 'e' || e.key === 'E')) {
                this.toggleTool(true);
                e.preventDefault();
                e.stopPropagation();
                return;
            }

            // Undo/Redo only when hovering the canvas
            if (!this.isHovered) return;

            const key = (e.key || '').toLowerCase();
            const isCtrl = e.ctrlKey || e.metaKey; // meta for mac just in case
            if (isCtrl && key === 'z') {
                this.undo();
                e.preventDefault();
                e.stopPropagation();
                return;
            }
            if (isCtrl && (key === 'y' || (key === 'z' && e.shiftKey))) {
                this.redo();
                e.preventDefault();
                e.stopPropagation();
                return;
            }
        });

    Q(document).on('keyup', (e) => {
            // Keyup not used; keep toggled state
        });

        // Drag and drop support for image upload - use native events for dataTransfer
        const containerEl = this.container;
        
        containerEl.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            containerEl.classList.add('drag-over');
        }, false);

        containerEl.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            // Only remove class if leaving the container itself
            if (e.target === containerEl) {
                containerEl.classList.remove('drag-over');
            }
        }, false);

        containerEl.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            containerEl.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                this.handleImageUpload(files[0]);
            }
        }, false);

        // Upload button for manual image selection
        const uploadButton = Q('<button>', { 
            class: 'inpaint-upload-button', 
            html: UI_ICONS.upload, 
            'data-tooltip': 'ui.inpaint.upload_image'
        }).get(0);
        
        Q(uploadButton).on('click', (e) => {
            e.stopPropagation();
            this.triggerFileSelect();
        });
        
    Q(this.settingsPanel).append(uploadButton);
    }

    toggleTool(fromKeyboard = false) {
        // Flip state
        this.isErasing = !this.isErasing;
        // Update icon
        this.toolToggleButton.innerHTML = this.isErasing ? UI_ICONS.eraser : UI_ICONS.brush;
        console.log('Inpaint: Tool changed to', this.isErasing ? 'Eraser' : 'Brush', fromKeyboard ? '(keyboard)' : '(click)');
    }

    triggerFileSelect() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleImageUpload(file);
            }
        };
        input.click();
    }

    handleImageUpload(file) {
        console.log('Inpaint: handleImageUpload called with file:', file);
        
        if (!file.type.startsWith('image/')) {
            console.warn('Inpaint: File is not an image:', file.type);
            return;
        }

        console.log('Inpaint: Reading file as DataURL...');
        const reader = new FileReader();
        reader.onload = (e) => {
            console.log('Inpaint: FileReader loaded, calling loadImageFromBase64...');
            this.loadImageFromBase64(e.target.result);
        };
        reader.onerror = (e) => {
            console.error('Inpaint: FileReader error:', e);
        };
        reader.readAsDataURL(file);
    }

    updateCursor(e) {
        if (!this.image || !this.imageWidth || !this.displayWidth) {
            // No image loaded yet, hide cursor
            this.cursorDiv.style.display = 'none';
            return;
        }

        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const scaledWidth = this.displayWidth * this.scale;
    const imageToDisplayScale = scaledWidth / this.imageWidth;
    let p = (typeof e.pressure === 'number') ? e.pressure : 1;
    // If we're not in active pointer mode or using mouse, show full brush size
    if (!this.pointerActive || e.pointerType === 'mouse') p = 1;
    const actualBrushSize = (this.brushSize * p) / this.scale;
        const displayBrushSize = actualBrushSize * imageToDisplayScale;
        const size = displayBrushSize * 2;

        this.cursorDiv.style.left = (mouseX - displayBrushSize) + "px";
        this.cursorDiv.style.top = (mouseY - displayBrushSize) + "px";
        this.cursorDiv.style.width = size + "px";
        this.cursorDiv.style.height = size + "px";
        this.cursorDiv.style.display = 'block';
    }

    loadImage(imagePath) {
        // Create image directly, not via Q wrapper
        this.image = new Image();
        
        this.image.onload = () => {
            this.imageWidth = this.image.width;
            this.imageHeight = this.image.height;

            this.imageCanvas = document.createElement('canvas');
            this.imageCanvas.width = this.imageWidth;
            this.imageCanvas.height = this.imageHeight;
            this.imageCtx = this.imageCanvas.getContext('2d');
            this.imageCtx.drawImage(this.image, 0, 0);

            this.fitImageToCanvas();
            this.draw();
        };
        
        this.image.onerror = (e) => {
            console.error('Inpaint: Failed to load image from path:', imagePath, e);
        };
        
        this.image.src = imagePath;
    }

    fitImageToCanvas() {
        // Use CSS pixel size (post-transform drawing uses CSS pixels)
        const canvasWidth = Math.floor(this.container.clientWidth);
        const canvasHeight = Math.floor(this.container.clientHeight);

        console.log(`Fitting image to canvas: ${canvasWidth}x${canvasHeight}`);
        console.log(`Image original size: ${this.imageWidth}x${this.imageHeight}`);
        const scaleX = canvasWidth / this.imageWidth;
        const scaleY = canvasHeight / this.imageHeight;
        this.scale = Math.max(scaleX, scaleY);

        this.displayWidth = this.imageWidth;
        this.displayHeight = this.imageHeight;

        this.panX = (canvasWidth - (this.displayWidth * this.scale)) / 2;
        this.panY = (canvasHeight - (this.displayHeight * this.scale)) / 2;
    }



    handleZoom(e) {
        const rect = this.canvas.getBoundingClientRect();
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
        this.updateCursor(e);
    }

    startPanning(e) {
        this.isPanning = true;
        this.lastPanX = e.clientX;
        this.lastPanY = e.clientY;
        this.canvas.style.cursor = 'grabbing';
        this.cursorDiv.style.display = 'none';
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
        this.canvas.style.cursor = 'none';
        if (this.image) {
            this.cursorDiv.style.display = 'block';
        }
    }

    startPainting(e) {
        this.isPainting = true;
        this.lastPaintX = undefined;
        this.lastPaintY = undefined;
        // Cache pressure at stroke start
        let p = (typeof e.pressure === 'number') ? e.pressure : 1;
        if (e.pointerType === 'mouse' && p === 0) p = 1;
        this.lastPaintPressure = p;
        // Snapshot before first stroke of this drag
        this.pushHistory();
        this.paint(e);
    }

    handlePainting(e) {
        if (!this.isPainting) return;
        this.paint(e);
    }

    stopPainting() {
        this.isPainting = false;
        this.lastPaintX = undefined;
        this.lastPaintY = undefined;
        this.lastPaintPressure = 1;
    }

    paint(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const scaledWidth = this.displayWidth * this.scale;
        const scaledHeight = this.displayHeight * this.scale;
        const x = this.displayX + this.panX;
        const y = this.displayY + this.panY;
        const imageX = (mouseX - x) / scaledWidth * this.imageWidth;
    const imageY = (mouseY - y) / scaledHeight * this.imageHeight;
    let p = (typeof e.pressure === 'number') ? e.pressure : 1;
    if (e.pointerType === 'mouse' && p === 0) p = 1;

        if (imageX >= 0 && imageX < this.imageWidth && imageY >= 0 && imageY < this.imageHeight) {
            this.imageCtx.save();

            if (this.lastPaintX !== undefined && this.lastPaintY !== undefined) {
                this.drawSmoothLine(this.lastPaintX, this.lastPaintY, this.lastPaintPressure, imageX, imageY, p);
            } else {
                this.drawBrushStroke(imageX, imageY, p);
            }

            this.lastPaintX = imageX;
            this.lastPaintY = imageY;
            this.lastPaintPressure = p;

            this.imageCtx.restore();
            this.draw();
        }
    }

    drawSmoothLine(x1, y1, p1, x2, y2, p2) {
        const distance = Math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
        const steps = Math.max(1, Math.floor(distance / 2));

        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const x = x1 + (x2 - x1) * t;
            const y = y1 + (y2 - y1) * t;
            const p = p1 + (p2 - p1) * t;
            this.drawBrushStroke(x, y, p);
        }
    }

    drawBrushStroke(x, y, pressure = 1) {
        const p = Math.max(0, Math.min(1, pressure));
        const radius = (this.brushSize * p) / this.scale;

        if (this.isErasing) {
            this.imageCtx.globalCompositeOperation = 'destination-out';
            this.imageCtx.fillStyle = 'rgba(0, 0, 0, 1)';
            this.imageCtx.beginPath();
            this.imageCtx.arc(x, y, radius, 0, 2 * Math.PI);
            this.imageCtx.fill();

            this.imageCtx.globalCompositeOperation = 'destination-over';
            this.imageCtx.drawImage(this.image, 0, 0);
        } else {
            this.imageCtx.globalCompositeOperation = 'source-over';
            this.imageCtx.fillStyle = this.brushColor;
            this.imageCtx.beginPath();
            this.imageCtx.arc(x, y, radius, 0, 2 * Math.PI);
            this.imageCtx.fill();
        }
    }

    setBrushSize(size) {
        this.brushSize = size;
        if (this.sizeSlider) {
            this.sizeSlider.set(size);
        }
    }

    clearMask() {
        if (this.imageCtx) {
            this.pushHistory();
            this.imageCtx.clearRect(0, 0, this.imageWidth, this.imageHeight);
            this.imageCtx.drawImage(this.image, 0, 0);
            this.draw();
        }
    }

    set(value) {
        if (typeof value === 'string' && value.startsWith('data:')) {
            // Base64 data URL
            this.loadImageFromBase64(value);
        } else if (typeof value === 'string' && value.startsWith('blob:')) {
            // Blob URL (binary WebSocket data)
            console.log('Inpaint: Setting blob URL image:', value);
            this.loadImage(value);
        } else if (typeof value === 'string') {
            // Regular URL/path
            this.loadImage(value);
        } else if (value instanceof Blob) {
            // Blob object (binary data)
            console.log('Inpaint: Setting image from Blob, size:', value.size);
            const blobUrl = URL.createObjectURL(value);
            this.loadImage(blobUrl);
            
            // Clean up blob URL after image loads
            setTimeout(() => {
                URL.revokeObjectURL(blobUrl);
            }, 30000);
        } else if (value instanceof ArrayBuffer) {
            // ArrayBuffer (binary data)
            console.log('Inpaint: Setting image from ArrayBuffer, size:', value.byteLength);
            const blob = new Blob([value], { type: 'image/jpeg' });
            const blobUrl = URL.createObjectURL(blob);
            this.loadImage(blobUrl);
            
            // Clean up blob URL after image loads
            setTimeout(() => {
                URL.revokeObjectURL(blobUrl);
            }, 30000);
        } else {
            console.warn('Inpaint: Unknown data type for image:', typeof value, value);
        }
    }

    get() {
        if (!this.imageCanvas) return null;
        return this.imageCanvas.toDataURL('image/png');
    }

    getMask() {
        if (!this.imageCanvas) return null;

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.imageWidth;
        tempCanvas.height = this.imageHeight;
        const tempCtx = tempCanvas.getContext('2d');

        tempCtx.drawImage(this.imageCanvas, 0, 0);

        return tempCanvas.toDataURL('image/png');
    }

    loadImageFromBase64(base64) {
        console.log('Inpaint: loadImageFromBase64 called, data length:', base64 ? base64.length : 0);
        
        // Check if container is visible and has dimensions
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        if (width === 0 || height === 0) {
            console.log('Inpaint: Container not visible yet, storing image data for later');
            this.pendingImageData = base64;
            return;
        }
        
        // Create image directly, not via Q wrapper - critical for onload to fire
        this.image = new Image();
        
        this.image.onload = () => {
            console.log('Inpaint: Image loaded successfully, size:', this.image.width, 'x', this.image.height);
            
            this.imageWidth = this.image.width;
            this.imageHeight = this.image.height;

            this.imageCanvas = document.createElement('canvas');
            this.imageCanvas.width = this.imageWidth;
            this.imageCanvas.height = this.imageHeight;
            this.imageCtx = this.imageCanvas.getContext('2d');
            this.imageCtx.drawImage(this.image, 0, 0);

            // Ensure canvas is properly sized before fitting image
            this.resizeCanvasToContainer();
            this.fitImageToCanvas();
            this.draw();
            // Reset history and push baseline state
            this.undoStack = [];
            this.redoStack = [];
            this.pushHistory();
            
            console.log('Inpaint: Image rendering complete');
        };
        
        this.image.onerror = (e) => {
            console.error('Inpaint: Image load error:', e);
        };
        
        this.image.src = base64;
    }

    // ===== History helpers =====
    pushHistory() {
        if (!this.imageCtx || !this.imageWidth || !this.imageHeight) return;
        try {
            const snapshot = this.imageCtx.getImageData(0, 0, this.imageWidth, this.imageHeight);
            this.undoStack.push(snapshot);
            if (this.undoStack.length > this.historyLimit) this.undoStack.shift();
            // Any new action invalidates redo history
            this.redoStack.length = 0;
        } catch (err) {
            console.warn('Inpaint: pushHistory failed', err);
        }
    }

    undo() {
        if (!this.imageCtx || this.undoStack.length === 0) return;
        try {
            // Move current state to redo, restore last from undo
            const current = this.imageCtx.getImageData(0, 0, this.imageWidth, this.imageHeight);
            this.redoStack.push(current);
            const prev = this.undoStack.pop();
            if (prev) {
                this.imageCtx.putImageData(prev, 0, 0);
                this.draw();
            }
        } catch (err) {
            console.warn('Inpaint: undo failed', err);
        }
    }

    redo() {
        if (!this.imageCtx || this.redoStack.length === 0) return;
        try {
            // Move current to undo, restore last from redo
            const current = this.imageCtx.getImageData(0, 0, this.imageWidth, this.imageHeight);
            this.undoStack.push(current);
            if (this.undoStack.length > this.historyLimit) this.undoStack.shift();
            const next = this.redoStack.pop();
            if (next) {
                this.imageCtx.putImageData(next, 0, 0);
                this.draw();
            }
        } catch (err) {
            console.warn('Inpaint: redo failed', err);
        }
    }

    draw() {
        // Clear using CSS pixels (context already scaled by DPR)
        const cssWidth = Math.floor(this.container.clientWidth);
        const cssHeight = Math.floor(this.container.clientHeight);
        this.ctx.clearRect(0, 0, cssWidth, cssHeight);

        if (!this.imageCanvas) return;

        this.ctx.save();

        const scaledWidth = this.displayWidth * this.scale;
        const scaledHeight = this.displayHeight * this.scale;
        const x = this.displayX + this.panX;
        const y = this.displayY + this.panY;

        this.ctx.drawImage(this.imageCanvas, x, y, scaledWidth, scaledHeight);
        this.ctx.restore();
    }

    destroy() {
        // Cleanup observers when component is destroyed
        if (this.visibilityObserver) {
            this.visibilityObserver.disconnect();
            this.visibilityObserver = null;
        }
        
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }
        if (this.sourceButtonsInterval) {
            clearInterval(this.sourceButtonsInterval);
            this.sourceButtonsInterval = null;
        }
    }
    // ===== Source buttons helpers =====
    componentExists(assign, framePreference = null) {
        try {
            const cb = window.COMPONENTS_BUILDER;
            if (!cb || typeof cb.getComponentsByAssign !== 'function') return false;
            const assigns = this.resolveAssignCandidates(assign);
            const frameTargets = this.resolveFrameTargets(framePreference, cb);
            for (const candidate of assigns) {
                for (const frameId of frameTargets) {
                    const list = cb.getComponentsByAssign(candidate, frameId);
                    if (list && list.length > 0) {
                        return true;
                    }
                }
            }
        } catch (_e) {}
        return false;
    }

    getPreviewComponent(assign, framePreference = null) {
        try {
            const cb = window.COMPONENTS_BUILDER;
            if (!cb || typeof cb.getComponentsByAssign !== 'function') return null;
            const assigns = this.resolveAssignCandidates(assign);
            const frameTargets = this.resolveFrameTargets(framePreference, cb);
            for (const candidate of assigns) {
                for (const frameId of frameTargets) {
                    const list = cb.getComponentsByAssign(candidate, frameId) || [];
                    if (list.length > 0) {
                        return list[0];
                    }
                }
            }
        } catch (_e) {}
        return null;
    }

    getPreviewData(assign, framePreference = null) {
        try {
            if (window.ModuleAPI && typeof window.ModuleAPI.get === 'function') {
                const v = window.ModuleAPI.get(assign);
                if (v && typeof v === 'string' && v.startsWith('data:')) return v;
            }
            const comp = this.getPreviewComponent(assign, framePreference);
            if (comp && typeof comp.get === 'function') {
                const value = comp.get();
                if (value && typeof value === 'string' && value.startsWith('data:')) {
                    return value;
                }
            }
        } catch (_e) {}
        return null;
    }

    resolveAssignCandidates(assign) {
        const MAP = {
            'img-preview': ['img-preview'],
            'i2i-img-preview': ['i2i-img-preview']
        };
        const list = MAP[assign];
        if (!list) return [assign];
        return Array.from(new Set(list.concat(assign)));
    }

    resolveFrameTargets(preference, cb) {
        const frames = [];
        if (Array.isArray(preference)) {
            preference.forEach(f => {
                if (f !== undefined && f !== null) frames.push(f);
            });
        } else if (preference) {
            frames.push(preference);
        }
        const active = cb && typeof cb.getCurrentFrameTabId === 'function' ? cb.getCurrentFrameTabId() : null;
        if (!preference && active) {
            frames.push(active);
        }
        // Always offer a catch-all lookup across all frame tabs
        frames.push(null);
        return Array.from(new Set(frames));
    }

    updateSourceButtonsVisibility() {
        const t2iData = this.getPreviewData('img-preview', ['tab-text-to-image', null]);
        const i2iData = this.getPreviewData('i2i-img-preview', ['tab-image-to-image', null]);
        const t2iExists = !!t2iData && this.componentExists('img-preview', ['tab-text-to-image', null]);
        const i2iExists = !!i2iData && this.componentExists('i2i-img-preview', ['tab-image-to-image', null]);
        this.t2iButton.style.display = t2iExists ? 'flex' : 'none';
        this.i2iButton.style.display = i2iExists ? 'flex' : 'none';
    }

    getElement() {
        return this.inputWrapper;
    }
}

window.Inpaint = Inpaint;
