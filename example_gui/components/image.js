class ImageComponent {
    constructor(identifier, options = {}) {
        this.identifier = identifier;
        this.options = {
            content: null, // url | dataURL | Blob
            width: undefined,
            height: undefined,
            fill: undefined, // 'contain' | 'cover'
            thumb: false,
            ...options
        };

        this._objectUrl = null; // track current object URL to revoke

        this.wrapper = Q('<div>', { class: 'image_wrapper' }).get(0);
        this.contentEl = Q('<div>', { class: 'image_content' }).get(0);
        this.img = Q('<img>', { class: 'image_el', id: identifier }).get(0);
        this.contentEl.appendChild(this.img);
        this.wrapper.appendChild(this.contentEl);

        this._applySizing();
        if (this.options.content) {
            this._renderContent(this.options.content).catch(err => console.error('Image render error:', err));
        }
    }

    getElement() { return this.wrapper; }

    // This component is set-only for payload purposes
    get() { return null; }

    destroy() {
        if (this._objectUrl) {
            URL.revokeObjectURL(this._objectUrl);
            this._objectUrl = null;
        }
    }

    updateOptions(opts = {}) {
        const { width, height, fill, thumb } = opts;
        if (width !== undefined) this.options.width = width;
        if (height !== undefined) this.options.height = height;
        if (fill !== undefined) this.options.fill = fill;
        if (thumb !== undefined) this.options.thumb = !!thumb;
        this._applySizing();
    }

    set(valueOrOptions) {
        if (valueOrOptions && typeof valueOrOptions === 'object' && !(valueOrOptions instanceof Blob)) {
            // Treat as options update; may include content
            const { content, width, height, fill, thumb } = valueOrOptions;
            this.updateOptions({ width, height, fill, thumb });
            if (content !== undefined) {
                this.options.content = content;
                return this._renderContent(content);
            }
            return Promise.resolve();
        } else {
            // Treat as content update only (string | Blob)
            this.options.content = valueOrOptions;
            return this._renderContent(valueOrOptions);
        }
    }

    _applySizing() {
        const { width, height, fill } = this.options;
        const imgStyle = this.img.style;

        // Reset first
        imgStyle.width = '';
        imgStyle.height = '';
        imgStyle.objectFit = '';
        imgStyle.objectPosition = '';

        const hasW = Number.isFinite(width);
        const hasH = Number.isFinite(height);

        if (hasW && !hasH) {
            imgStyle.width = `${width}px`;
            imgStyle.height = 'auto';
        } else if (!hasW && hasH) {
            imgStyle.height = `${height}px`;
            imgStyle.width = 'auto';
        } else if (hasW && hasH) {
            imgStyle.width = `${width}px`;
            imgStyle.height = `${height}px`;
            if (fill === 'contain' || fill === 'cover') {
                imgStyle.objectFit = fill;
                imgStyle.objectPosition = 'center center';
            } else {
                // No fill provided: stretch
                imgStyle.objectFit = 'fill';
            }
        } else {
            // No explicit sizing: keep natural size within container
            imgStyle.maxWidth = '100%';
            imgStyle.height = 'auto';
        }
    }

    async _renderContent(content) {
        // Clean previous URL
        if (this._objectUrl) {
            URL.revokeObjectURL(this._objectUrl);
            this._objectUrl = null;
        }

        const { width, height, fill, thumb } = this.options;

        // Helper: load an image from a src and await natural sizes
        const loadImage = (src) => new Promise((resolve, reject) => {
            this.img.onload = () => resolve({ width: this.img.naturalWidth, height: this.img.naturalHeight });
            this.img.onerror = (e) => reject(e);
            this.img.src = src;
        });

        // Determine source type
        let sourceType = 'url';
        let srcUrl = null;
        let blob = null;
        if (content instanceof Blob) {
            sourceType = 'blob';
            blob = content;
        } else if (typeof content === 'string') {
            if (content.startsWith('data:')) {
                sourceType = 'dataurl';
            } else if (/^(https?:)?\//i.test(content)) {
                sourceType = 'url';
            } else {
                // Assume path-like URL
                sourceType = 'url';
            }
        } else if (content == null) {
            // nothing to render
            this.img.removeAttribute('src');
            return;
        }

        // Thumbnail path: render through canvas and use Blob URL (never base64 for thumb)
        const hasW = Number.isFinite(width);
        const hasH = Number.isFinite(height);
        if (thumb && (hasW || hasH)) {
            try {
                const { blobUrl } = await this._renderThumbnail(content, { width, height, fill });
                this._objectUrl = blobUrl;
                await loadImage(blobUrl);
                return;
            } catch (e) {
                console.warn('Thumbnail render failed, falling back to original source:', e);
            }
        }

        // Non-thumb path: set appropriate src
        if (sourceType === 'blob') {
            srcUrl = URL.createObjectURL(blob);
            this._objectUrl = srcUrl;
            await loadImage(srcUrl);
        } else if (sourceType === 'dataurl') {
            await loadImage(content);
        } else if (sourceType === 'url') {
            await loadImage(content);
        }
    }

    async _renderThumbnail(content, { width, height, fill }) {
        const imgEl = new window.Image();
        imgEl.decoding = 'async';
        imgEl.crossOrigin = 'anonymous'; // allow CORS-friendly sources

        // Resolve input
        let inputUrl = null;
        let revokeInputUrl = null;

        if (content instanceof Blob) {
            inputUrl = URL.createObjectURL(content);
            revokeInputUrl = inputUrl;
        } else if (typeof content === 'string') {
            if (content.startsWith('data:')) {
                inputUrl = content; // OK as input
            } else {
                inputUrl = content; // URL/path
            }
        } else {
            throw new Error('Unsupported content for thumbnail');
        }

        const imgLoaded = new Promise((resolve, reject) => {
            imgEl.onload = () => resolve();
            imgEl.onerror = (e) => reject(e);
        });
        imgEl.src = inputUrl;
        await imgLoaded;

        const srcW = imgEl.naturalWidth || imgEl.width;
        const srcH = imgEl.naturalHeight || imgEl.height;

        const hasW = Number.isFinite(width);
        const hasH = Number.isFinite(height);

        // Determine target dimensions
        let targetW = hasW ? width : undefined;
        let targetH = hasH ? height : undefined;

        if (hasW && !hasH) {
            targetH = Math.round((width / srcW) * srcH);
        } else if (!hasW && hasH) {
            targetW = Math.round((height / srcH) * srcW);
        } else if (!hasW && !hasH) {
            // No constraints: just use source size
            targetW = srcW; targetH = srcH;
        } // both W & H present -> keep as provided

        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, targetW);
        canvas.height = Math.max(1, targetH);
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        if (Number.isFinite(width) && Number.isFinite(height)) {
            if (fill === 'contain') {
                ctx.clearRect(0, 0, targetW, targetH);
                const scale = Math.min(targetW / srcW, targetH / srcH);
                const drawW = Math.round(srcW * scale);
                const drawH = Math.round(srcH * scale);
                const dx = Math.floor((targetW - drawW) / 2);
                const dy = Math.floor((targetH - drawH) / 2);
                ctx.drawImage(imgEl, 0, 0, srcW, srcH, dx, dy, drawW, drawH);
            } else if (fill === 'cover') {
                const scale = Math.max(targetW / srcW, targetH / srcH);
                const cropW = Math.round(targetW / scale);
                const cropH = Math.round(targetH / scale);
                const sx = Math.max(0, Math.floor((srcW - cropW) / 2));
                const sy = Math.max(0, Math.floor((srcH - cropH) / 2));
                ctx.drawImage(imgEl, sx, sy, cropW, cropH, 0, 0, targetW, targetH);
            } else {
                // stretch (no aspect)
                ctx.drawImage(imgEl, 0, 0, srcW, srcH, 0, 0, targetW, targetH);
            }
        } else {
            // Only one side constrained or none -> keep aspect by computed targetW/H
            ctx.drawImage(imgEl, 0, 0, srcW, srcH, 0, 0, targetW, targetH);
        }

        const blob = await new Promise((resolve, reject) => {
            try {
                canvas.toBlob((b) => {
                    if (b) resolve(b); else reject(new Error('toBlob returned null'));
                }, 'image/png', 0.92);
            } catch (e) { reject(e); }
        });

        if (revokeInputUrl) URL.revokeObjectURL(revokeInputUrl);
        const blobUrl = URL.createObjectURL(blob);
        return { blob, blobUrl };
    }
}
