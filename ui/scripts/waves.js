/**
 * HootSight - Waves Background Animation
 * Optimized canvas-based animated wave effect
 */
(function() {
    'use strict';

    const PI2 = Math.PI * 2;

    function Waves(selector, options) {
        const self = this;

        self.options = Object.assign({
            resize: true,
            rotation: 0,
            waves: 5,
            width: 800,
            hue: [11, 30],
            amplitude: 0.5,
            background: false,
            preload: true,
            speed: [0.0004, 0.0001],
            fps: 30
        }, options || {});

        self.holder = Q(selector).get();
        if (!self.holder) return;

        self.canvas = Q('<canvas>').get();
        self.ctx = self.canvas.getContext('2d');
        Q(self.holder).append(self.canvas);

        self.waves = [];
        self.hue = self.options.hue[0];
        self.hueFw = true;
        self.color = '';
        self.animationId = null;
        self._lastFrame = 0;
        self._frameInterval = 1000 / self.options.fps;

        self._rotation = self.options.rotation * Math.PI / 180;
        self._amplitude = self.options.amplitude;

        self._resize();
        self._init();

        if (self.options.resize) {
            let resizeTimeout;
            window.addEventListener('resize', function() {
                clearTimeout(resizeTimeout);
                resizeTimeout = setTimeout(function() {
                    self._resize();
                }, 1000);
            }, false);
        }
    }

    Waves.prototype._init = function() {
        const self = this;
        for (let i = 0; i < self.options.waves; i++) {
            self.waves[i] = new Wave(self);
        }
        if (self.options.preload) {
            self._preload();
        }
    };

    Waves.prototype._preload = function() {
        const self = this;
        for (let i = 0; i < self.options.waves; i++) {
            self._updateColor();
            for (let j = 0; j < self.options.width; j++) {
                self.waves[i].update(self.color);
            }
        }
    };

    Waves.prototype._resize = function() {
        const self = this;
        const width = self.holder.offsetWidth;
        const height = self.holder.offsetHeight;
        const scale = window.devicePixelRatio || 1;

        self.width = width * scale;
        self.height = height * scale;
        self.canvas.width = self.width;
        self.canvas.height = self.height;
        self.canvas.style.width = width + 'px';
        self.canvas.style.height = height + 'px';

        self.centerX = self.width / 2;
        self.centerY = self.height / 2;
        self.radius = Math.sqrt(self.width * self.width + self.height * self.height) / 2;
        self.radius3 = self.radius / 3;
    };

    Waves.prototype._updateColor = function() {
        const self = this;
        self.hue += self.hueFw ? 0.01 : -0.01;

        if (self.hue > self.options.hue[1]) {
            self.hue = self.options.hue[1];
            self.hueFw = false;
        } else if (self.hue < self.options.hue[0]) {
            self.hue = self.options.hue[0];
            self.hueFw = true;
        }

        const h = self.hue * 0.3;
        const a = (127 * Math.sin(h) + 128) | 0;
        const b = (127 * Math.sin(h + 2) + 128) | 0;
        const c = (127 * Math.sin(h + 4) + 128) | 0;

        self.color = 'rgba(' + a + ',' + b + ',' + c + ',0.1)';
    };

    Waves.prototype._clear = function() {
        this.ctx.clearRect(0, 0, this.width, this.height);
    };

    Waves.prototype._background = function() {
        const self = this;
        const ctx = self.ctx;
        const gradient = ctx.createLinearGradient(0, 0, 0, self.height);
        gradient.addColorStop(0, '#000');
        gradient.addColorStop(1, self.color);
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, self.width, self.height);
    };

    Waves.prototype._render = function() {
        const self = this;
        self._updateColor();
        self._clear();

        if (self.options.background) {
            self._background();
        }

        for (let i = 0; i < self.waves.length; i++) {
            self.waves[i].update(self.color);
            self.waves[i].draw();
        }
    };

    Waves.prototype.animate = function(timestamp) {
        const self = this;
        
        self.animationId = requestAnimationFrame(function(ts) {
            self.animate(ts);
        });
        
        // FPS limiter
        if (!timestamp) timestamp = 0;
        const elapsed = timestamp - self._lastFrame;
        if (elapsed < self._frameInterval) return;
        self._lastFrame = timestamp - (elapsed % self._frameInterval);
        
        self._render();
    };

    Waves.prototype.stop = function() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    };

    function Wave(parent) {
        const self = this;
        const speed = parent.options.speed;

        self.parent = parent;
        self.lines = [];

        self.angle = new Float32Array([
            Math.random() * PI2,
            Math.random() * PI2,
            Math.random() * PI2,
            Math.random() * PI2
        ]);

        const sign = function() { return Math.random() > 0.5 ? 1 : -1; };
        const rndSpeed = function() { 
            return (speed[0] + Math.random() * (speed[1] - speed[0])) * sign(); 
        };

        self.speed = new Float32Array([
            rndSpeed(), rndSpeed(), rndSpeed(), rndSpeed()
        ]);
    }

    Wave.prototype.update = function(color) {
        const self = this;
        const angle = self.angle;
        const speed = self.speed;

        self.lines.push({
            angle: new Float32Array([
                Math.sin(angle[0] += speed[0]),
                Math.sin(angle[1] += speed[1]),
                Math.sin(angle[2] += speed[2]),
                Math.sin(angle[3] += speed[3])
            ]),
            color: color
        });

        if (self.lines.length > self.parent.options.width) {
            self.lines.shift();
        }
    };

    Wave.prototype.draw = function() {
        const self = this;
        const parent = self.parent;
        const ctx = parent.ctx;
        const lines = self.lines;
        const len = lines.length;

        const x = parent.centerX;
        const y = parent.centerY;
        const radius = parent.radius;
        const radius3 = parent.radius3;
        const rotation = parent._rotation;
        const amplitude = parent._amplitude;
        const amp2 = amplitude * 2;

        for (let i = 0; i < len; i++) {
            const line = lines[i];
            const a = line.angle;
            
            // Alpha fade: older lines (low index) = transparent, newer = visible
            const alpha = (i / len) * 0.15;

            const a0r = a[0] * amplitude + rotation;
            const a3r = a[3] * amplitude + rotation;

            ctx.strokeStyle = line.color.replace(/[\d.]+\)$/, alpha.toFixed(3) + ')');
            ctx.beginPath();
            ctx.moveTo(
                x - radius * Math.cos(a0r),
                y - radius * Math.sin(a0r)
            );
            ctx.bezierCurveTo(
                x - radius3 * Math.cos(a[1] * amp2),
                y - radius3 * Math.sin(a[1] * amp2),
                x + radius3 * Math.cos(a[2] * amp2),
                y + radius3 * Math.sin(a[2] * amp2),
                x + radius * Math.cos(a3r),
                y + radius * Math.sin(a3r)
            );
            ctx.stroke();
        }
    };

    window.Waves = Waves;
})();