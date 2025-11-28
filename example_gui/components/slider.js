class Slider {
    constructor(identifier, min, max, step, value = 0, title = "", description = "") {
        this.identifier = identifier;
        this.min = min;
        this.max = max;
        this.step = step;
        this.isInteger = Number.isInteger(min) && Number.isInteger(max) && Number.isInteger(step);
        this.sliderWrapper = Q('<div>', { class: 'slider_wrapper' }).get(0);
        if (title) {
            const heading = Q('<h3>', { class: 'inputs_title', text: title }).get(0);
            this.sliderWrapper.appendChild(heading);
        }
        if (description) {
            const descriptionHeading = Q('<h4>', { class: 'inputs_description', text: description }).get(0);
            this.sliderWrapper.appendChild(descriptionHeading);
        }
        const sliderContent = Q('<div>', { class: 'slider_content' }).get(0);
        this.track = Q('<div>', { class: 'custom_slider_wrapper' }).get(0);
        this.fill = Q('<div>', { class: 'custom_slider_inner' }).get(0);
        this.inputField = Q('<input>', { class: 'slider_input', type: 'number', step: this.isInteger ? '1' : step, value: value, id: identifier }).get(0);
        this.track.appendChild(this.fill);
        sliderContent.append(this.track, this.inputField);
        this.sliderWrapper.appendChild(sliderContent);
        this.updateSlider(value);
        this.setupEventListeners();
    }
    
    updateSlider(val) {
        val = Math.max(this.min, Math.min(this.max, val));
        const steppedValue = Math.round(val / this.step) * this.step;
        const clampedValue = Math.max(this.min, Math.min(this.max, steppedValue));
        const percent = ((clampedValue - this.min) / (this.max - this.min)) * 100;
        this.fill.style.width = percent + "%";
        this.inputField.value = this.formatNumber(clampedValue);
    }
    
    formatNumber(num) {
        if (this.isInteger) {
            return Math.round(num).toString();
        }
        if (num % 1 === 0) return num.toString();
        const precision = this.step.toString().split('.')[1]?.length || 0;
        if (precision > 0) {
            return parseFloat(num.toFixed(precision)).toString();
        }
        let str = parseFloat(num.toPrecision(12)).toString();
        if (str.includes('.')) {
            str = str.replace(/\.?0+$/, '');
        }
        return str;
    }
    
    getValueFromPosition(x) {
        const rect = this.track.getBoundingClientRect();
        const percent = Math.min(Math.max((x - rect.left) / rect.width, 0), 1);
        const rawValue = this.min + percent * (this.max - this.min);
        const steppedValue = Math.round(rawValue / this.step) * this.step;
        return Math.min(Math.max(steppedValue, this.min), this.max);
    }
    
    setupEventListeners() {
        let dragging = false;
        const onMouseMove = (e) => {
            if (dragging) this.updateSlider(this.getValueFromPosition(e.clientX));
        };
        const onMouseUp = () => {
            dragging = false;
            Q(document).off("mousemove", onMouseMove);
            Q(document).off("mouseup", onMouseUp);
            Q(this.inputField).trigger('change');
        };
        Q(this.track).on("mousedown", (e) => {
            dragging = true;
            this.updateSlider(this.getValueFromPosition(e.clientX));
            Q(document).on("mousemove", onMouseMove);
            Q(document).on("mouseup", onMouseUp);
            Q(this.inputField).trigger('change');
        });
        Q(this.inputField).on("input", () => {
            Q(this.inputField).trigger('change');
        });
        
        Q(this.inputField).on("blur", () => {
            let inputValue = parseFloat(this.inputField.value);
            if (isNaN(inputValue)) {
                inputValue = this.min;
            } else {
                inputValue = Math.max(this.min, Math.min(this.max, inputValue));
                if (this.isInteger) {
                    inputValue = Math.round(inputValue);
                }
            }
            this.updateSlider(inputValue);
            this.syncVisualToValue();
        });
    }
    
    get() {
        const value = parseFloat(this.inputField.value);
        return isNaN(value) ? this.min : value;
    }
    
    set(value) {
        this.updateSlider(value);
        Q(this.inputField).trigger('change');
    }
    
    syncVisualToValue() {
        const currentValue = this.get();
        const percent = ((currentValue - this.min) / (this.max - this.min)) * 100;
        this.fill.style.width = percent + "%";
    }
    
    getElement() {
        return this.sliderWrapper;
    }
}
