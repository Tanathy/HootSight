class Flexbox {
    constructor(container, config) {
        this.container = container;
        this.config = config;
        this.element = null;
        this.contentElement = null;

        this.init();
    }

    init() {
        this.createElement();
        this.attachToContainer();
    }

    createElement() {
        this.element = Q('<div>', { class: 'flexbox-component' }).get(0);
        this.element.id = this.config.id;

        // Set flexbox properties
        if (this.config.direction) {
            Q(this.element).css('flexDirection', this.config.direction);
        }
        if (this.config.gap) {
            Q(this.element).css('gap', this.config.gap);
        }
        if (this.config.align) {
            Q(this.element).css('alignItems', this.config.align);
        }
        if (this.config.justify) {
            Q(this.element).css('justifyContent', this.config.justify);
        }

        // The flexbox element itself is the content element
        this.contentElement = this.element;

        if (this.config.colors) {
            if (this.config.colors.startsWith('--')) {
                Q(this.element).css('backgroundColor', `var(${this.config.colors})`);
            } else {
                Q(this.element).css('backgroundColor', this.config.colors);
            }
        }
    }

    attachToContainer() {
        if (this.container && this.element) {
            Q(this.container).append(this.element);
        }
    }

    getElement() {
        return this.element;
    }

    getContentElement() {
        return this.contentElement;
    }

    setContent(content) {
        const contentElement = this.getContentElement();
        if (contentElement) {
            if (typeof content === 'string') {
                Q(contentElement).html(content);
            } else if (content instanceof Element) {
                Q(contentElement).empty();
                Q(contentElement).append(content);
            }
        }
    }

    destroy() {
        if (this.element) {
            Q(this.element).remove();
        }
        this.element = null;
    }
}

function createFlexbox(container, config) {
    return new Flexbox(container, config);
}
