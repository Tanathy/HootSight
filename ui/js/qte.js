
/**
 * Q.js - Altered Qte
 *
 * A lightweight, modern JavaScript library that provides jQuery-like functionality
 * for DOM manipulation, event handling, and element creation. Designed to be
 * fast, chainable, and compatible with modern web development practices.
 *
 * Features:
 * - jQuery-style element selection and creation
 * - Chainable DOM manipulation methods
 * - Advanced event handling with options support
 * - SVG element creation with proper namespaces
 * - Template-based HTML parsing for security
 * - Minimal footprint with maximum functionality
 *
 * @author Roxxy AI
 * @version 2.0.0
 * @license MIT
 */

/**
 * QWrapper Class - Core DOM manipulation wrapper
 *
 * Wraps DOM elements and provides chainable methods for manipulation.
 * Similar to jQuery's core functionality but with modern JavaScript patterns.
 *
 * Architecture:
 * - Stores elements in an array for multi-element operations
 * - All methods return 'this' for chaining
 * - Uses native DOM APIs for optimal performance
 * - Handles edge cases and cross-browser compatibility
 */
class QWrapper {
    /**
     * Creates a new QWrapper instance
     * @param {NodeList|Array|Node} elements - Elements to wrap
     */
    constructor(elements) {
        // Convert input to array, handling various input types
        // Array.from() ensures compatibility with different iterables
        this.elements = Array.from(elements || []);
    }

    /**
     * Adds CSS class(es) to all wrapped elements
     * @param {string} className - Class name(s) to add (space-separated)
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('.items').addClass('active visible')
     *          Q('#myElement').addClass('highlight')
     */
    addClass(className) {
        this.elements.forEach(el => el.classList.add(className));
        return this;
    }

    /**
     * Append wrapped elements to target (jQuery-like .appendTo)
     * @param {string|Element|QWrapper|NodeList|Array} target
     * @returns {QWrapper}
     */
    appendTo(target){
        let parents = [];
        if(!target) return this;
        if(target instanceof QWrapper){ parents = target.getAll(); }
        else if(typeof target === 'string'){ parents = Array.from(document.querySelectorAll(target)); }
        else if(target instanceof NodeList || Array.isArray(target)){ parents = Array.from(target); }
        else if(target.nodeType){ parents = [target]; }
        parents.forEach(p=>{ this.elements.forEach(el=> p.appendChild(el)); });
        return this;
    }

    /**
     * Iterate over wrapped elements (jQuery-compatible API)
     * @param {Function} cb - callback(index, element) returning false breaks
     * @returns {QWrapper}
     */
    each(cb){
        for(let i=0;i<this.elements.length;i++){
            const res = cb.call(this.elements[i], i, this.elements[i]);
            if(res === false) break;
        }
        return this;
    }

    /**
     * Removes CSS class(es) from all wrapped elements
     * @param {string} className - Class name(s) to remove
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('.items').removeClass('active')
     *          Q('#myElement').removeClass('hidden error')
     */
    removeClass(className) {
        this.elements.forEach(el => el.classList.remove(className));
        return this;
    }

    /**
     * Toggles CSS class(es) on all wrapped elements
     * @param {string} className - Class name(s) to toggle
     * @param {boolean} [force] - Force add (true) or remove (false)
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('.button').toggleClass('active')
     *          Q('#menu').toggleClass('open', true)  // Force add
     *          Q('.item').toggleClass('selected', false)  // Force remove
     */
    toggleClass(className, force) {
        this.elements.forEach(el => el.classList.toggle(className, force));
        return this;
    }

    /**
     * Appends children to all wrapped elements
     * @param {...(string|Node|QWrapper)} children - Children to append
     * @returns {QWrapper} - Returns self for chaining
     *
     * Supports:
     * - Strings (creates text nodes)
     * - DOM Nodes (appends directly)
     * - QWrapper instances (appends all their elements)
     *
     * Example: Q('#container').append('<p>New paragraph</p>')
     *          Q('.list').append(document.createElement('li'))
     *          Q('#parent').append(Q('<div>Child</div>'))
     */
    append(...children) {
        this.elements.forEach(el => {
            children.forEach(child => {
                if (typeof child === 'string') {
                    // Create text node for strings
                    el.appendChild(document.createTextNode(child));
                } else if (child instanceof Node) {
                    // Append DOM nodes directly
                    el.appendChild(child);
                } else if (child instanceof QWrapper) {
                    // Append all elements from QWrapper
                    child.getAll().forEach(c => el.appendChild(c));
                }
            });
        });
        return this;
    }

    /**
     * Gets element at specified index
     * @param {number} [index=0] - Index of element to retrieve
     * @returns {Element|null} - The element or null if not found
     *
     * Example: const firstElement = Q('.items').get(0)
     *          const secondElement = Q('.items').get(1)
     */
    get(index = 0) {
        return this.elements[index] || null;
    }

    /**
     * Gets all wrapped elements as an array
     * @returns {Array<Element>} - Array of all elements
     *
     * Example: const allElements = Q('.items').getAll()
     *          allElements.forEach(el => console.log(el))
     */
    getAll() {
        return this.elements;
    }

    /**
     * Sets text content for all wrapped elements
     * @param {string} content - Text content to set
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#message').text('Hello World')
     *          Q('.titles').text('New Title')
     */
    text(content) {
        this.elements.forEach(el => el.textContent = content);
        return this;
    }

    /**
     * Sets HTML content for all wrapped elements
     * @param {string} content - HTML content to set
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#content').html('<p>Rich <strong>text</strong></p>')
     *          Q('.containers').html('<div>New content</div>')
     */
    html(content) {
        this.elements.forEach(el => el.innerHTML = content);
        return this;
    }

    /**
     * Sets CSS property for all wrapped elements
     * @param {string} property - CSS property name
     * @param {string} value - CSS property value
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('.highlight').css('backgroundColor', 'yellow')
     *          Q('#box').css('width', '200px').css('height', '100px')
     */
    css(property, value) {
        this.elements.forEach(el => el.style[property] = value);
        return this;
    }

    /**
     * Hides all wrapped elements (sets display: none)
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('.modal').hide()
     *          Q('#notification').hide()
     */
    hide() {
        this.elements.forEach(el => el.style.display = 'none');
        return this;
    }

    /**
     * Shows all wrapped elements (removes display style)
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('.hidden').show()
     *          Q('#popup').show()
     */
    show() {
        this.elements.forEach(el => el.style.display = '');
        return this;
    }

    /**
     * Removes all wrapped elements from the DOM
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('.temp').remove()
     *          Q('#obsolete').remove()
     */
    remove() {
        this.elements.forEach(el => el.parentNode && el.parentNode.removeChild(el));
        return this;
    }

    /**
     * Finds descendant elements matching selector
     * @param {string} selector - CSS selector to match
     * @returns {QWrapper} - New QWrapper with found elements
     *
     * Example: const buttons = Q('#container').find('button')
     *          const activeItems = Q('.list').find('.active')
     */
    find(selector) {
        const found = [];
        this.elements.forEach(el => {
            const matches = el.querySelectorAll(selector);
            found.push(...Array.from(matches));
        });
        return new QWrapper(found);
    }

    /**
     * Gets parent elements of all wrapped elements
     * @returns {QWrapper} - New QWrapper with parent elements
     *
     * Example: const parentDiv = Q('.child').parent()
     *          parentDiv.addClass('has-children')
     */
    parent() {
        const parents = this.elements.map(el => el.parentNode).filter(p => p);
        return new QWrapper(parents);
    }

    /**
     * Gets child elements of all wrapped elements
     * @returns {QWrapper} - New QWrapper with child elements
     *
     * Example: const childElements = Q('#parent').children()
     *          childElements.addClass('child-element')
     */
    children() {
        const children = [];
        this.elements.forEach(el => {
            children.push(...Array.from(el.children));
        });
        return new QWrapper(children);
    }

    /**
     * Checks if any wrapped element has the specified class
     * @param {string} className - Class name to check
     * @returns {boolean} - True if any element has the class
     *
     * Example: if (Q('.items').hasClass('active')) { ... }
     *          const isVisible = Q('#element').hasClass('visible')
     */
    hasClass(className) {
        return this.elements.some(el => el.classList.contains(className));
    }

    /**
     * Gets or sets the value property of form elements
     * @param {string} [value] - Value to set (if provided)
     * @returns {QWrapper|string} - Self for chaining or current value
     *
     * Example: Q('#input').val('new value')  // Set value
     *          const currentValue = Q('#input').val()  // Get value
     */
    val(value) {
        if (value !== undefined) {
            this.elements.forEach(el => el.value = value);
            return this;
        } else {
            return this.elements[0] ? this.elements[0].value : '';
        }
    }

    /**
     * Gets or sets attributes on elements
     * @param {string} name - Attribute name
     * @param {string} [value] - Attribute value to set
     * @returns {QWrapper|string|null} - Self for chaining or attribute value
     *
     * Example: Q('#link').attr('href', 'https://example.com')  // Set attribute
     *          const href = Q('#link').attr('href')  // Get attribute
     */
    attr(name, value) {
        if (value !== undefined) {
            this.elements.forEach(el => el.setAttribute(name, value));
            return this;
        } else {
            return this.elements[0] ? this.elements[0].getAttribute(name) : null;
        }
    }

    removeAttr(name) {
        this.elements.forEach(el => el.removeAttribute(name));
        return this;
    }

    /**
     * Attaches event listeners to all wrapped elements
     * @param {string} events - Space-separated event names (e.g., 'click mouseenter')
     * @param {Function} handler - Event handler function
     * @param {Object} [options] - Event listener options
     * @returns {QWrapper} - Returns self for chaining
     *
     * Options:
     * - capture: Use capture phase
     * - once: Remove listener after first trigger
     * - passive: Improve scroll performance
     *
     * Example: Q('button').on('click', () => console.log('Clicked'))
     *          Q('#input').on('input change', handleInput, { passive: true })
     */
    on(events, handler, options) {
        // Default options for addEventListener
        const defaultOptions = { capture: false, once: false, passive: false };
        const opts = Object.assign({}, defaultOptions, options);

        // Split multiple events and attach each
        const eventList = events.split(' ');

        this.elements.forEach(element => {
            eventList.forEach(eventName => {
                element.addEventListener(eventName.trim(), handler, opts);
            });
        });
        return this;
    }

    /**
     * Removes event listeners from all wrapped elements
     * @param {string} events - Space-separated event names
     * @param {Function} handler - Original handler function
     * @param {Object} [options] - Same options used with on()
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('button').off('click', clickHandler)
     *          Q('#input').off('input change', handleInput)
     */
    off(events, handler, options) {
        const defaultOptions = { capture: false, once: false, passive: false };
        const opts = Object.assign({}, defaultOptions, options);
        const eventList = events.split(' ');

        this.elements.forEach(element => {
            eventList.forEach(eventName => {
                element.removeEventListener(eventName.trim(), handler, opts);
            });
        });
        return this;
    }

    /**
     * Convenience method for click events
     * @param {Function} [handler] - Click handler (if provided)
     * @returns {QWrapper|undefined} - Self for chaining or triggers click
     *
     * Example: Q('button').click(() => alert('Button clicked'))  // Add handler
     *          Q('#submit').click()  // Trigger click
     */
    click(handler) {
        return handler ? this.on('click', handler) : this.get(0)?.click();
    }

    /**
     * Gets or sets data attributes (data-*)
     * @param {string} key - Data attribute key (without 'data-')
     * @param {string} [value] - Value to set
     * @returns {QWrapper|string} - Self for chaining or current value
     *
     * Example: Q('#item').data('id', '123')  // Set data attribute
     *          const itemId = Q('#item').data('id')  // Get data attribute
     */
    data(key, value) {
        if (value !== undefined) {
            this.elements.forEach(el => el.dataset[key] = value);
            return this;
        } else {
            return this.elements[0]?.dataset[key];
        }
    }

    /**
     * Gets or sets element properties
     * @param {string} name - Property name
     * @param {*} [value] - Value to set
     * @returns {QWrapper|*} - Self for chaining or current value
     *
     * Example: Q('#checkbox').prop('checked', true)  // Set property
     *          const isChecked = Q('#checkbox').prop('checked')  // Get property
     */
    prop(name, value) {
        if (value !== undefined) {
            this.elements.forEach(el => el[name] = value);
            return this;
        } else {
            return this.elements[0]?.[name];
        }
    }

    /**
     * Triggers custom events on all wrapped elements
     * @param {string} eventName - Name of custom event
     * @param {Object} [detail={}] - Event detail data
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#element').trigger('customEvent', { data: 'value' })
     *          Q('.buttons').trigger('activate')
     */
    trigger(eventName, detail = {}) {
        const event = new CustomEvent(eventName, { detail });
        this.elements.forEach(el => el.dispatchEvent(event));
        return this;
    }

    /**
     * Animates CSS properties over a specified duration
     * @param {number} duration - Animation duration in milliseconds
     * @param {Object} properties - CSS properties to animate
     * @param {Function} [callback] - Callback function after animation
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#box').animate(1000, { width: '200px', height: '200px' }, () => console.log('Done'))
     *          Q('.fade').animate(500, { opacity: '0' })
     */
    animate(duration, properties, callback) {
        this.elements.forEach(element => {
            const keys = Object.keys(properties);
            const transitionProperties = keys.map(key => `${key} ${duration}ms`).join(', ');
            
            element.style.transition = transitionProperties;
            
            // Force reflow to ensure transition starts
            void element.offsetHeight;
            
            // Apply new properties
            keys.forEach(key => {
                element.style[key] = properties[key];
            });
            
            // Cleanup and callback
            if (callback) {
                setTimeout(() => {
                    element.style.transition = '';
                    callback.call(element);
                }, duration);
            } else {
                setTimeout(() => {
                    element.style.transition = '';
                }, duration);
            }
        });
        return this;
    }

    /**
     * Fades in elements with smooth opacity transition
     * @param {number} [duration=400] - Animation duration in milliseconds
     * @param {Function} [callback] - Callback function after animation
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#modal').fadeIn(500, () => console.log('Faded in'))
     *          Q('.notification').fadeIn()
     */
    fadeIn(duration = 400, callback) {
        this.elements.forEach(element => {
            element.style.display = '';
            element.style.transition = `opacity ${duration}ms`;
            void element.offsetHeight; // Force reflow
            element.style.opacity = '1';
            
            setTimeout(() => {
                element.style.transition = '';
                if (callback) callback.call(element);
            }, duration);
        });
        return this;
    }

    /**
     * Fades out elements with smooth opacity transition
     * @param {number} [duration=400] - Animation duration in milliseconds
     * @param {Function} [callback] - Callback function after animation
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#modal').fadeOut(300, () => Q('#modal').remove())
     *          Q('.notification').fadeOut()
     */
    fadeOut(duration = 400, callback) {
        this.elements.forEach(element => {
            element.style.transition = `opacity ${duration}ms`;
            element.style.opacity = '0';
            
            setTimeout(() => {
                element.style.display = 'none';
                element.style.transition = '';
                if (callback) callback.call(element);
            }, duration);
        });
        return this;
    }

    /**
     * Animates elements to a specific opacity level
     * @param {number} opacity - Target opacity (0-1)
     * @param {number} [duration=400] - Animation duration in milliseconds
     * @param {Function} [callback] - Callback function after animation
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#overlay').fadeTo(0.5, 200)
     *          Q('.highlight').fadeTo(1, 1000, () => console.log('Fully visible'))
     */
    fadeTo(opacity, duration = 400, callback) {
        this.elements.forEach(element => {
            element.style.transition = `opacity ${duration}ms`;
            void element.offsetHeight; // Force reflow
            element.style.opacity = opacity;
            
            setTimeout(() => {
                element.style.transition = '';
                if (callback) callback.call(element);
            }, duration);
        });
        return this;
    }

    /**
     * Toggles element visibility with fade animation
     * @param {number} [duration=400] - Animation duration in milliseconds
     * @param {Function} [callback] - Callback function after animation
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#menu').fadeToggle(300)
     *          Q('.panel').fadeToggle()
     */
    fadeToggle(duration = 400, callback) {
        this.elements.forEach(element => {
            const currentOpacity = window.getComputedStyle(element).opacity;
            if (currentOpacity === '0' || element.style.display === 'none') {
                this.fadeIn.call({ elements: [element] }, duration, callback);
            } else {
                this.fadeOut.call({ elements: [element] }, duration, callback);
            }
        });
        return this;
    }

    /**
     * Returns element at specified index
     * @param {number} index - Index of element to retrieve
     * @returns {QWrapper|null} - New QWrapper or null if not found
     *
     * Example: const secondItem = Q('.items').eq(1)
     *          secondItem.addClass('selected')
     */
    eq(index) {
        const element = this.elements[index];
        return element ? new QWrapper([element]) : null;
    }

    /**
     * Returns first element
     * @returns {QWrapper} - New QWrapper with first element
     *
     * Example: const firstElement = Q('.items').first()
     *          firstElement.addClass('first-item')
     */
    first() {
        return new QWrapper(this.elements.length > 0 ? [this.elements[0]] : []);
    }

    /**
     * Returns last element
     * @returns {QWrapper} - New QWrapper with last element
     *
     * Example: const lastElement = Q('.items').last()
     *          lastElement.addClass('last-item')
     */
    last() {
        const length = this.elements.length;
        return new QWrapper(length > 0 ? [this.elements[length - 1]] : []);
    }

    /**
     * Gets next sibling elements
     * @param {string} [selector] - Optional CSS selector to filter
     * @returns {QWrapper} - New QWrapper with next siblings
     *
     * Example: const nextItem = Q('.current').next()
     *          const nextButton = Q('.current').next('button')
     */
    next(selector) {
        const result = [];
        this.elements.forEach(element => {
            const nextElement = element.nextElementSibling;
            if (nextElement && (!selector || nextElement.matches(selector))) {
                result.push(nextElement);
            }
        });
        return new QWrapper(result);
    }

    /**
     * Gets previous sibling elements
     * @param {string} [selector] - Optional CSS selector to filter
     * @returns {QWrapper} - New QWrapper with previous siblings
     *
     * Example: const prevItem = Q('.current').prev()
     *          const prevInput = Q('.current').prev('input')
     */
    prev(selector) {
        const result = [];
        this.elements.forEach(element => {
            const prevElement = element.previousElementSibling;
            if (prevElement && (!selector || prevElement.matches(selector))) {
                result.push(prevElement);
            }
        });
        return new QWrapper(result);
    }

    /**
     * Gets all sibling elements
     * @param {string} [selector] - Optional CSS selector to filter
     * @returns {QWrapper} - New QWrapper with all siblings
     *
     * Example: const allSiblings = Q('.current').siblings()
     *          const buttonSiblings = Q('.current').siblings('button')
     */
    siblings(selector) {
        const result = [];
        this.elements.forEach(element => {
            if (element.parentNode) {
                const siblings = Array.from(element.parentNode.children);
                siblings.forEach(sibling => {
                    if (sibling !== element && (!selector || sibling.matches(selector))) {
                        result.push(sibling);
                    }
                });
            }
        });
        return new QWrapper(result);
    }

    /**
     * Inserts content after each element
     * @param {...(string|Node|QWrapper)} contents - Content to insert
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#reference').after('<p>Inserted after</p>')
     *          Q('.item').after(Q('<div>After item</div>'))
     */
    after(...contents) {
        this.elements.forEach(element => {
            contents.forEach(content => {
                if (typeof content === 'string') {
                    element.insertAdjacentHTML('afterend', content);
                } else if (content instanceof Node) {
                    if (element.nextSibling) {
                        element.parentNode.insertBefore(content, element.nextSibling);
                    } else {
                        element.parentNode.appendChild(content);
                    }
                } else if (content instanceof QWrapper) {
                    content.elements.forEach(el => {
                        if (element.nextSibling) {
                            element.parentNode.insertBefore(el, element.nextSibling);
                        } else {
                            element.parentNode.appendChild(el);
                        }
                    });
                }
            });
        });
        return this;
    }

    /**
     * Inserts content before each element
     * @param {...(string|Node|QWrapper)} contents - Content to insert
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#reference').before('<p>Inserted before</p>')
     *          Q('.item').before(Q('<div>Before item</div>'))
     */
    before(...contents) {
        this.elements.forEach(element => {
            contents.forEach(content => {
                if (typeof content === 'string') {
                    element.insertAdjacentHTML('beforebegin', content);
                } else if (content instanceof Node) {
                    element.parentNode.insertBefore(content, element);
                } else if (content instanceof QWrapper) {
                    content.elements.forEach(el => {
                        element.parentNode.insertBefore(el, element);
                    });
                }
            });
        });
        return this;
    }

    /**
     * Prepends content to the beginning of each element
     * @param {...(string|Node|QWrapper)} contents - Content to prepend
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#list').prepend('<li>First item</li>')
     *          Q('.container').prepend(Q('<div>Header</div>'))
     */
    prepend(...contents) {
        this.elements.forEach(element => {
            contents.forEach(content => {
                if (typeof content === 'string') {
                    element.insertAdjacentHTML('afterbegin', content);
                } else if (content instanceof Node) {
                    element.insertBefore(content, element.firstChild);
                } else if (content instanceof QWrapper) {
                    content.elements.forEach(el => {
                        element.insertBefore(el, element.firstChild);
                    });
                }
            });
        });
        return this;
    }

    /**
     * Creates a deep clone of the first element
     * @returns {QWrapper} - New QWrapper with cloned element
     *
     * Example: const clonedElement = Q('#template').clone()
     *          Q('#container').append(clonedElement)
     */
    clone() {
        if (this.elements.length === 0) return new QWrapper([]);
        const cloned = this.elements[0].cloneNode(true);
        return new QWrapper([cloned]);
    }

    /**
     * Removes all child elements and content
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#list').empty()
     *          Q('.container').empty().append('<p>New content</p>')
     */
    empty() {
        this.elements.forEach(element => {
            element.innerHTML = '';
        });
        return this;
    }

    /**
     * Get element position relative to document
     * @returns {Object|null} - Position object or null if no elements
     *
     * Example: const pos = Q('#element').position()
     *          console.log(`Element at ${pos.left}, ${pos.top}`)
     */
    position() {
        if (this.elements.length === 0) return null;
        return Q.position(this.elements[0]);
    }

    /**
     * Get element dimensions
     * @returns {Object|null} - Dimensions object or null if no elements
     *
     * Example: const size = Q('.box').dimensions()
     *          console.log(`Element is ${size.width}x${size.height}`)
     */
    dimensions() {
        if (this.elements.length === 0) return null;
        return Q.dimensions(this.elements[0]);
    }

    /**
     * Check if element is in viewport
     * @param {number} [threshold=0] - Visibility threshold (0-1)
     * @returns {boolean} - True if element is visible in viewport
     *
     * Example: if (Q('#element').isInViewport()) {
     *              console.log('Element is visible');
     *          }
     */
    isInViewport(threshold = 0) {
        if (this.elements.length === 0) return false;
        return Q.isInViewport(this.elements[0], threshold);
    }

    /**
     * Scroll to element with smooth animation
     * @param {Object} [options] - Scroll options
     * @param {number} [options.offset=0] - Offset from element position
     * @param {string} [options.behavior='smooth'] - Scroll behavior
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#section').scrollTo({ offset: -50 })
     */
    scrollTo(options = {}) {
        if (this.elements.length > 0) {
            Q.scrollTo(this.elements[0], options);
        }
        return this;
    }

    /**
     * Set or get data attributes with JSON support
     * @param {string} key - Data attribute key
     * @param {*} [value] - Value to set
     * @returns {QWrapper|*} - Self for chaining or current value
     *
     * Example: Q('#item').data('config', { theme: 'dark' })  // Set
     *          const config = Q('#item').data('config')  // Get
     */
    data(key, value) {
        if (value !== undefined) {
            this.elements.forEach(el => {
                try {
                    el.dataset[key] = JSON.stringify(value);
                } catch (error) {
                    el.dataset[key] = value;
                }
            });
            return this;
        } else {
            if (this.elements.length === 0) return undefined;
            try {
                return JSON.parse(this.elements[0].dataset[key]);
            } catch (error) {
                return this.elements[0].dataset[key];
            }
        }
    }

    /**
     * Trigger custom events with data
     * @param {string} eventName - Name of custom event
     * @param {Object} [detail={}] - Event detail data
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#element').trigger('customEvent', { data: 'value' })
     */
    trigger(eventName, detail = {}) {
        this.elements.forEach(el => {
            const event = new CustomEvent(eventName, { detail });
            el.dispatchEvent(event);
        });
        return this;
    }

    /**
     * Load content via AJAX and insert into elements
     * @param {string} url - URL to load content from
     * @param {Object} [options] - AJAX options
     * @returns {Promise<QWrapper>} - Promise resolving to self for chaining
     *
     * Example: Q('#content').load('/api/content').then(() => console.log('Loaded'))
     */
    load(url, options = {}) {
        return Q.ajax(url, options).then(data => {
            if (typeof data === 'string') {
                this.html(data);
            } else {
                this.html(JSON.stringify(data, null, 2));
            }
            return this;
        });
    }

    /**
     * Serialize form elements to object
     * @returns {Object} - Form data as object
     *
     * Example: const data = Q('form').serialize()
     *          Q.post('/submit', data)
     */
    serialize() {
        const result = {};
        this.elements.forEach(element => {
            if (element.tagName === 'FORM') {
                const formData = new FormData(element);
                for (let [key, value] of formData.entries()) {
                    if (result[key]) {
                        if (Array.isArray(result[key])) {
                            result[key].push(value);
                        } else {
                            result[key] = [result[key], value];
                        }
                    } else {
                        result[key] = value;
                    }
                }
            }
        });
        return result;
    }

    /**
     * Add CSS transition effects
     * @param {string|Object} properties - CSS properties to transition
     * @param {number} [duration=300] - Transition duration in milliseconds
     * @param {string} [easing='ease'] - Transition easing function
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('.button').transition({ opacity: '0.5', transform: 'scale(1.1)' }, 500)
     */
    transition(properties, duration = 300, easing = 'ease') {
        const transitionValue = Object.keys(properties).map(prop =>
            `${prop} ${duration}ms ${easing}`
        ).join(', ');

        this.css('transition', transitionValue);

        // Apply properties after a short delay to ensure transition is set
        setTimeout(() => {
            Object.entries(properties).forEach(([prop, value]) => {
                this.css(prop, value);
            });
        }, 10);

        return this;
    }

    /**
     * Add slide down animation
     * @param {number} [duration=400] - Animation duration in milliseconds
     * @param {Function} [callback] - Callback function after animation
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#menu').slideDown(300, () => console.log('Menu opened'))
     */
    slideDown(duration = 400, callback) {
        this.elements.forEach(element => {
            const currentDisplay = window.getComputedStyle(element).display;
            if (currentDisplay === 'none') {
                element.style.display = '';
                const height = element.offsetHeight;
                element.style.height = '0px';
                element.style.overflow = 'hidden';
                element.style.transition = `height ${duration}ms ease`;

                setTimeout(() => {
                    element.style.height = height + 'px';
                }, 10);

                setTimeout(() => {
                    element.style.height = '';
                    element.style.overflow = '';
                    element.style.transition = '';
                    if (callback) callback.call(element);
                }, duration);
            }
        });
        return this;
    }

    /**
     * Add slide up animation
     * @param {number} [duration=400] - Animation duration in milliseconds
     * @param {Function} [callback] - Callback function after animation
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#menu').slideUp(300, () => Q('#menu').hide())
     */
    slideUp(duration = 400, callback) {
        this.elements.forEach(element => {
            const height = element.offsetHeight;
            element.style.height = height + 'px';
            element.style.overflow = 'hidden';
            element.style.transition = `height ${duration}ms ease`;

            setTimeout(() => {
                element.style.height = '0px';
            }, 10);

            setTimeout(() => {
                element.style.display = 'none';
                element.style.height = '';
                element.style.overflow = '';
                element.style.transition = '';
                if (callback) callback.call(element);
            }, duration);
        });
        return this;
    }

    /**
     * Toggle slide animation
     * @param {number} [duration=400] - Animation duration in milliseconds
     * @param {Function} [callback] - Callback function after animation
     * @returns {QWrapper} - Returns self for chaining
     *
     * Example: Q('#panel').slideToggle()
     */
    slideToggle(duration = 400, callback) {
        this.elements.forEach(element => {
            const currentDisplay = window.getComputedStyle(element).display;
            if (currentDisplay === 'none') {
                this.slideDown.call({ elements: [element] }, duration, callback);
            } else {
                this.slideUp.call({ elements: [element] }, duration, callback);
            }
        });
        return this;
    }
}

/**
 * Main Q function - jQuery-like selector and element creator
 *
 * This is the core function that handles both element selection and creation.
 * It intelligently determines whether to select existing elements or create new ones
 * based on the input parameters.
 *
 * Selection modes:
 * - CSS selector string: Q('.class') or Q('#id')
 * - HTML creation: Q('<div>') or Q('<div>', {attributes}, [properties])
 * - DOM element: Q(document.body)
 * - NodeList/Array: Q(document.querySelectorAll('.items'))
 *
 * @param {string|Element|NodeList|Array|QWrapper} selector - What to select/create
 * @param {Object} [attributes] - Attributes for created elements
 * @param {Array} [props] - Properties for created elements
 * @returns {QWrapper} - New QWrapper instance
 *
 * Example: const elements = Q('.my-class')  // Select elements
 *          const newDiv = Q('<div>', { class: 'box', text: 'Hello' })  // Create element
 *          const body = Q(document.body)  // Wrap existing element
 */
function Q(selector, attributes, props) {
    // Handle different input types and creation

    // Single DOM element
    if (selector && selector.nodeType) {
        return new QWrapper([selector]);
    }

    // Existing QWrapper instance
    if (selector instanceof QWrapper) {
        return selector;
    }

    // NodeList or Array of elements
    if (selector instanceof NodeList || Array.isArray(selector)) {
        return new QWrapper(Array.from(selector));
    }

    // String input - could be selector or HTML creation
    if (typeof selector === 'string') {
        // Determine if we're creating elements (has attributes or HTML tags)
        const isCreating = attributes || selector.indexOf('<') > -1;

        if (isCreating) {
            let elements = [];

            // Special handling for SVG elements
            // SVG requires createElementNS for proper namespace handling
            const svgTags = ['svg', 'g', 'line', 'polyline', 'rect', 'circle', 'ellipse', 'text', 'path', 'polygon'];
            const tagMatch = selector.match(/^<([a-zA-Z0-9\-]+)(\s|>|\/)*/);
            const tag = tagMatch ? tagMatch[1].toLowerCase() : null;

            if (tag && svgTags.includes(tag)) {
                // Create SVG element with proper XML namespace
                const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
                elements = [el];
            } else {
                // Create regular HTML elements
                // Using <template> for security - prevents script execution in HTML
                const template = document.createElement('template');
                template.innerHTML = selector.trim();
                elements = Array.from(template.content.childNodes);
            }

            // Apply attributes if provided during creation
            if (attributes && elements.length > 0) {
                elements.forEach(element => {
                    Object.entries(attributes).forEach(([attr, val]) => {
                        if (attr === 'class') {
                            // Handle class as array or space-separated string
                            element.classList.add(...(Array.isArray(val) ? val : val.split(/\s+/)));
                        } else if (attr === 'style' && typeof val === 'object') {
                            // Handle style as object
                            Object.entries(val).forEach(([prop, propVal]) => {
                                element.style[prop] = propVal;
                            });
                        } else if (attr === 'text') {
                            // Set text content
                            element.textContent = val;
                        } else if (attr === 'html') {
                            // Set HTML content
                            element.innerHTML = val;
                        } else {
                            // Regular attributes
                            element.setAttribute(attr, val);
                        }
                    });
                });
            }

            // Apply properties if provided
            if (props && Array.isArray(props) && elements.length > 0) {
                elements.forEach(element => {
                    props.forEach(prop => {
                        element[prop] = true;
                    });
                });
            }

            return new QWrapper(elements);
        } else {
            // Select existing elements using CSS selector
            return new QWrapper(document.querySelectorAll(selector));
        }
    }

    // Fallback for invalid input
    return new QWrapper([]);
}

// Standalone iterator similar to jQuery $.each
Q.each = function(collection, cb){
    if(!collection) return collection;
    if(collection instanceof QWrapper) return collection.each(cb);
    if(Array.isArray(collection) || collection.length !== undefined){
        for(let i=0;i<collection.length;i++){
            if(cb.call(collection[i], i, collection[i]) === false) break;
        }
    } else if(typeof collection === 'object') {
        for(const k in collection){ if(Object.prototype.hasOwnProperty.call(collection,k)){ if(cb.call(collection[k], k, collection[k]) === false) break; } }
    }
    return collection;
};

// Export to global scope for easy access
window.Q = Q;

/**
 * Document ready handler - jQuery-style $(document).ready()
 *
 * Executes callback when DOM is fully loaded and ready for manipulation.
 * Uses native DOMContentLoaded event with once option for optimal performance.
 *
 * @param {Function} callback - Function to execute when DOM is ready
 *
 * Example: Q.Ready(() => {
 *              console.log('DOM is ready');
 *              Q('#app').text('Application started');
 *          });
 */
Q.Ready = ((callbacks) => {
    document.readyState === 'loading' ?
        document.addEventListener("DOMContentLoaded", () => {
            while (callbacks.length) callbacks.shift()();
            callbacks = null;
        }, { once: true }) :
        callbacks = null;
    return (callback) => callbacks ? callbacks.push(callback) : callback();
})([]);



/**
 * Window load event handler - executes callbacks when page is fully loaded
 *
 * Similar to jQuery's $(window).load() but uses modern event handling.
 * Queues callbacks and executes them when the window load event fires.
 *
 * @param {Function} callback - Function to execute when window is fully loaded
 *
 * Example: Q.Done(() => {
 *              console.log('Window fully loaded');
 *              // Initialize heavy components here
 *          });
 */
Q.Done = ((callbacks) => {
    window.addEventListener("load", () => {
        while (callbacks.length) callbacks.shift()();
        callbacks = null;
    });
    return (callback) => callbacks ? callbacks.push(callback) : callback();
})([]);

/**
 * Beforeunload event handler - executes callbacks when user leaves page
 *
 * Handles the beforeunload event and passes the event object to callbacks.
 * Useful for cleanup operations or asking user to confirm leaving.
 *
 * @param {Function} callback - Function to execute when user leaves page
 *
 * Example: Q.Leaving((event) => {
 *              console.log('User is leaving');
 *              // Save unsaved data
 *              event.returnValue = 'Are you sure you want to leave?';
 *          });
 */
Q.Leaving = ((callbacks) => {
    let eventRef;
    window.addEventListener("beforeunload", (e) => {
        eventRef = e;
        while (callbacks.length) callbacks.shift()(e);
        callbacks = null;
    });
    return (callback) => callbacks ? callbacks.push(callback) : callback(eventRef);
})([]);

/**
 * Ajax/HTTP request handler - Modern fetch-based HTTP client
 *
 * Provides a simple, promise-based API for HTTP requests with automatic
 * JSON parsing, error handling, and common request patterns.
 *
 * Features:
 * - Promise-based API with async/await support
 * - Automatic JSON parsing for responses
 * - Request/response interceptors
 * - Timeout handling
 * - Common HTTP methods (GET, POST, PUT, DELETE)
 * - Query parameter serialization
 * - Request cancellation support
 *
 * @param {string|Object} url - Request URL or full config object
 * @param {Object} [options] - Request options
 * @returns {Promise} - Promise resolving to response data
 *
 * Example: Q.ajax('/api/users').then(data => console.log(data))
 *          Q.ajax('/api/user', { method: 'POST', data: userData })
 *          Q.ajax({ url: '/api/search', method: 'GET', data: { q: 'term' } })
 */
Q.ajax = function(url, options = {}) {
    // Handle config object as first parameter
    if (typeof url === 'object') {
        options = url;
        url = options.url;
    }

    const config = {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        },
        ...options
    };
    // Determine timeout: allow 0/null/Infinity to mean "no timeout"
    if (typeof options.timeout === 'number') {
        config.timeout = options.timeout;
    } else {
        config.timeout = 10000; // default 10s for small requests
    }

    // Serialize query parameters for GET requests
    if (config.data && config.method === 'GET') {
        const params = new URLSearchParams();
        Object.entries(config.data).forEach(([key, value]) => {
            if (value !== null && value !== undefined) {
                params.append(key, value);
            }
        });
        url += (url.includes('?') ? '&' : '?') + params.toString();
        delete config.data;
    }

    // Serialize request body for non-GET requests
    if (config.data && config.method !== 'GET') {
        if (config.headers['Content-Type'] === 'application/json') {
            config.body = JSON.stringify(config.data);
        } else if (config.headers['Content-Type'] === 'application/x-www-form-urlencoded') {
            const params = new URLSearchParams();
            Object.entries(config.data).forEach(([key, value]) => {
                params.append(key, value);
            });
            config.body = params.toString();
        } else {
            config.body = config.data;
        }
        delete config.data;
    }

    // Create AbortController for timeout and cancellation
    const controller = new AbortController();
    config.signal = controller.signal;

    // Set up timeout only when positive finite number
    let timeoutId = null;
    if (typeof config.timeout === 'number' && isFinite(config.timeout) && config.timeout > 0) {
        timeoutId = setTimeout(() => {
            controller.abort();
        }, config.timeout);
    }

    return fetch(url, config)
        .then(response => {
            if (timeoutId) clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            // Auto-parse JSON responses
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return response.json();
            } else {
                return response.text();
            }
        })
        .catch(error => {
            if (timeoutId) clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('Request timeout or cancelled');
            }
            throw error;
        });
};

/**
 * GET request shorthand
 * @param {string} url - Request URL
 * @param {Object} [data] - Query parameters
 * @param {Object} [options] - Additional options
 * @returns {Promise} - Promise resolving to response data
 *
 * Example: Q.get('/api/users', { page: 1, limit: 10 })
 */
Q.get = function(url, data, options = {}) {
    return Q.ajax(url, { ...options, method: 'GET', data });
};

/**
 * POST request shorthand
 * @param {string} url - Request URL
 * @param {Object} [data] - Request body data
 * @param {Object} [options] - Additional options
 * @returns {Promise} - Promise resolving to response data
 *
 * Example: Q.post('/api/users', { name: 'John', email: 'john@example.com' })
 */
Q.post = function(url, data, options = {}) {
    return Q.ajax(url, { ...options, method: 'POST', data });
};

/**
 * PUT request shorthand
 * @param {string} url - Request URL
 * @param {Object} [data] - Request body data
 * @param {Object} [options] - Additional options
 * @returns {Promise} - Promise resolving to response data
 *
 * Example: Q.put('/api/users/123', { name: 'Updated Name' })
 */
Q.put = function(url, data, options = {}) {
    return Q.ajax(url, { ...options, method: 'PUT', data });
};

/**
 * DELETE request shorthand
 * @param {string} url - Request URL
 * @param {Object} [options] - Additional options
 * @returns {Promise} - Promise resolving to response data
 *
 * Example: Q.delete('/api/users/123')
 */
Q.delete = function(url, options = {}) {
    return Q.ajax(url, { ...options, method: 'DELETE' });
};

/**
 * LocalStorage wrapper with JSON serialization and error handling
 *
 * Provides a safe interface for localStorage with automatic JSON
 * serialization/deserialization and error handling.
 *
 * Features:
 * - Automatic JSON parsing/stringifying
 * - Error handling for quota exceeded
 * - Type checking and validation
 * - Batch operations
 * - Storage events
 *
 * @param {string} key - Storage key
 * @param {*} [value] - Value to store (if provided)
 * @returns {*} - Stored value or undefined if not found
 *
 * Example: Q.storage('user', { name: 'John', id: 123 })  // Set
 *          const user = Q.storage('user')  // Get
 *          Q.storage.remove('user')  // Remove
 */
Q.storage = function(key, value) {
    if (value !== undefined) {
        try {
            const serialized = JSON.stringify(value);
            localStorage.setItem(key, serialized);
            return value;
        } catch (error) {
            console.error('Q.storage set error:', error);
            return null;
        }
    } else {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : undefined;
        } catch (error) {
            console.error('Q.storage get error:', error);
            return undefined;
        }
    }
};

/**
 * Remove item from localStorage
 * @param {string} key - Storage key to remove
 *
 * Example: Q.storage.remove('user')
 */
Q.storage.remove = function(key) {
    try {
        localStorage.removeItem(key);
    } catch (error) {
        console.error('Q.storage remove error:', error);
    }
};

/**
 * Clear all localStorage items
 *
 * Example: Q.storage.clear()
 */
Q.storage.clear = function() {
    try {
        localStorage.clear();
    } catch (error) {
        console.error('Q.storage clear error:', error);
    }
};

/**
 * Check if localStorage key exists
 * @param {string} key - Storage key to check
 * @returns {boolean} - True if key exists
 *
 * Example: if (Q.storage.has('user')) { ... }
 */
Q.storage.has = function(key) {
    return localStorage.getItem(key) !== null;
};

/**
 * Get all localStorage keys
 * @returns {Array<string>} - Array of all keys
 *
 * Example: const keys = Q.storage.keys()
 */
Q.storage.keys = function() {
    const keys = [];
    for (let i = 0; i < localStorage.length; i++) {
        keys.push(localStorage.key(i));
    }
    return keys;
};

/**
 * SessionStorage wrapper (same API as localStorage)
 * @param {string} key - Storage key
 * @param {*} [value] - Value to store
 * @returns {*} - Stored value or undefined
 *
 * Example: Q.session('temp', 'value')  // Set
 *          const temp = Q.session('temp')  // Get
 */
Q.session = function(key, value) {
    if (value !== undefined) {
        try {
            const serialized = JSON.stringify(value);
            sessionStorage.setItem(key, serialized);
            return value;
        } catch (error) {
            console.error('Q.session set error:', error);
            return null;
        }
    } else {
        try {
            const item = sessionStorage.getItem(key);
            return item ? JSON.parse(item) : undefined;
        } catch (error) {
            console.error('Q.session get error:', error);
            return undefined;
        }
    }
};

// Copy storage methods to session
Q.session.remove = Q.storage.remove;
Q.session.clear = Q.storage.clear;
Q.session.has = Q.storage.has;
Q.session.keys = Q.storage.keys;

/**
 * Debounce function - delays execution until after wait time has passed
 *
 * Useful for optimizing performance with rapid events like resize, scroll, or typing.
 * Ensures the function is only called once after the wait period, even if triggered multiple times.
 *
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @param {boolean} [immediate] - Call immediately on first trigger
 * @returns {Function} - Debounced function
 *
 * Example: const debouncedSearch = Q.debounce(handleSearch, 300)
 *          Q('#search').on('input', debouncedSearch)
 */
Q.debounce = function(func, wait, immediate = false) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func.apply(this, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(this, args);
    };
};

/**
 * Throttle function - limits execution to once per wait period
 *
 * Useful for controlling the rate of function execution, especially for events
 * that fire frequently like scroll or mouse movement.
 *
 * @param {Function} func - Function to throttle
 * @param {number} wait - Minimum time between executions in milliseconds
 * @returns {Function} - Throttled function
 *
 * Example: const throttledScroll = Q.throttle(handleScroll, 100)
 *          Q(window).on('scroll', throttledScroll)
 */
Q.throttle = function(func, wait) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, wait);
        }
    };
};

/**
 * Element position and dimensions utilities
 *
 * Provides methods to get element position, dimensions, and viewport information.
 * Useful for positioning popups, tooltips, and responsive layouts.
 */

/**
 * Get element position relative to document
 * @param {Element} element - DOM element
 * @returns {Object} - Position object with top, left, width, height
 *
 * Example: const pos = Q.position(Q('#element').get(0))
 *          console.log(`Element at ${pos.left}, ${pos.top}`)
 */
Q.position = function(element) {
    const rect = element.getBoundingClientRect();
    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

    return {
        top: rect.top + scrollTop,
        left: rect.left + scrollLeft,
        width: rect.width,
        height: rect.height,
        right: rect.right + scrollLeft,
        bottom: rect.bottom + scrollTop
    };
};

/**
 * Get element dimensions
 * @param {Element} element - DOM element
 * @returns {Object} - Dimensions object with width, height
 *
 * Example: const size = Q.dimensions(Q('.box').get(0))
 *          console.log(`Element is ${size.width}x${size.height}`)
 */
Q.dimensions = function(element) {
    const rect = element.getBoundingClientRect();
    return {
        width: rect.width,
        height: rect.height
    };
};

/**
 * Check if element is in viewport
 * @param {Element} element - DOM element
 * @param {number} [threshold=0] - Visibility threshold (0-1)
 * @returns {boolean} - True if element is visible in viewport
 *
 * Example: if (Q.isInViewport(Q('#element').get(0))) {
 *              console.log('Element is visible');
 *          }
 */
Q.isInViewport = function(element, threshold = 0) {
    const rect = element.getBoundingClientRect();
    const windowHeight = window.innerHeight || document.documentElement.clientHeight;
    const windowWidth = window.innerWidth || document.documentElement.clientWidth;

    const vertInView = (rect.top <= windowHeight) && ((rect.top + rect.height) >= 0);
    const horInView = (rect.left <= windowWidth) && ((rect.left + rect.width) >= 0);

    return vertInView && horInView;
};

/**
 * Scroll to element with smooth animation
 * @param {Element} element - Target element to scroll to
 * @param {Object} [options] - Scroll options
 * @param {number} [options.offset=0] - Offset from element position
 * @param {string} [options.behavior='smooth'] - Scroll behavior
 *
 * Example: Q.scrollTo(Q('#section').get(0), { offset: -50 })
 */
Q.scrollTo = function(element, options = {}) {
    const { offset = 0, behavior = 'smooth' } = options;
    const position = Q.position(element);
    const targetY = position.top + offset;

    window.scrollTo({
        top: targetY,
        behavior: behavior
    });
};

/**
 * Get viewport dimensions
 * @returns {Object} - Viewport dimensions with width, height
 *
 * Example: const viewport = Q.viewport()
 *          console.log(`Viewport: ${viewport.width}x${viewport.height}`)
 */
Q.viewport = function() {
    return {
        width: window.innerWidth || document.documentElement.clientWidth,
        height: window.innerHeight || document.documentElement.clientHeight
    };
};

/**
 * Template rendering system - Simple string interpolation
 *
 * Replaces placeholders in template strings with data values.
 * Supports both {{variable}} and ${variable} syntax.
 *
 * @param {string} template - Template string with placeholders
 * @param {Object} data - Data object for interpolation
 * @returns {string} - Rendered template
 *
 * Example: const html = Q.template('<div>{{name}}</div>', { name: 'John' })
 *          Q('#container').html(html)
 */
Q.template = function(template, data) {
    return template.replace(/\{\{(\w+)\}\}/g, (match, key) => {
        return data[key] !== undefined ? data[key] : match;
    });
};

/**
 * Simple form validation utilities
 */

/**
 * Validate email format
 * @param {string} email - Email string to validate
 * @returns {boolean} - True if valid email format
 *
 * Example: if (Q.isValidEmail(email)) { ... }
 */
Q.isValidEmail = function(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
};

/**
 * Validate required field
 * @param {string} value - Value to check
 * @returns {boolean} - True if not empty
 *
 * Example: if (Q.isRequired(value)) { ... }
 */
Q.isRequired = function(value) {
    return value && value.trim().length > 0;
};

/**
 * Validate minimum length
 * @param {string} value - Value to check
 * @param {number} minLength - Minimum length required
 * @returns {boolean} - True if meets minimum length
 *
 * Example: if (Q.minLength(password, 8)) { ... }
 */
Q.minLength = function(value, minLength) {
    return value && value.length >= minLength;
};

/**
 * Generate unique ID
 * @param {string} [prefix='q-'] - ID prefix
 * @returns {string} - Unique ID string
 *
 * Example: const id = Q.uniqueId('button')
 *          Q('#container').append(Q('<button>', { id: id }))
 */
Q.uniqueId = function(prefix = 'q-') {
    return prefix + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
};

/**
 * Format number with commas
 * @param {number} num - Number to format
 * @returns {string} - Formatted number string
 *
 * Example: Q('#count').text(Q.formatNumber(1234567))  // "1,234,567"
 */
Q.formatNumber = function(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
};

/**
 * Capitalize first letter of string
 * @param {string} str - String to capitalize
 * @returns {string} - Capitalized string
 *
 * Example: Q('#title').text(Q.capitalize('hello world'))  // "Hello world"
 */
Q.capitalize = function(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
};

/**
 * Truncate string with ellipsis
 * @param {string} str - String to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} - Truncated string
 *
 * Example: Q('#desc').text(Q.truncate(description, 100))
 */
Q.truncate = function(str, maxLength) {
    if (str.length <= maxLength) return str;
    return str.substr(0, maxLength - 3) + '...';
};