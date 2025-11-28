/**
 * HootSight - Layout Builder
 * Page element constructors for schema-based UI generation
 */

const LayoutBuilder = {
    
    /**
     * Creates a card section container (4 cards per row)
     * @param {string} id - Section identifier
     * @returns {HTMLElement}
     */
    cardSection(id) {
        return Q('<div>', { class: 'card-section', id: id }).get(0);
    },
    
    /**
     * Creates a card element
     * @param {Object} options - Card options
     * @param {string} options.id - Card identifier
     * @param {string} options.title - Card header title
     * @returns {HTMLElement}
     */
    card(options = {}) {
        const card = Q('<div>', { class: 'card' }).get(0);
        if (options.id) card.id = options.id;
        
        if (options.title) {
            const header = Q('<div>', { class: 'card-header' }).get(0);
            const title = Q('<span>', { class: 'card-title', text: options.title }).get(0);
            header.appendChild(title);
            card.appendChild(header);
        }
        
        const body = Q('<div>', { class: 'card-body' }).get(0);
        card.appendChild(body);
        
        return card;
    },
    
    /**
     * Creates a heading element (title + description)
     * @param {string} title - H2 title text
     * @param {string} description - Description text
     * @returns {HTMLElement}
     */
    heading(title, description) {
        const heading = Q('<div>', { class: 'heading' }).get(0);
        
        if (title) {
            const h2 = Q('<h2>', { class: 'heading-title', text: title }).get(0);
            heading.appendChild(h2);
        }
        
        if (description) {
            const desc = Q('<div>', { class: 'heading-description', text: description }).get(0);
            heading.appendChild(desc);
        }
        
        return heading;
    },
    
    /**
     * Creates a list from JSON data
     * @param {Object} data - Key-value pairs to display
     * @returns {HTMLElement}
     */
    list(data) {
        const ul = Q('<ul>', { class: 'list' }).get(0);
        
        for (const [key, value] of Object.entries(data)) {
            const li = Q('<li>', { class: 'list-item' }).get(0);
            const keySpan = Q('<span>', { class: 'list-key', text: key }).get(0);
            const valueSpan = Q('<span>', { class: 'list-value' }).get(0);
            
            if (typeof value === 'object' && value !== null) {
                valueSpan.appendChild(this.list(value));
            } else {
                valueSpan.textContent = String(value);
            }
            
            li.appendChild(keySpan);
            li.appendChild(valueSpan);
            ul.appendChild(li);
        }
        
        return ul;
    },
    
    /**
     * Creates a table from JSON data
     * @param {Object} data - Key-value pairs to display
     * @returns {HTMLElement}
     */
    table(data) {
        const table = Q('<table>', { class: 'table' }).get(0);
        
        for (const [key, value] of Object.entries(data)) {
            const tr = Q('<tr>', { class: 'table-row' }).get(0);
            const keyTd = Q('<td>', { class: 'table-key', text: key }).get(0);
            const valueTd = Q('<td>', { class: 'table-value', text: String(value) }).get(0);
            
            tr.appendChild(keyTd);
            tr.appendChild(valueTd);
            table.appendChild(tr);
        }
        
        return table;
    },
    
    /**
     * Creates a section divider
     * @returns {HTMLElement}
     */
    divider() {
        return Q('<div>', { class: 'divider' }).get(0);
    }
};

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LayoutBuilder;
}
