/**
 * CardSection widget - grid container for cards
 */
class CardSection {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            cards: options.cards || []
        };
        
        this._element = Q('<div>', { class: 'card-section', id: `card-section-${this.id}` }).get();
        this._cards = [];
        
        if (this.options.cards.length) {
            this.options.cards.forEach(card => this.addCard(card));
        }
    }
    
    addCard(cardConfig) {
        let cardInstance = null;
        if (cardConfig instanceof Card) {
            cardInstance = cardConfig;
        } else {
            const cardId = `${this.id}-card-${this._cards.length + 1}`;
            cardInstance = new Card(cardId, cardConfig || {});
        }
        this._cards.push(cardInstance);
        Q(this._element).append(cardInstance.getElement());
        return cardInstance;
    }
    
    clear() {
        this._cards = [];
        Q(this._element).empty();
    }
    
    getCards() {
        return this._cards;
    }
    
    getElement() {
        return this._element;
    }
    
    static fromSchema(id, schema = {}) {
        const ui = schema.ui || {};
        const cards = (ui.cards || []).map((card, index) => ({
            title: card.title || `Card ${index + 1}`,
            subtitle: card.subtitle || '',
            footer: card.footer || ''
        }));
        return new CardSection(id, { cards });
    }
}
