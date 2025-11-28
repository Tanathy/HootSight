/**
 * CardSection widget - grid container for cards
 */
class CardSection {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            cards: options.cards || []
        };
        
        this._element = document.createElement('div');
        this._element.className = 'card-section';
        this._element.id = `card-section-${this.id}`;
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
        this._element.appendChild(cardInstance.getElement());
        return cardInstance;
    }
    
    clear() {
        this._cards = [];
        this._element.innerHTML = '';
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
