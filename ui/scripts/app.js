/**
 * HootSight - Main Application
 * Core app initialization and navigation
 */

(function() {
    'use strict';

    /**
     * Current active page
     */
    let currentPage = 'main';
    
    /**
     * Waves background instance
     */
    let wavesInstance = null;

    /**
     * Initialize animated background
     */
    function initWavesBackground() {
        if (typeof Waves === 'undefined') return;
        
        wavesInstance = new Waves('#waves-bg', {
            resize: true,
            waves: 2,
            width: 200,
            hue: [11, 30],
            amplitude: 0.2,
            background: false,
            preload: true,
            fps: 15,
            speed: [0.004, 0.008]
        });
        
        wavesInstance.animate();
    }

    /**
     * Clear header actions when switching tabs
     */
    function clearHeaderActions() {
        const headerActions = Q('#header-actions').get();
        if (headerActions) {
            Q(headerActions).empty();
        }
    }

    /**
     * Navigate to a page
     * @param {string} pageName - Page identifier
     */
    async function navigateTo(pageName) {
        const container = Q('#page-content').get();
        if (!container) return;

        // Cleanup previous page if it has a cleanup method
        const previousPage = Pages.get(currentPage);
        if (previousPage && typeof previousPage.cleanup === 'function') {
            previousPage.cleanup();
        }

        // Clear header actions before switching
        clearHeaderActions();

        // Update nav active state
        const navItems = Q('.nav-item').getAll();
        navItems.forEach(item => {
            if (item.dataset.page === pageName) {
                Q(item).addClass('active');
            } else {
                Q(item).removeClass('active');
            }
        });

        // Render page
        const page = Pages.get(pageName);
        if (page && typeof page.build === 'function') {
            currentPage = pageName;
            await page.build(container);
            
            // Setup page-specific header actions if the page defines them
            if (typeof page.setupHeaderActions === 'function') {
                page.setupHeaderActions();
            }
        } else {
            Q(container).empty();
            const heading = new Heading('not_found', {
                title: 'Page Not Found',
                description: `The page "${pageName}" does not exist`,
                titleLangKey: 'app.page_not_found.title',
                descriptionLangKey: 'app.page_not_found.description'
            });
            Q(container).append(heading.getElement());
        }
    }

    /**
     * Initialize navigation click handlers
     */
    function initNavigation() {
        const navItems = Q('.nav-item').getAll();

        navItems.forEach(item => {
            Q(item).on('click', function() {
                const page = this.dataset.page;
                if (page) {
                    navigateTo(page);
                }
            });
        });
    }

    /**
     * Initialize the application
     */
    async function init() {
        // Initialize animated background
        initWavesBackground();
        
        // Load config and schema first (parallel)
        await Promise.all([
            Config.load(),
            Config.loadSchema()
        ]);

        // Load localization
        await Lang.load();

        // Build navigation from config
        Navigation.build();

        // Initialize navigation click handlers
        initNavigation();

        // Initialize training controller and check for existing training
        // This MUST complete before page navigation so TrainingMonitor is ready
        await TrainingController.init();

        // Load initial page
        const defaultPage = Navigation.getDefaultPage();
        await navigateTo(defaultPage);

        console.log('HootSight UI initialized');
    }

    // Wait for DOM
    if (document.readyState === 'loading') {
        Q(document).on('DOMContentLoaded', init);
    } else {
        init();
    }
})();
