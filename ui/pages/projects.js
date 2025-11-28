/**
 * HootSight - Projects Page
 * Main page displaying project cards with statistics
 */

const ProjectsPage = {
    /**
     * Page identifier
     */
    name: 'projects',

    /**
     * Internal state
     */
    _cardSection: null,
    _projectCards: {},

    /**
     * Build the Projects page
     * @param {HTMLElement} container - Container element
     */
    build: async function(container) {
        // Clear container
        container.innerHTML = '';

        // Page heading
        const heading = new Heading('projects_heading', {
            title: lang('projects_page.title'),
            description: lang('projects_page.description')
        });
        container.appendChild(heading.getElement());

        // Card section for projects
        this._cardSection = new CardSection('projects_section');
        container.appendChild(this._cardSection.getElement());

        // Load projects
        await this.loadProjects();
    },

    /**
     * Load and render projects
     */
    loadProjects: async function() {
        try {
            const response = await API.projects.list();
            const projects = response.projects || [];

            // Clear existing cards
            this._cardSection.clear();
            this._projectCards = {};

            if (projects.length === 0) {
                const emptyCard = this._cardSection.addCard({
                    title: lang('projects_page.empty.title'),
                    subtitle: lang('projects_page.empty.description')
                });
                return;
            }

            // Create a card for each project
            projects.forEach(project => {
                this._createProjectCard(project);
            });

        } catch (err) {
            console.error('Failed to load projects:', err);
            this._cardSection.clear();
            const errorCard = this._cardSection.addCard({
                title: lang('projects_page.error.title'),
                subtitle: err.message
            });
        }
    },

    /**
     * Create a card for a project
     * @param {Object} project - Project data
     */
    _createProjectCard: function(project) {
        const name = project.name;
        const isActive = Config.getActiveProject() === name;

        // Get localized dataset type
        const datasetType = project.dataset_type 
            ? lang(`projects_page.dataset_types.${project.dataset_type}`)
            : lang('projects_page.card.unknown_type');

        // Create card with actions
        const card = this._cardSection.addCard({
            title: name,
            subtitle: datasetType
        });

        // Mark as active if this is the loaded project
        if (isActive) {
            card.getElement().classList.add('card-active');
        }

        // Store reference
        this._projectCards[name] = {
            card: card,
            tableWidget: null
        };

        // Stats table widget - use balance_analysis from cached stats
        const statsTable = new TableWidget(`${name}_stats`, {
            data: this._formatStats(project.balance_analysis),
            emptyText: lang('projects_page.card.no_stats')
        });
        card.addWidget(statsTable);
        this._projectCards[name].tableWidget = statsTable;

        // Button container
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'card-button-row';
        buttonContainer.style.display = 'flex';
        buttonContainer.style.gap = 'var(--spacing-sm)';
        buttonContainer.style.marginTop = 'var(--spacing-md)';

        // Refresh button
        const refreshBtn = new ActionButton(`${name}_refresh`, {
            label: lang('projects_page.buttons.refresh'),
            className: 'btn btn-secondary',
            onClick: () => this._refreshStats(name)
        });
        buttonContainer.appendChild(refreshBtn.getElement());

        // Load button
        const loadBtn = new ActionButton(`${name}_load`, {
            label: isActive ? lang('projects_page.buttons.loaded') : lang('projects_page.buttons.load'),
            className: isActive ? 'btn btn-success' : 'btn btn-primary',
            onClick: () => this._loadProject(name)
        });
        if (isActive) {
            loadBtn.setDisabled(true);
        }
        buttonContainer.appendChild(loadBtn.getElement());
        this._projectCards[name].loadBtn = loadBtn;

        card.addContent(buttonContainer);
    },

    /**
     * Format stats for table display
     * @param {Object} stats - Raw stats object
     * @returns {Object|null} - Formatted stats for TableWidget
     */
    _formatStats: function(stats) {
        if (!stats) return null;

        const result = {};
        result[lang('projects_page.stats.total_images')] = stats.total_images ?? '-';
        result[lang('projects_page.stats.labels')] = stats.num_labels ?? '-';
        result[lang('projects_page.stats.min_images')] = stats.min_images ?? '-';
        result[lang('projects_page.stats.max_images')] = stats.max_images ?? '-';
        return result;
    },

    /**
     * Refresh statistics for a project
     * @param {string} name - Project name
     */
    _refreshStats: async function(name) {
        const projectData = this._projectCards[name];
        if (!projectData) return;

        try {
            // Show loading state
            const loadingData = {};
            loadingData[lang('projects_page.card.loading')] = '...';
            projectData.tableWidget.setData(loadingData);

            // Recompute and save stats via dataset editor API
            await API.datasetEditor.refreshStats(name);
            
            // Fetch the computed stats
            const stats = await API.datasetEditor.getStats(name);

            // Update table with new stats - format from StatsSummary model
            const formattedStats = this._formatStatsFromEditor(stats);
            projectData.tableWidget.setData(formattedStats);

        } catch (err) {
            console.error(`Failed to refresh stats for ${name}:`, err);
            const errorData = {};
            errorData[lang('projects_page.card.error')] = err.message;
            projectData.tableWidget.setData(errorData);
        }
    },

    /**
     * Format stats from dataset editor StatsSummary
     * @param {Object} stats - StatsSummary from API
     * @returns {Object|null} - Formatted stats for TableWidget
     */
    _formatStatsFromEditor: function(stats) {
        if (!stats) return null;

        const labelCount = stats.label_distribution ? Object.keys(stats.label_distribution).length : 0;
        const counts = stats.label_distribution ? Object.values(stats.label_distribution) : [];
        const minImages = counts.length > 0 ? Math.min(...counts) : 0;
        const maxImages = counts.length > 0 ? Math.max(...counts) : 0;

        const result = {};
        result[lang('projects_page.stats.total_images')] = stats.total_images ?? '-';
        result[lang('projects_page.stats.labels')] = labelCount || '-';
        result[lang('projects_page.stats.min_images')] = minImages || '-';
        result[lang('projects_page.stats.max_images')] = maxImages || '-';
        return result;
    },

    /**
     * Load a project - loads project config and merges with global
     * @param {string} name - Project name
     */
    _loadProject: async function(name) {
        try {
            console.log('Loading project:', name);
            
            // Load and merge project config (resets to global first)
            await Config.loadProject(name);
            
            // Update all card visuals
            this._updateActiveProjectVisuals(name);
            
            // Enable delete button when a project is loaded
            if (this._deleteProjectBtn) {
                this._deleteProjectBtn.setDisabled(false);
            }
            
            console.log('Project loaded:', name);
            
        } catch (err) {
            console.error(`Failed to load project ${name}:`, err);
        }
    },

    /**
     * Update visual state of all project cards to reflect active project
     * @param {string} activeName - Name of the active project
     */
    _updateActiveProjectVisuals: function(activeName) {
        for (const [name, data] of Object.entries(this._projectCards)) {
            const isActive = name === activeName;
            const cardEl = data.card.getElement();
            
            // Update card active class
            if (isActive) {
                cardEl.classList.add('card-active');
            } else {
                cardEl.classList.remove('card-active');
            }
            
            // Update load button
            if (data.loadBtn) {
                if (isActive) {
                    data.loadBtn.setLabel(lang('projects_page.buttons.loaded'));
                    data.loadBtn.setDisabled(true);
                    data.loadBtn.getElement().className = 'btn btn-success disabled';
                } else {
                    data.loadBtn.setLabel(lang('projects_page.buttons.load'));
                    data.loadBtn.setDisabled(false);
                    data.loadBtn.getElement().className = 'btn btn-primary';
                }
            }
        }
    },

    /**
     * Setup header action buttons (called by app.js after page build)
     */
    setupHeaderActions: function() {
        const headerActions = document.getElementById('header-actions');
        if (!headerActions) return;

        // New Project button
        this._newProjectBtn = new ActionButton('projects-new', {
            label: lang('projects_page.buttons.new_project'),
            className: 'btn btn-primary',
            onClick: () => this._createNewProject()
        });
        headerActions.appendChild(this._newProjectBtn.getElement());

        // Delete Selected Project button
        this._deleteProjectBtn = new ActionButton('projects-delete', {
            label: lang('projects_page.buttons.delete_project'),
            className: 'btn btn-danger',
            onClick: () => this._deleteSelectedProject()
        });
        this._deleteProjectBtn.setDisabled(true); // Disabled until a project is selected
        headerActions.appendChild(this._deleteProjectBtn.getElement());
    },

    /**
     * Create a new project
     */
    _createNewProject: async function() {
        const name = await Modal.prompt(
            lang('projects_page.new_project.prompt'),
            lang('projects_page.new_project.title')
        );
        
        if (!name || !name.trim()) return;
        
        try {
            const result = await API.projects.create(name.trim());
            
            if (result.status === 'success') {
                // Reload projects list
                await this.loadProjects();
            } else {
                Modal.alert(result.message || lang('projects_page.new_project.error'));
            }
        } catch (err) {
            console.error('Failed to create project:', err);
            Modal.alert(lang('projects_page.new_project.error') + ': ' + err.message);
        }
    },

    /**
     * Delete the currently active/selected project
     */
    _deleteSelectedProject: async function() {
        const activeProject = Config.getActiveProject();
        if (!activeProject) {
            Modal.alert(lang('projects_page.delete_project.no_selection'));
            return;
        }
        
        const confirmed = await Modal.confirm(
            lang('projects_page.delete_project.confirm', { name: activeProject }),
            lang('projects_page.delete_project.title')
        );
        
        if (!confirmed) return;
        
        try {
            const result = await API.projects.delete(activeProject);
            
            if (result.status === 'success') {
                // Clear active project
                Config.clearActiveProject();
                // Reload projects list
                await this.loadProjects();
                // Update delete button state
                this._deleteProjectBtn.setDisabled(true);
            } else {
                Modal.alert(result.message || lang('projects_page.delete_project.error'));
            }
        } catch (err) {
            console.error('Failed to delete project:', err);
            Modal.alert(lang('projects_page.delete_project.error') + ': ' + err.message);
        }
    }
};

// Register page
Pages.register('projects', ProjectsPage);
