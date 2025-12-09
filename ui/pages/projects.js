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
    _resumeSwitch: null,
    _contextRegistered: false,

    /**
     * Build the Projects page
     * @param {HTMLElement} container - Container element
     */
    build: async function(container) {
        // Clear container
        Q(container).empty();

        // Page heading with lang keys for live translation
        const heading = new Heading('projects_heading', {
            title: lang('projects_page.title'),
            titleLangKey: 'projects_page.title',
            description: lang('projects_page.description'),
            descriptionLangKey: 'projects_page.description'
        });
        Q(container).append(heading.getElement());

        // Card section for projects
        this._cardSection = new CardSection('projects_section');
        Q(container).append(this._cardSection.getElement());

        // Load projects
        await this.loadProjects();

        // Setup context menu once
        if (!this._contextRegistered) {
            this._registerContextMenu();
            this._contextRegistered = true;
        }
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
                    titleLangKey: 'projects_page.empty.title',
                    subtitle: lang('projects_page.empty.description'),
                    subtitleLangKey: 'projects_page.empty.description'
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
                titleLangKey: 'projects_page.error.title',
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
        Q(card.getElement()).attr('data-project', name);

        // Mark as active if this is the loaded project
        if (isActive) {
            Q(card.getElement()).addClass('card-active');
        }

        // Store reference
        this._projectCards[name] = {
            card: card,
            tableWidget: null
        };

        // Stats table widget - use project-level stats
        const statsTable = new TableWidget(`${name}_stats`, {
            data: this._formatStats(project),
            emptyText: lang('projects_page.card.no_stats')
        });
        card.addWidget(statsTable);
        this._projectCards[name].tableWidget = statsTable;

        // Button container (only load button now; refresh moved to context menu)
        const buttonContainer = Q('<div>', { class: 'card-button-row' })
            .css({ display: 'flex', gap: 'var(--spacing-sm)', marginTop: 'var(--spacing-md)' })
            .get();

        const loadBtn = new ActionButton(`${name}_load`, {
            label: isActive ? lang('projects_page.buttons.loaded') : lang('projects_page.buttons.load'),
            labelLangKey: isActive ? 'projects_page.buttons.loaded' : 'projects_page.buttons.load',
            className: isActive ? 'btn btn-primary' : 'btn btn-secondary',
            onClick: () => this._loadProject(name)
        });
        if (isActive) {
            loadBtn.setDisabled(true);
        }
        Q(buttonContainer).append(loadBtn.getElement());
        this._projectCards[name].loadBtn = loadBtn;

        card.addContent(buttonContainer);
    },

    /**
     * Format project stats for display in table widget
     * @param {Object} project - Full project object with stats
     * @returns {Object|null} - Formatted stats for TableWidget
     */
    _formatStats: function(project) {
        if (!project) return null;

        const result = {};
        result[lang('projects_page.stats.total_images')] = project.image_count ?? '-';
        result[lang('projects_page.stats.balance_score')] = project.balance_score != null 
            ? (project.balance_score * 100).toFixed(1) + '%' 
            : '-';
        
        // Localize balance status key (e.g. "Excellent" -> lang key "balance_status_excellent")
        let statusDisplay = '-';
        if (project.balance_status) {
            const statusKey = `projects_page.stats.balance_status_${project.balance_status.toLowerCase()}`;
            statusDisplay = lang(statusKey) || project.balance_status;
        }
        result[lang('projects_page.stats.balance_status')] = statusDisplay;
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

            // Recompute and save stats via dataset editor API - returns updated stats
            const result = await API.datasetEditor.refreshStats(name);
            
            // Format and display the returned stats
            const formattedStats = this._formatStats({
                image_count: result.image_count,
                balance_score: result.balance_score,
                balance_status: result.balance_status
            });
            projectData.tableWidget.setData(formattedStats);

        } catch (err) {
            console.error(`Failed to refresh stats for ${name}:`, err);
            const errorData = {};
            errorData[lang('projects_page.card.error')] = err.message;
            projectData.tableWidget.setData(errorData);
        }
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
            
            // Enable delete/rename buttons when a project is loaded
            if (this._deleteProjectBtn) {
                this._deleteProjectBtn.setDisabled(false);
            }
            if (this._renameProjectBtn) {
                this._renameProjectBtn.setDisabled(false);
            }
            
            // Update training button state
            this._updateTrainingButton();
            
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
            const cardEl = Q(data.card.getElement());
            
            // Update card active class
            if (isActive) {
                cardEl.addClass('card-active');
            } else {
                cardEl.removeClass('card-active');
            }
            
            // Update load button
            if (data.loadBtn) {
                if (isActive) {
                    data.loadBtn.setLabel(lang('projects_page.buttons.loaded'), 'projects_page.buttons.loaded');
                    data.loadBtn.setDisabled(true);
                    Q(data.loadBtn.getElement()).removeClass('btn-secondary').addClass('btn-primary');
                } else {
                    data.loadBtn.setLabel(lang('projects_page.buttons.load'), 'projects_page.buttons.load');
                    data.loadBtn.setDisabled(false);
                    Q(data.loadBtn.getElement()).removeClass('btn-primary').addClass('btn-secondary');
                }
            }
        }
    },

    /**
     * Setup header action buttons (called by app.js after page build)
     */
    setupHeaderActions: function() {
        const isTraining = TrainingController.isTraining();
        const hasProject = !!Config.getActiveProject();

        this._resumeSwitch = new Switch('projects_resume_switch', {
            label: lang('projects_page.training.finetune_toggle') || 'Fine-tune existing model',
            labelLangKey: 'projects_page.training.finetune_toggle',
            default: false,
            disabled: !hasProject || isTraining
        });
        Q(this._resumeSwitch.getElement()).addClass('header-switch');

        // Wrap switch in container for proper header styling
        const switchContainer = Q('<div>', { class: 'switch-container' })
            .append(this._resumeSwitch.getElement())
            .get();

        HeaderActions.clear().add([
            {
                id: 'projects-resume-toggle',
                customElement: switchContainer
            },
            {
                id: 'projects-new',
                label: lang('projects_page.buttons.new_project'),
                labelLangKey: 'projects_page.buttons.new_project',
                type: 'secondary',
                onClick: () => this._createNewProject()
            }
        ]);

        // Store references for later updates
        this._trainingBtn = null;
        this._newProjectBtn = HeaderActions.get('projects-new');
    },

    _startTrainingFor: async function(projectName) {
        if (!projectName) return;

        // Do not queue another training if one is already active
        if (TrainingController.isTraining()) {
            Modal.alert(lang('training_controller.already_running'));
            return;
        }

        // Load project first so config is current
        await this._loadProject(projectName);

        const trainingConfig = Config.get('training') || {};
        const modelType = trainingConfig.model_type || 'resnet';
        const modelName = trainingConfig.model_name || 'resnet50';

        let resume = false;
        let wantsFineTune = false;
        let hasModel = false;
        try {
            const response = await API.projects.get(projectName);
            const projectInfo = response.project || response;
            hasModel = !!(projectInfo && projectInfo.has_model);
        } catch (e) {
            console.warn('Could not check for existing model:', e);
        }

        if (this._resumeSwitch) {
            wantsFineTune = !!this._resumeSwitch.get();
            if (wantsFineTune && !hasModel) {
                Modal.alert(lang('projects_page.training.resume_missing_checkpoint') || 'No checkpoint found. Starting a new training run.');
                wantsFineTune = false;
            }
        } else if (hasModel) {
            const choice = await Modal.confirm(
                lang('projects_page.training.resume_prompt'),
                lang('projects_page.training.resume_title'),
                lang('projects_page.training.resume_yes'),
                lang('projects_page.training.resume_no')
            );
            wantsFineTune = choice;
        }

        resume = wantsFineTune && hasModel;
        const mode = resume ? 'finetune' : 'new';
        const result = await TrainingController.startTraining(projectName, modelType, modelName, null, resume, mode);
        if (result.started) {
            this._updateTrainingButton();
        } else if (result.error) {
            Modal.alert(result.error || lang('projects_page.training.start_error'));
        }
    },

    /**
     * Start training with a specific mode
     * @param {string} projectName - Project to train
     * @param {string} mode - Training mode: 'new', 'resume', or 'finetune'
     */
    _startTrainingWithMode: async function(projectName, mode) {
        if (!projectName) return;

        // Do not queue another training if one is already active
        if (TrainingController.isTraining()) {
            Modal.alert(lang('training_controller.already_running'));
            return;
        }

        // Load project first so config is current
        await this._loadProject(projectName);

        const trainingConfig = Config.get('training') || {};
        const modelType = trainingConfig.model_type || 'resnet';
        const modelName = trainingConfig.model_name || 'resnet50';

        // Determine resume flag based on mode
        const resume = mode === 'resume' || mode === 'finetune';

        const result = await TrainingController.startTraining(projectName, modelType, modelName, null, resume, mode);
        if (result.started) {
            this._updateTrainingButton();
        } else if (result.error) {
            Modal.alert(result.error || lang('projects_page.training.start_error'));
        }
    },

    /**
     * Update training button state
     */
    _updateTrainingButton: function() {
        const isTraining = TrainingController.isTraining();
        const hasProject = !!Config.getActiveProject();

        if (this._resumeSwitch) {
            if (!hasProject || isTraining) {
                this._resumeSwitch.disable();
            } else {
                this._resumeSwitch.enable();
            }
        }
    },

    _registerContextMenu: function() {
        // Context menu on project cards
        ContextMenu.register('.card-section .card', async (element) => {
            const projectName = element.getAttribute('data-project');
            if (!projectName) return [];

            const isTraining = TrainingController.isTraining();
            const trainingProject = TrainingController.getTrainingProject();
            const trainingOnThis = isTraining && (!trainingProject || trainingProject === projectName);

            // Check if project has a trained model
            let hasModel = false;
            try {
                const response = await API.projects.get(projectName);
                const projectInfo = response.project || response;
                hasModel = !!(projectInfo && projectInfo.has_model);
            } catch (e) {
                console.warn('Could not check for existing model:', e);
            }

            const items = [
                {
                    label: lang('projects_page.buttons.refresh'),
                    icon: 'sync.svg',
                    action: () => this._refreshStats(projectName)
                }
            ];

            // Training options - show different options based on state
            if (trainingOnThis) {
                // Training is running on this project - show stop option
                items.push({
                    label: lang('projects_page.buttons.stop_training'),
                    icon: 'stop.svg',
                    danger: true,
                    action: () => TrainingController.stopTraining()
                });
            } else if (!isTraining) {
                // No training running - show training options
                items.push({
                    label: lang('projects_page.buttons.new_training'),
                    icon: 'not_started.svg',
                    action: () => this._startTrainingWithMode(projectName, 'new')
                });

                if (hasModel) {
                    items.push({
                        label: lang('projects_page.buttons.resume_training'),
                        icon: 'start.svg',
                        action: () => this._startTrainingWithMode(projectName, 'resume')
                    });
                    items.push({
                        label: lang('projects_page.buttons.finetune_training'),
                        icon: 'tune.svg',
                        action: () => this._startTrainingWithMode(projectName, 'finetune')
                    });
                }
            }

            // Other options
            items.push(
                {
                    label: lang('projects_page.buttons.rename_project'),
                    icon: 'edit.svg',
                    action: () => this._renameSelectedProject(projectName)
                },
                {
                    label: lang('projects_page.buttons.delete_project'),
                    icon: 'trash.svg',
                    danger: true,
                    action: () => this._deleteProjectByName(projectName)
                }
            );
            return items;
        });
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
                Modal.alert(langMsg(result, lang('projects_page.new_project.error')));
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
                this._renameProjectBtn.setDisabled(true);
            } else {
                Modal.alert(langMsg(result, lang('projects_page.delete_project.error')));
            }
        } catch (err) {
            console.error('Failed to delete project:', err);
            Modal.alert(lang('projects_page.delete_project.error') + ': ' + err.message);
        }
    },

    /**
     * Rename the currently active/selected project
     */
    _renameSelectedProject: async function(projectName = null) {
        const activeProject = projectName || Config.getActiveProject();
        if (!activeProject) {
            Modal.alert(lang('projects_page.rename_project.no_selection'));
            return;
        }
        
        const newName = await Modal.prompt(
            lang('projects_page.rename_project.prompt'),
            lang('projects_page.rename_project.title'),
            activeProject
        );
        
        if (!newName || newName === activeProject) return;
        
        try {
            const result = await API.projects.rename(activeProject, newName);
            
            if (result.status === 'success') {
                // Update active project to new name
                Config.setActiveProject(result.new_name || newName);
                // Reload projects list
                await this.loadProjects();
            } else {
                Modal.alert(langMsg(result, lang('projects_page.rename_project.error')));
            }
        } catch (err) {
            console.error('Failed to rename project:', err);
            Modal.alert(lang('projects_page.rename_project.error') + ': ' + err.message);
        }
    },

    _deleteProjectByName: async function(projectName) {
        const target = projectName || Config.getActiveProject();
        if (!target) {
            Modal.alert(lang('projects_page.delete_project.no_selection'));
            return;
        }

        const confirmed = await Modal.confirm(
            lang('projects_page.delete_project.confirm', { name: target }),
            lang('projects_page.delete_project.title')
        );
        
        if (!confirmed) return;
        
        try {
            const result = await API.projects.delete(target);
            
            if (result.status === 'success') {
                if (Config.getActiveProject() === target) {
                    Config.clearActiveProject();
                }
                await this.loadProjects();
                if (this._deleteProjectBtn) this._deleteProjectBtn.setDisabled(true);
                if (this._renameProjectBtn) this._renameProjectBtn.setDisabled(true);
            } else {
                Modal.alert(langMsg(result, lang('projects_page.delete_project.error')));
            }
        } catch (err) {
            console.error('Failed to delete project:', err);
            Modal.alert(lang('projects_page.delete_project.error') + ': ' + err.message);
        }
    }
};

// Register page
Pages.register('projects', ProjectsPage);
