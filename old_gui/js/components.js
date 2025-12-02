(function(){
  const hs = window.Hootsight || (window.Hootsight = {});
  const state = hs.state;
  const dom = hs.dom;
  const text = hs.text || {};
  const t = text.t || ((key, fallback)=> fallback ?? key);
  const actions = hs.actions || {};
  const markdown = hs.markdown || null;
  const _selectDropdownMap = new WeakMap();

  // Global click handler for custom dropdowns
  Q(document).on('click', (e) => {
    if(!e.target.closest('.custom-dropdown')) {
      Q('.custom-dropdown').removeClass('open');
    }
  });

  function showPage(name){
    if(!state.pages[name]){
      buildPage(name);
    }
    
    // Special handling for pages with titles and descriptions
  const pagesWithDescriptions = ['augmentation', 'training', 'dataset_editor', 'projects', 'status', 'heatmap', 'updates', 'docs', 'about'];
    if(pagesWithDescriptions.includes(name)){
      const title = t(`${name}_ui.page_title`, name.charAt(0).toUpperCase() + name.slice(1));
      const description = t(`${name}_ui.page_description`, '');
      dom.pageTitle.html(`<div class="page-title-main">${title}</div><div class="page-title-description">${description}</div>`);
    } else {
      dom.pageTitle.text(t('page.' + name, name.charAt(0).toUpperCase() + name.slice(1)));
    }
    
    dom.pageContainer.html('');
    dom.pageContainer.append(state.pages[name]);

    if(hs.datasetEditor && typeof hs.datasetEditor.setActive === 'function'){
      hs.datasetEditor.setActive(name === 'dataset_editor');
    }
    
    // Always set page actions when showing page
    setPageActionsForPage(name);
    
    if(name === 'heatmap' && typeof window.adjustImageSize === 'function'){
      setTimeout(() => window.adjustImageSize(), 0);
    }
  }

  function setPageActionsForPage(pageName){
    if(!window.app) return;
    
    switch(pageName){
      case 'heatmap':
        window.app.setPageActions([
          {
            label: t('ui.generate_heatmap', 'Generate Heatmap'),
            type: 'primary',
            callback: actions.generateHeatmap
          }
        ]);
        break;
        
      case 'training':
        const pageActions = [
          {
            label: t('actions.save_training_config', 'Save Training Config'),
            type: 'primary',
            callback: actions.saveTrainingConfig
          }
        ];
        
        if(!state.currentProject){
          pageActions[0].disabled = true;
          pageActions[0].title = t('ui.load_project_first', 'Please load a project first from the Projects tab.');
        }
        
        window.app.setPageActions(pageActions);
        break;
      
      case 'projects':
        const canCreateProject = typeof actions.showProjectCreateDialog === 'function';
        const createProjectAction = {
          label: t('projects_ui.toolbar_create', 'Create New Project'),
          type: 'primary',
          callback: canCreateProject ? actions.showProjectCreateDialog : () => {}
        };

        if(!canCreateProject){
          createProjectAction.disabled = true;
          createProjectAction.title = t('projects_ui.create_disabled_hint', 'Project creation is currently unavailable.');
        }

        window.app.setPageActions([createProjectAction]);
        break;
        
      default:
        // Clear page actions for other pages
        window.app.clearPageActions();
        break;
    }
  }

  function buildPage(name){
    switch(name){
      case 'training':
        state.pages[name] = buildTrainingSetupPage();
        break;
      case 'augmentation':
        state.pages[name] = buildAugmentationPage();
        break;
      case 'projects':
        state.pages[name] = buildProjectsPage();
        break;
      // 'dataset' page removed
      case 'dataset_editor':
        state.pages[name] = buildDatasetEditorPage();
        break;
      case 'status':
        state.pages[name] = buildStatusPage();
        break;
      case 'heatmap':
        state.pages[name] = buildHeatmapPage();
        break;
      case 'updates':
        state.pages[name] = buildUpdatesPage();
        break;
      case 'docs':
        state.pages[name] = buildDocsPage();
        break;
      case 'about':
        state.pages[name] = buildAboutPage();
        break;
      default:
        state.pages[name] = Q('<div>').text(t('ui.page_not_implemented', 'Page not implemented')).elements[0];
    }
  }

  function buildSectionCard(sectionKey, sectionSchema, current){
    const block = Q('<div class="config-section-block">');
    const heading = sectionSchema?.title || sectionKey;
    const headingText = t(`config.sections.${sectionKey}`, heading);
    Q('<div class="config-section-heading">').text(headingText).appendTo(block);

    const body = Q('<div class="config-section-content">');

    // Default handling for other sections
    if(sectionSchema.properties){
      const groups = groupProperties(sectionSchema.properties);
      const groupEntries = Object.entries(groups);
      if(groupEntries.length){
        const grid = Q('<div class="config-group-grid">');
        groupEntries.forEach(([groupKey, props], index) => {
          const groupCard = Q('<section class="config-group">');
          const title = formatGroupHeading(sectionKey, groupKey);
          const toggle = Q('<button type="button" class="config-group-header">')
            .attr('aria-expanded', index === 0 ? 'true' : 'false')
            .text(title);
          const icon = Q('<span class="config-group-icon" aria-hidden="true">');
          toggle.append(icon);
          const groupBody = Q('<div class="config-group-body">');
          if(index !== 0){
            groupBody.attr('hidden', true);
          } else {
            groupCard.addClass('expanded');
          }

          const fieldGrid = Q('<div class="config-field-grid">');
          let hasFields = false;
          Object.entries(props).forEach(([key, propSchema]) => {
            if(propSchema && propSchema.visible === false) return;
            const fieldCard = buildConfigFieldCard(sectionKey, key, propSchema, current);
            if(fieldCard){
              fieldGrid.append(fieldCard);
              hasFields = true;
            }
          });

          if(hasFields){
            groupBody.append(fieldGrid);
          } else {
            groupBody.append(Q('<p class="config-group-empty">').text(t('ui.no_data_available', 'No data available.')));
          }

          toggle.on('click', () => {
            const isExpanded = groupCard.hasClass('expanded');
            groupCard.toggleClass('expanded', !isExpanded);
            toggle.attr('aria-expanded', String(!isExpanded));
            if(isExpanded){
              groupBody.attr('hidden', true);
            } else {
              groupBody.removeAttr('hidden');
            }
          });

          groupCard.append(toggle, groupBody);
          grid.append(groupCard);
        });
        body.append(grid);
      }
    }

    block.append(body);
    return block.elements[0];
  }

  function buildCustomSchedulerLayout(sectionKey, sectionSchema, current){
    // Custom logic for schedulers - ignore schema nesting, build intuitive UI
    const block = Q('<div class="config-section-block">');
    // Implement custom scheduler UI here
    return block.elements[0];
  }

  function buildCustomLossesLayout(sectionKey, sectionSchema, current){
    // Custom logic for losses
    const block = Q('<div class="config-section-block">');
    // Implement custom losses UI here
    return block.elements[0];
  }

  function buildCustomOptimizersLayout(sectionKey, sectionSchema, current){
    // Custom logic for optimizers
    const block = Q('<div class="config-section-block">');
    // Implement custom optimizers UI here
    return block.elements[0];
  }

  function buildSectionEntityLayout(sectionKey, sectionSchema, current){
    if(!sectionSchema || !sectionSchema.properties) return null;
    const defaultsSchema = sectionSchema.properties.defaults;
    if(!defaultsSchema || !defaultsSchema.properties) return null;

    const sectionConfig = ensurePath(sectionKey);
    if(!sectionConfig.defaults || typeof sectionConfig.defaults !== 'object'){
      sectionConfig.defaults = {};
    }

    const container = Q('<div class="config-entity-grid">');
    const entities = [];

    Object.entries(defaultsSchema.properties).forEach(([entityKey, entitySchema]) => {
      if(!entitySchema || entitySchema.visible === false) return;

  const entityCard = hs.createCard({ classes: 'config-entity-card' });
      const entityHeader = Q('<div class="config-entity-header">');
      Q('<h4>').text(formatEntityHeading(sectionKey, entityKey)).appendTo(entityHeader);
      entityCard.append(entityHeader);

      const entityBody = Q('<div class="config-entity-body">');

      if(entitySchema.properties) {
        Object.entries(entitySchema.properties).forEach(([fieldKey, fieldSchema]) => {
          if(fieldSchema && fieldSchema.visible === false) return;

          const fieldPath = `${sectionKey}.defaults.${entityKey}.${fieldKey}`;
          const fieldValue = (sectionConfig.defaults[entityKey] || {})[fieldKey];

          const fieldElement = Q(renderField(fieldPath, fieldSchema, fieldValue, newValue => {
            setConfigValueAtPath(fieldPath, newValue);
            actions.markDirty();
          }));

          entityBody.append(fieldElement);
        });
      }

      entityCard.append(entityBody);
      entities.push(entityCard);
    });

    entities.forEach(card => container.append(card));

    return container.elements[0];
  }

  function cloneValue(value){
    if(Array.isArray(value)){
      return value.map(cloneValue);
    }
    if(value && typeof value === 'object'){
      const out = {};
      Object.entries(value).forEach(([key, val]) => {
        out[key] = cloneValue(val);
      });
      return out;
    }
    return value;
  }

  function getDynamicRegistry(group){
    if(!state.dynamicParams){
      state.dynamicParams = { optimizer: [], scheduler: [], loss: [] };
    }
    if(!state.dynamicParams[group]){
      state.dynamicParams[group] = [];
    }
    return state.dynamicParams[group];
  }

  function clearDynamicParams(group){
    const registry = getDynamicRegistry(group);
    registry.forEach(path => {
      if(state.fieldIndex && state.fieldIndex[path]){
        delete state.fieldIndex[path];
      }
    });
    state.dynamicParams[group] = [];
  }

  function trackDynamicFieldPath(group, path){
    const registry = getDynamicRegistry(group);
    if(!registry.includes(path)){
      registry.push(path);
    }
  }

  function getDefaultsForGroupType(group, typeKey){
    const configKey = group === 'optimizer' ? 'optimizers' : group === 'scheduler' ? 'schedulers' : 'losses';
    const defaults = state.config?.[configKey]?.defaults;
    if(defaults && typeof defaults === 'object' && defaults[typeKey] && typeof defaults[typeKey] === 'object'){
      return cloneValue(defaults[typeKey]);
    }
    return {};
  }

  function ensureParamStore(group, typeKey){
    const training = ensurePath('training');
    const propKey = `${group}_params`;
    if(!training[propKey] || typeof training[propKey] !== 'object'){
      training[propKey] = {};
    }
    if(!training[propKey][typeKey] || typeof training[propKey][typeKey] !== 'object'){
      training[propKey][typeKey] = getDefaultsForGroupType(group, typeKey);
    }
    const store = training[propKey][typeKey];
    if(group === 'optimizer'){
      const lrValue = training.optimizer_lr ?? training.learning_rate;
      if(typeof lrValue === 'number') store.lr = lrValue;
      const wdValue = training.optimizer_weight_decay ?? training.weight_decay;
      if(typeof wdValue === 'number') store.weight_decay = wdValue;
    } else if(group === 'scheduler'){
      if(typeof training.scheduler_step_size === 'number') store.step_size = training.scheduler_step_size;
      if(typeof training.scheduler_gamma === 'number') store.gamma = training.scheduler_gamma;
    } else if(group === 'loss'){
      if(typeof training.loss_reduction === 'string') store.reduction = training.loss_reduction;
    }
    return store;
  }

  function getSchemaForGroupType(group, typeKey){
    let schemaRoot;
    if(group === 'optimizer'){
      schemaRoot = state.schema?.properties?.optimizers?.properties?.defaults?.properties;
    } else if(group === 'scheduler'){
      schemaRoot = state.schema?.properties?.schedulers?.properties?.defaults?.properties;
    } else if(group === 'loss'){
      schemaRoot = state.schema?.properties?.losses?.properties?.defaults?.properties;
    }
    if(schemaRoot && typeof schemaRoot === 'object' && schemaRoot[typeKey]){
      return schemaRoot[typeKey];
    }
    return null;
  }

  function renderDynamicParamGroup(group, typeKey, container){
    clearDynamicParams(group);
    container.html('');
    if(!typeKey){
      container.append(Q('<div class="config-empty">').text(t('training_ui.select_type_first', 'Select a type to view parameters.')));
      return;
    }

    const typeSchema = getSchemaForGroupType(group, typeKey);
    if(!typeSchema || !typeSchema.properties){
      container.append(Q('<div class="config-empty">').text(t('training_ui.no_extra_params', 'No additional parameters for this selection.')));
      return;
    }

    const entries = Object.entries(typeSchema.properties);
    if(!entries.length){
      container.append(Q('<div class="config-empty">').text(t('training_ui.no_extra_params', 'No additional parameters for this selection.')));
      return;
    }

    const store = ensureParamStore(group, typeKey);

    entries.forEach(([fieldKey, fieldSchema]) => {
      if(fieldSchema && fieldSchema.visible === false) return;
      const path = `training.${group}_params.${typeKey}.${fieldKey}`;
      const value = store[fieldKey];
      const field = Q(renderField(path, fieldSchema, value, newValue => {
        actions.assignPathValue(state.config, path, newValue);
        if(group === 'optimizer'){
          if(fieldKey === 'lr' && (typeof newValue === 'number' || newValue === null)){
            actions.assignPathValue(state.config, 'training.optimizer_lr', newValue);
          }
          if(fieldKey === 'weight_decay' && (typeof newValue === 'number' || newValue === null)){
            actions.assignPathValue(state.config, 'training.optimizer_weight_decay', newValue);
          }
        } else if(group === 'scheduler'){
          if(fieldKey === 'step_size' && typeof newValue === 'number'){
            actions.assignPathValue(state.config, 'training.scheduler_step_size', newValue);
          }
          if(fieldKey === 'gamma' && (typeof newValue === 'number' || newValue === null)){
            actions.assignPathValue(state.config, 'training.scheduler_gamma', newValue);
          }
        } else if(group === 'loss' && fieldKey === 'reduction' && typeof newValue === 'string'){
          actions.assignPathValue(state.config, 'training.loss_reduction', newValue);
        }
        actions.markDirty();
      }));
      container.append(field);
      trackDynamicFieldPath(group, path);
    });
  }

  function buildOptimizerSettingsCard(groupName, trainingSchema, trainingConfig){
  const card = hs.createCard();
    Q('<h3 class="card-header">').text(groupName).appendTo(card);
  const body = Q('<div class="card-body">');

    const typeSchema = trainingSchema.properties.optimizer_type;
    const currentType = trainingConfig.optimizer_type || typeSchema?.default || 'adamw';
    const typeField = Q(renderField('training.optimizer_type', typeSchema, currentType, newValue => {
      ensurePath('training').optimizer_type = newValue;
      actions.markDirty();
      renderParams(newValue);
    }));
    body.append(typeField);

    const paramsSection = Q('<div class="config-dynamic-params">');
    Q('<div class="config-dynamic-params-title">').text(t('training_ui.optimizer_params_title', 'Optimizer Parameters')).appendTo(paramsSection);
    const paramsBody = Q('<div class="config-dynamic-params-body">');
    paramsSection.append(paramsBody);
    body.append(paramsSection);

    function renderParams(typeKey){
      renderDynamicParamGroup('optimizer', typeKey, paramsBody);
    }

    renderParams(currentType);
    card.append(body);
    return card.elements[0];
  }

  function buildSchedulerSettingsCard(groupName, trainingSchema, trainingConfig){
  const card = hs.createCard();
    Q('<h3 class="card-header">').text(groupName).appendTo(card);
    const body = Q('<div class="card-body column gap">');

    const typeSchema = trainingSchema.properties.scheduler_type;
    const currentType = trainingConfig.scheduler_type || typeSchema?.default || 'step_lr';
    const typeField = Q(renderField('training.scheduler_type', typeSchema, currentType, newValue => {
      ensurePath('training').scheduler_type = newValue;
      actions.markDirty();
      renderParams(newValue);
    }));
    body.append(typeField);

    const paramsSection = Q('<div class="config-dynamic-params">');
    Q('<div class="config-dynamic-params-title">').text(t('training_ui.scheduler_params_title', 'Scheduler Parameters')).appendTo(paramsSection);
    const paramsBody = Q('<div class="config-dynamic-params-body">');
    paramsSection.append(paramsBody);
    body.append(paramsSection);

    function renderParams(typeKey){
      renderDynamicParamGroup('scheduler', typeKey, paramsBody);
    }

    renderParams(currentType);
    card.append(body);
    return card.elements[0];
  }

  function buildLossSettingsCard(groupName, trainingSchema, trainingConfig){
  const card = hs.createCard();
    Q('<h3 class="card-header">').text(groupName).appendTo(card);
    const body = Q('<div class="card-body column gap">');

    const typeSchema = trainingSchema.properties.loss_type;
    const currentType = trainingConfig.loss_type || typeSchema?.default || 'cross_entropy';
    const typeField = Q(renderField('training.loss_type', typeSchema, currentType, newValue => {
      ensurePath('training').loss_type = newValue;
      actions.markDirty();
      renderParams(newValue);
    }));
    body.append(typeField);

    const paramsSection = Q('<div class="config-dynamic-params">');
    Q('<div class="config-dynamic-params-title">').text(t('training_ui.loss_params_title', 'Loss Parameters')).appendTo(paramsSection);
    const paramsBody = Q('<div class="config-dynamic-params-body">');
    paramsSection.append(paramsBody);
    body.append(paramsSection);

    function renderParams(typeKey){
      renderDynamicParamGroup('loss', typeKey, paramsBody);
    }

    renderParams(currentType);
    card.append(body);
    return card.elements[0];
  }

  function buildSimpleEntityCard(sectionKey, entityKey, entitySchema, currentDefaults) {
    if(!currentDefaults[entityKey] || typeof currentDefaults[entityKey] !== 'object'){
      currentDefaults[entityKey] = {};
    }

    const entityValue = currentDefaults[entityKey];
  const card = hs.createCard({ classes: 'config-entity-card' });
    
    // Header with title and description
    const header = Q('<div class="config-entity-card-header">');
    const title = formatEntityHeading(sectionKey, entityKey);
  Q('<h4 class="config-entity-card-title card-header">').text(title).appendTo(header);
    
    // Add description from schema localization
    let descKey;
    if (sectionKey === 'optimizers') {
      descKey = `schema.training_optimizer_type_enum_descriptor.${entityKey}`;
    } else if (sectionKey === 'schedulers') {
      descKey = `schema.training_scheduler_type_enum_descriptor.${entityKey}`;
    } else if (sectionKey === 'losses') {
      descKey = `schema.training_loss_type_enum_descriptor.${entityKey}`;
    }
    
    if (descKey) {
      const description = t(descKey, '');
      if(description && description !== descKey) {
        Q('<p class="config-entity-card-description">').text(description).appendTo(header);
      }
    }
    
    // Body with fields directly - NO groups!
    const body = Q('<div class="config-entity-card-body">');
    
    if(entitySchema.properties) {
      Object.entries(entitySchema.properties).forEach(([fieldKey, fieldSchema]) => {
        if(fieldSchema && fieldSchema.visible === false) return;
        
        const fieldPath = `${sectionKey}.defaults.${entityKey}.${fieldKey}`;
        const fieldValue = entityValue[fieldKey];
        
        const fieldElement = Q(renderField(fieldPath, fieldSchema, fieldValue, newValue => {
          setConfigValueAtPath(fieldPath, newValue);
          actions.markDirty();
        }));
        
        body.append(fieldElement);
      });
    }
    
    card.append(header, body);
    return card.elements[0];
  }



  function formatEntityHeading(sectionKey, entityKey){
    const translationKey = `config.entities.${sectionKey}.${entityKey}`;
    const translated = t(translationKey);
    if(translated && translated !== translationKey){
      return translated;
    }
    return entityKey
      .split(/[_./-]+/)
      .filter(Boolean)
      .map(segment => segment.charAt(0).toUpperCase() + segment.slice(1))
      .join(' ');
  }

  function groupProperties(props){
    const out = { general: {} };
    Object.entries(props).forEach(([key, schema]) => {
      const desc = schema.description || '';
      const match = desc.match(/\[group:([\w-]+)\]/);
      const group = match ? match[1] : 'general';
      if(!out[group]) out[group] = {};
      out[group][key] = schema;
    });
    return out;
  }

  function formatGroupHeading(sectionKey, groupKey){
    const translationKey = `config.subsections.${sectionKey}.${groupKey}`;
    const translated = t(translationKey);
    if(translated !== translationKey){
      return translated;
    }
    return groupKey
      .split(/[./]/)
      .filter(Boolean)
      .map(segment => {
        const cleaned = segment.replace(/[_-]+/g, ' ').trim();
        if(!cleaned) return '';
        if(cleaned.length <= 4){
          return cleaned.toUpperCase();
        }
        return cleaned.replace(/(^|\s)\w/g, char => char.toUpperCase());
      })
      .filter(Boolean)
      .join(' / ');
  }

  function buildConfigFieldCard(sectionKey, fieldKey, schema, current){
    const path = `${sectionKey}.${fieldKey}`;
    const value = current[fieldKey];
    const fieldWrapper = Q(renderField(path, schema, value, newValue => {
      ensurePath(sectionKey)[fieldKey] = newValue;
      actions.markDirty();
    }));

    const firstChild = fieldWrapper.children().get(0);
    let labelText;
    if(firstChild && firstChild.tagName && firstChild.tagName.toLowerCase() === 'label'){
      labelText = firstChild.textContent.trim();
      firstChild.remove();
    } else {
      labelText = formatFieldLabel(path, fieldKey);
    }

    const description = getFieldDescription(path, schema);

    if(description){
      fieldWrapper.find('.help').each(function(){
        const text = (this.textContent || '').replace(/\s+/g, ' ').trim();
        if(text && text.toLowerCase() === description.toLowerCase()){
          this.remove();
        }
      });
    }

  const card = hs.createCard({ tag: 'article', classes: 'config-field-card card--compact' });
    const header = Q('<div class="config-field-header">');
    Q('<div class="config-field-title">').text(labelText).appendTo(header);
    if(description){
      Q('<div class="config-field-description">').text(description).appendTo(header);
    } else {
      header.addClass('no-description');
    }
    card.append(header);

    const controlContainer = Q('<div class="config-field-control">');
    const remaining = fieldWrapper.children();
    if(remaining.getAll().length){
      controlContainer.append(remaining);
    }
    card.append(controlContainer);

    return card;
  }

  function formatFieldLabel(path, fieldKey){
    const labelKey = 'field.' + path.replace(/\./g, '_');
    const fallback = fieldKey
      .split(/[./]/)
      .filter(Boolean)
      .map(segment => {
        const cleaned = segment.replace(/[_-]+/g, ' ').trim();
        if(!cleaned) return '';
        if(cleaned.length <= 5){
          return cleaned.toUpperCase();
        }
        return cleaned.replace(/(^|\s)\w/g, char => char.toUpperCase());
      })
      .filter(Boolean)
      .join(' / ');
    return t(labelKey, fallback || fieldKey);
  }

  function getFieldDescription(path, schema){
    if(!schema) return '';
    const descKey = 'schema.' + path.replace(/\./g, '_') + '_description';
    const fallback = (schema.description || '').replace(/\[group:[^\]]+\]/gi, '').trim();
    return t(descKey, fallback).replace(/\s+/g, ' ').trim();
  }

  function buildTrainingSetupPage(){
    const wrap = Q('<div class="cards">');

    if(!state.schema?.properties?.training){
  const errorCard = hs.createCard();
      Q('<h2 class="card-header">').text(t('ui.error', 'Error')).appendTo(errorCard);
      Q('<div class="card-body">').text(t('ui.schema_not_loaded', 'Schema not loaded yet. Please wait...')).appendTo(errorCard);
      wrap.append(errorCard);
      return wrap.elements[0];
    }

    if(!state.config?.training){
  const errorCard = hs.createCard();
      Q('<h2 class="card-header">').text(t('ui.error', 'Error')).appendTo(errorCard);
      Q('<div class="card-body">').text(t('ui.config_not_loaded', 'Config not loaded yet. Please wait...')).appendTo(errorCard);
      wrap.append(errorCard);
      return wrap.elements[0];
    }

    const trainingGroups = {
      model_settings: ['model_type', 'model_name', 'pretrained'],
      task_configuration: ['task', 'input_size'],
      training_parameters: ['batch_size', 'epochs', 'learning_rate', 'weight_decay', 'val_ratio'],
      optimizer_settings: ['optimizer_type'],
      scheduler_settings: ['scheduler_type'],
      loss_configuration: ['loss_type'],
      data_loading: ['dataloader'],
      normalization: ['normalize'],
      checkpointing: ['checkpoint'],
      weight_initialization: ['weight_init']
    };

    const trainingSchema = state.schema.properties.training;
    const trainingConfig = state.config.training;

    Object.entries(trainingGroups).forEach(([groupKey, fields]) => {
      const groupName = (state.i18n?.groups && state.i18n.groups[groupKey]) || groupKey.replace(/_/g, ' ');

      if(groupKey === 'optimizer_settings'){
        wrap.append(buildOptimizerSettingsCard(groupName, trainingSchema, trainingConfig));
        return;
      }
      if(groupKey === 'scheduler_settings'){
        wrap.append(buildSchedulerSettingsCard(groupName, trainingSchema, trainingConfig));
        return;
      }
      if(groupKey === 'loss_configuration'){
        wrap.append(buildLossSettingsCard(groupName, trainingSchema, trainingConfig));
        return;
      }

  const card = hs.createCard();
      Q('<h3 class="card-header">').text(groupName).appendTo(card);
      const body = Q('<div class="card-body column gap">');
      let added = 0;
      fields.forEach(fieldName => {
        const fieldSchema = trainingSchema.properties[fieldName];
        if(!fieldSchema || fieldSchema.visible === false) return;
        const path = `training.${fieldName}`;
        const value = trainingConfig[fieldName];
        const field = renderField(path, fieldSchema, value, newValue => {
          ensurePath('training')[fieldName] = newValue;
          actions.markDirty();
        });
        body.append(field);
        added += 1;
      });
      if(added > 0){
        card.append(body);
        wrap.append(card);
      }
    });

    return wrap.elements[0];
  }

  function buildAugmentationPage(){
    const wrap = Q('<div class="column gap augmentation-page">');
    const phases = [
      {
        key: 'train',
        titleKey: 'augmentation_ui.train_title',
        descriptionKey: 'augmentation_ui.train_description'
      },
      {
        key: 'val',
        titleKey: 'augmentation_ui.val_title',
        descriptionKey: 'augmentation_ui.val_description'
      }
    ];

    phases.forEach(phaseInfo => {
      wrap.append(buildAugmentationPhaseCard(phaseInfo));
    });

    return wrap.elements[0];
  }

  function buildAugmentationPhaseCard({ key: phase, titleKey, descriptionKey }){
    const block = Q('<div class="augmentation-phase-block">');

    const options = actions.getAugmentationOptions(phase);
    if(options.length){
      const list = Q('<div class="column gap augmentation-toggle-list">');
      options.forEach(option => {
        const enabled = actions.isAugmentationEnabled(phase, option.key);
  const row = hs.createCard({ classes: 'augmentation-toggle-row' });
  row.attr('tabindex', '0');
        const checkboxId = `aug-toggle-${phase}-${option.key}`;

        // Header with title and toggle
        const headerWrap = Q('<div class="augmentation-toggle-header">');
        const titleId = `${checkboxId}-label`;
        Q('<div class="augmentation-toggle-title">')
          .attr('id', titleId)
          .text(t(option.labelKey, option.key))
          .appendTo(headerWrap);
        
        const controlWrap = Q('<div class="augmentation-toggle-control">');
        const toggle = Q('<label class="toggle augmentation-toggle-switch">');
        const checkbox = Q('<input type="checkbox">')
          .attr('id', checkboxId)
          .prop('checked', enabled);
        const slider = Q('<span class="slider"></span>');
        toggle.append(checkbox, slider);
        controlWrap.append(toggle);
        headerWrap.append(controlWrap);

        const textWrap = Q('<div class="augmentation-toggle-text">');
        const desc = t(option.descriptionKey, '');
        if(desc){
          Q('<div class="augmentation-toggle-description">').text(desc).appendTo(textWrap);
        }

        const fields = Array.isArray(option.fields) ? option.fields : [];
        let paramsWrap = null;
        if(fields.length){
          paramsWrap = Q('<div class="augmentation-params">');
          fields.forEach(field => {
            paramsWrap.append(buildAugmentationParamField(phase, option.key, field, enabled));
          });
          setAugmentationFieldsState(paramsWrap, enabled);
          textWrap.append(paramsWrap);
        }

        checkbox.on('change', function(){
          const isChecked = this.checked;
          actions.setAugmentationEnabled(phase, option.key, isChecked);
          if(paramsWrap){
            setAugmentationFieldsState(paramsWrap, isChecked);
            if(isChecked){
              refreshAugmentationFieldInputs(paramsWrap, phase, option.key);
            }
          }
          row.attr('aria-checked', String(isChecked));
        });
        checkbox.attr('aria-labelledby', titleId);

        textWrap.on('click', event => {
          const target = event.target;
          // Ignore clicks that originate from the toggle control itself so
          // the input/label's own behavior isn't doubled by the row click handler.
          if(target && typeof target.closest === 'function' && target.closest('.augmentation-toggle-control')){
            return;
          }
          if(paramsWrap && target && typeof target.closest === 'function' && target.closest('.augmentation-params')){
            return;
          }
          if(event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT' || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'BUTTON' || event.target.tagName === 'A'){
            return;
          }
          const node = checkbox.get();
          if(node){
            node.click();
          }
        });

        row.on('click', event => {
          const target = event.target;
          // If click originated from the toggle control (label/slider/input), don't
          // handle it here - the toggle control will handle its own click.
          if(target && typeof target.closest === 'function' && target.closest('.augmentation-toggle-control')){
            return;
          }
          if(paramsWrap && target && typeof target.closest === 'function' && target.closest('.augmentation-params')){
            return;
          }
          if(event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT' || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'BUTTON' || event.target.tagName === 'A'){
            return;
          }
          const node = checkbox.get();
          if(node){
            node.click();
          }
        });

        row.on('keydown', event => {
          // Avoid toggling when keyboard interactions originate from input controls
          // or the toggle control itself; only act when the row itself is focused.
          const target = event.target;
          if(target && typeof target.closest === 'function'){
            if(target.closest('.augmentation-toggle-control') || target.closest('.augmentation-params')){
              return;
            }
          }
          if(event.key === 'Enter' || event.key === ' ' || event.key === 'Spacebar'){
            // If focus is inside an input/textarea/select, don't intercept.
            if(event.target && (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT' || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'BUTTON' || event.target.tagName === 'A')){
              return;
            }
            event.preventDefault();
            const node = checkbox.get();
            if(node){
              node.click();
            }
          }
        });

        // Set ARIA attributes
        row.attr('role', 'switch');
        row.attr('aria-checked', String(enabled));
        row.attr('aria-labelledby', titleId);

        row.append(headerWrap, textWrap);
        list.append(row);
      });
      block.append(list);
      const help = t('augmentation_ui.toggle_help', 'Toggle an augmentation to enable or disable it for this phase.');
      if(help){
        block.append(Q('<p class="augmentation-toggle-help">').text(help));
      }
    } else {
      block.append(Q('<p>').text(t('augmentation_ui.no_options', 'No augmentation options available.')));
    }

  const previewSection = hs.createCard({ classes: 'augmentation-preview-section card--glass' });
    const previewBody = Q('<div class="card-body">');
    Q('<div class="augmentation-preview-header">')
      .text(t('augmentation_ui.preview_section_title', 'Preview'))
      .appendTo(previewSection);

    const previewDescription = t('augmentation_ui.preview_description', 'Apply the current pipeline to a random dataset image.');
  if(previewDescription){
    previewBody.append(Q('<p class="augmentation-preview-description">').text(previewDescription));
  }

    const previewControls = Q('<div class="augmentation-preview-controls">');
    const previewButton = Q('<button type="button" class="secondary augmentation-preview-button">')
      .text(t('augmentation_ui.preview_button', 'Check Preview'))
      .on('click', () => actions.requestAugmentationPreview(phase));
    previewControls.append(previewButton);
  previewBody.append(previewControls);

    const previewResult = Q(`<div class="augmentation-preview-result" id="augmentation-preview-${phase}">`);
    previewBody.append(previewResult);
    previewSection.append(previewBody);
    block.append(previewSection);

    setTimeout(() => actions.renderAugmentationPreview(phase), 0);

    const customTransforms = actions.getCustomAugmentations(phase);
    if(customTransforms.length){
      const note = Q('<div class="augmentation-custom-note">');
      Q('<div class="augmentation-custom-note-title">')
        .text(t('augmentation_ui.custom_warning', 'The following transforms are preserved but cannot be edited here:'))
        .appendTo(note);
      const list = Q('<ul class="augmentation-custom-list">');
      customTransforms.forEach(item => {
        const label = item?.type || t('augmentation_ui.unknown_transform', 'Unknown transform');
        Q('<li>').text(label).appendTo(list);
      });
      note.append(list);
      block.append(note);
    }

    return block.elements[0];
  }

  function setAugmentationFieldsState(wrapper, enabled){
    if(!wrapper) return;
    wrapper.toggleClass('disabled', !enabled);
    wrapper.find('input, select, textarea').prop('disabled', !enabled);
  }

  function refreshAugmentationFieldInputs(wrapper, phase, optionKey){
    if(!wrapper) return;
    wrapper.find('[data-aug-field]').each(function(){
      const field = this.__augField;
      if(!field) return;
      const input = Q(this);
      const value = actions.getAugmentationFieldValue(phase, optionKey, field);
      if(field.type === 'boolean'){
        input.prop('checked', !!value);
      } else if(value === undefined || value === null){
        input.val('');
      } else {
        input.val(value);
      }
    });
  }

  function buildAugmentationParamField(phase, optionKey, field, enabled){
    const wrap = Q('<div class="augmentation-param-field">');
    const sanitizedPath = (field.path || '').replace(/[^a-z0-9]+/gi, '-');
    const inputId = `aug-${phase}-${optionKey}-${sanitizedPath}`.replace(/-+/g, '-').toLowerCase();

    const labelText = t(field.labelKey, field.path || 'param');
    const label = Q('<label class="augmentation-param-label">')
      .attr('for', inputId)
      .text(labelText);

    const control = Q('<div class="augmentation-param-control">');
    let input;

    if(field.type === 'boolean'){
      input = Q('<input type="checkbox">');
      input.attr('id', inputId);
      const currentValue = actions.getAugmentationFieldValue(phase, optionKey, field);
      input.prop('checked', !!currentValue);
      input.on('change', function(){
        const sanitized = actions.setAugmentationFieldValue(phase, optionKey, field, this.checked);
        input.prop('checked', !!sanitized);
      });
      const toggle = Q('<label class="toggle augmentation-param-toggle">');
      const slider = Q('<span class="slider"></span>');
      toggle.append(input, slider);
      control.append(toggle);
    } else {
      input = Q('<input type="number" class="augmentation-param-input">');
      input.attr('id', inputId);
      input.attr('inputmode', 'decimal');
      if(field.step !== undefined) input.attr('step', field.step);
      if(field.min !== undefined) input.attr('min', field.min);
      if(field.max !== undefined) input.attr('max', field.max);
      const currentValue = actions.getAugmentationFieldValue(phase, optionKey, field);
      if(currentValue !== undefined && currentValue !== null){
        input.val(currentValue);
      }
      input.on('change', function(){
        const sanitized = actions.setAugmentationFieldValue(phase, optionKey, field, this.value);
        if(sanitized === undefined || sanitized === null || Number.isNaN(sanitized)){
          input.val('');
        } else {
          input.val(sanitized);
        }
      });
      control.append(input);
    }

    input.attr('data-aug-field', 'true');
    const inputNode = input.get();
    if(inputNode){
      inputNode.__augField = field;
    }

    if(field.descriptionKey){
      const description = t(field.descriptionKey, '');
      if(description){
        control.append(Q('<div class="augmentation-param-help">').text(description));
      }
    }

    wrap.append(label, control);
    if(!enabled){
      control.find('input, select, textarea').prop('disabled', true);
    }
    return wrap;
  }

  function buildProjectsPage(){
    const page = Q('<div class="projects-page">');

    const toolbar = Q('<div class="projects-toolbar">');
    const hintText = t('projects_ui.toolbar_hint', 'Projects keep datasets, configs, and checkpoints isolated.');
    Q('<p class="projects-toolbar-hint muted">').text(hintText).appendTo(toolbar);
    page.append(toolbar);

    const cardsWrap = Q('<div class="cards" id="projects-cards">');
    page.append(cardsWrap);

    setTimeout(() => actions.refreshProjects(), 0);
    return page.elements[0];
  }

  // Dataset page removed from UI — this function intentionally left out.

  function buildDatasetEditorPage(){
    const host = document.getElementById('dataset-editor-root');
    if(!host){
      return Q('<div class="dataset-editor-unavailable">').text(t('dataset_editor_ui.unavailable', 'Dataset editor unavailable')).elements[0];
    }
    host.style.display = '';
    host.removeAttribute('aria-hidden');
    if(hs.datasetEditor && typeof hs.datasetEditor.init === 'function'){
      hs.datasetEditor.init();
    }
    return host;
  }

  function buildHeatmapPage(){
    const wrap = Q('<div class="heatmap-page">');

    const bottomSection = Q('<div class="heatmap-bottom">').css('display', 'flex');
    const left = Q('<div class="heatmap-left">');
    const imageContainer = Q('<div class="heatmap-image" id="heatmap-image">')
      .text(t('ui.no_heatmap_generated', 'No heatmap generated yet.'))
      .css('overflow', 'hidden');
    left.append(imageContainer);
    bottomSection.append(left);

    const right = Q('<div class="heatmap-right">');
    const dataContainer = Q('<div class="heatmap-data" id="heatmap-data">')
      .text(t('ui.no_data_available', 'No data available.'));
    right.append(dataContainer);
    bottomSection.append(right);

    wrap.append(bottomSection);
    return wrap.elements[0];
  }
  function buildStatusPage(){
    const wrap = Q('<div class="status-page">');
    Q('<h2 class="status-page-title">').text(t('ui.training_status', 'Training Status')).appendTo(wrap);
    const body = Q('<div class="status-panel" id="status-body">').text(t('status_graph.no_training', 'No active training.'));
    wrap.append(body);
    setTimeout(() => actions.pollStatus(), 0);
    return wrap.elements[0];
  }

  function buildUpdatesPage(){
    const wrap = Q('<div class="updates-page">');
  const card = hs.createCard();
    Q('<h3 class="card-header">').text(t('updates_ui.card_title', 'System Updates')).appendTo(card);

    const body = Q('<div class="card-body column gap">');
    Q('<p class="updates-intro muted">').text(t('updates_ui.intro', 'Compare local files with the upstream reference and synchronize missing fixes.')).appendTo(body);

    const actionsRow = Q('<div class="updates-actions">');
    const checkBtn = Q('<button type="button" class="primary" id="updates-check-button">')
      .text(t('updates_ui.check_button', 'Check for updates'));
    const applyBtn = Q('<button type="button" class="secondary" id="updates-apply-button">')
      .text(t('updates_ui.apply_button', 'Apply updates'))
      .prop('disabled', true)
      .attr('title', t('updates_ui.apply_disabled_hint', 'Run a check to enable updates.'));
    actionsRow.append(checkBtn, applyBtn);
    body.append(actionsRow);

    const statusBox = Q('<div class="updates-status" id="updates-status">')
      .text(t('updates_ui.status_idle', 'No update checks have been run yet.'));
    body.append(statusBox);

    const errorBox = Q('<div class="updates-error" id="updates-error" hidden>');
    body.append(errorBox);

    const resultsContainer = Q('<div class="updates-results" id="updates-results">');
    body.append(resultsContainer);

    const orphanedContainer = Q('<div class="updates-orphaned" id="updates-orphaned">')
      .text(t('updates_ui.orphaned_none', 'No extra local files detected.'));
    body.append(orphanedContainer);

    card.append(body);
    wrap.append(card);

    checkBtn.on('click', () => actions.runUpdatesCheck());
    applyBtn.on('click', () => actions.applySystemUpdates());
    setTimeout(() => actions.renderUpdatesState(), 0);

    return wrap.elements[0];
  }

  function buildDocsPage(){
    const page = Q('<div class="docs-page">');

    const layout = Q('<div class="docs-layout">');

  const sidebar = hs.createCard({ tag: 'aside', classes: 'docs-sidebar' });
    Q('<h3 class="card-header">').text(t('docs_ui.sidebar_title', 'Documentation')).appendTo(sidebar);
    const sidebarBody = Q('<div class="card-body docs-list-container">');
    const list = Q('<ul class="docs-list" id="docs-list" role="list">');
    const emptyState = Q('<div class="docs-empty muted" id="docs-empty">')
      .text(t('docs_ui.empty', 'No documentation files found.'));
    sidebarBody.append(list, emptyState);
    sidebar.append(sidebarBody);

  const contentCard = hs.createCard({ tag: 'section', classes: 'docs-content' });
    const header = Q('<div class="card-header docs-content-header">');
    const title = Q('<span class="docs-current-title" id="docs-current-title">')
      .text(t('docs_ui.placeholder_title', 'Documentation'));
    const openExternal = Q('<a class="docs-open-external" id="docs-open-external" target="_blank" rel="noopener noreferrer">')
      .text(t('docs_ui.open_externally', 'Open raw file'))
      .attr('hidden', true);
    header.append(title, openExternal);

    const body = Q('<div class="card-body docs-content-body">');
    const status = Q('<div class="docs-status" id="docs-status">')
      .text(t('docs_ui.loading_placeholder', 'Select a document to view.'));
    const content = Q('<div class="markdown-body docs-markdown" id="docs-content">');
    body.append(status, content);
    contentCard.append(header, body);

    layout.append(sidebar, contentCard);
    page.append(layout);

    list.on('click', (event) => {
      const target = event.target && event.target.closest ? event.target.closest('button[data-doc-path]') : null;
      if(!target) return;
      const docPath = target.getAttribute('data-doc-path');
      if(docPath && actions.openDoc){
        actions.openDoc(docPath);
      }
    });

    content.on('click', (event) => {
      if(!actions.handleDocsLinkClick) return;
      const anchor = event.target && event.target.closest ? event.target.closest('a') : null;
      if(!anchor) return;
      actions.handleDocsLinkClick(event);
    });

    setTimeout(() => {
      if(actions.initDocsPage){
        actions.initDocsPage();
      }
    }, 0);

    return page.elements[0];
  }

  function buildAboutPage(){
    const wrap = Q('<div class="about-page">');
  const card = hs.createCard();
    Q('<h3 class="card-header">').text(t('about_ui.card_title', 'About Hootsight')).appendTo(card);

    const body = Q('<div class="card-body column gap">');
    const introText = t('about_ui.intro', '');
    if(introText && introText !== 'about_ui.intro'){
      Q('<p class="about-intro muted">').text(introText).appendTo(body);
    }

    let contentSource = t('about_ui.content_markdown', '');
    if(!contentSource || contentSource === 'about_ui.content_markdown'){
      contentSource = '';
    }
    const content = Q('<div class="markdown-body">');
    if(markdown && typeof markdown.render === 'function'){
      content.html(markdown.render(contentSource));
    } else {
      content.text(contentSource);
    }

    body.append(content);
    card.append(body);
    wrap.append(card);

    return wrap.elements[0];
  }

  function renderField(path, schema, value, onChange){
    const wrap = Q('<div class="field">');
    const labelKey = 'field.' + path.replace(/\./g, '_');
    const label = Q('<label>').text(t(labelKey, path.split('.').slice(-1)[0]));
    label.appendTo(wrap);
    
    // Add help text under label if description exists (only for top-level fields to avoid duplication)
    const pathDepth = path.split('.').length;
    if(schema.description && pathDepth <= 2){
      const descKey = 'schema.' + path.replace(/\./g, '_') + '_description';
      const localizedDesc = t(descKey, schema.description);
      wrap.append(Q('<small class="help">').text(localizedDesc));
    }
    
    const type = schema.type || inferTypeFrom(value);

    if(schema.oneOf){
      const autoOpt = schema.oneOf.find(option => option.const === 'auto');
      const numOpt = schema.oneOf.find(option => option.type === 'integer' || option.type === 'number');
      if(autoOpt && numOpt){
        const selectWrap = Q('<div class="field">');
        
        // Create custom dropdown for mode selection
        const dropdownContainer = Q('<div class="custom-dropdown">');
        const dropdownButton = Q('<div class="custom-dropdown-button">');
        const dropdownValue = Q('<span class="custom-dropdown-value">');
        const dropdownArrow = Q('<span class="custom-dropdown-arrow">▼</span>');
        
        dropdownButton.append(dropdownValue, dropdownArrow);
        
        const dropdownList = Q('<div class="custom-dropdown-list">');
        
        const modes = [t('ui.auto', 'auto'), t('ui.value', 'value')];
        let currentMode = value === 'auto' || value === undefined || value === null ? t('ui.auto', 'auto') : t('ui.value', 'value');
        dropdownValue.text(currentMode);
        
        modes.forEach(mode => {
          const optionEl = Q('<div class="custom-dropdown-option">').text(mode).attr('data-value', mode);
          if(mode === currentMode) optionEl.addClass('selected');
          optionEl.on('click', (e) => {
            e.stopPropagation();
            dropdownValue.text(mode);
            dropdownContainer.removeClass('open');
            currentMode = mode;
            if(mode === t('ui.auto', 'auto')) {
              numInput.prop('disabled', true);
              onChange('auto');
            } else {
              numInput.prop('disabled', false);
              const parsed = parseInt(numInput.val(), 10);
              if(!Number.isNaN(parsed)) onChange(parsed);
            }
            // Update selected state
            dropdownList.find('.custom-dropdown-option').removeClass('selected');
            optionEl.addClass('selected');
          });
          dropdownList.append(optionEl);
        });
        
        dropdownButton.on('click', (e) => {
          e.stopPropagation();
          const isOpen = dropdownContainer.hasClass('open');
          // Close all other dropdowns first
          Q('.custom-dropdown').removeClass('open');
          if(!isOpen){
            dropdownContainer.addClass('open');
          }
        });
        
        dropdownContainer.append(dropdownButton, dropdownList);
        
        const numInput = Q('<input type="number" style="margin-top:6px;">');
        if(numOpt.minimum !== undefined) numInput.attr('min', numOpt.minimum);
        if(numOpt.maximum !== undefined) numInput.attr('max', numOpt.maximum);
        const isAuto = value === 'auto' || value === undefined || value === null;
        if(!isAuto && typeof value === 'number') numInput.val(value);
        numInput.prop('disabled', isAuto);
        numInput.on('input', () => {
          const parsed = parseInt(numInput.val(), 10);
          if(!Number.isNaN(parsed)) onChange(parsed);
        });
        selectWrap.append(dropdownContainer, numInput);
        actions.registerField(path, numInput, schema, () => currentMode === t('ui.auto', 'auto') ? 'auto' : parseInt(numInput.val(), 10));
        return selectWrap.elements[0];
      }
    }

    if(schema.enum){
      // Create custom dropdown instead of native select
      const dropdownContainer = Q('<div class="custom-dropdown">');
      const dropdownButton = Q('<div class="custom-dropdown-button">');
      const dropdownValue = Q('<span class="custom-dropdown-value">');
      const dropdownArrow = Q('<span class="custom-dropdown-arrow">▼</span>');
      
      dropdownButton.append(dropdownValue, dropdownArrow);
      
      const dropdownList = Q('<div class="custom-dropdown-list">');
      
      let options = schema.enum;
      if(schema.model_type_variants && path === 'training.model_name'){
        const modelType = state.config?.training?.model_type || 'resnet';
        options = schema.model_type_variants[modelType] || schema.enum;
      }
      
      let currentValue = value;
      if(currentValue === undefined || !options.includes(currentValue)){
        currentValue = options.length > 0 ? options[0] : '';
      }
      
      dropdownValue.text(currentValue);
      
      options.forEach(opt => {
        const optionEl = Q('<div class="custom-dropdown-option">').text(opt).attr('data-value', opt);
        if(opt === currentValue) optionEl.addClass('selected');
        optionEl.on('click', (e) => {
          e.stopPropagation();
          dropdownValue.text(opt);
          dropdownContainer.removeClass('open');
          currentValue = opt;
          onChange(opt);
          // Update enum description if exists
          if(enumDescEl) updateEnumDesc();
          // Update selected state
          dropdownList.find('.custom-dropdown-option').removeClass('selected');
          optionEl.addClass('selected');
        });
        dropdownList.append(optionEl);
      });
      
      dropdownButton.on('click', () => {
        const isOpen = dropdownContainer.hasClass('open');
        // Close all other dropdowns first
        Q('.custom-dropdown').removeClass('open');
        if(!isOpen){
          dropdownContainer.addClass('open');
        }
      });
      
      dropdownContainer.append(dropdownButton, dropdownList);
      
      let enumDescEl = null;
      if(schema.enum_descriptor && typeof schema.enum_descriptor === 'object'){
        enumDescEl = Q('<div class="enum-desc">');
        function updateEnumDesc(){
          const cur = currentValue;
          const enumKey = 'schema.' + path.replace(/\./g, '_') + '_enum_descriptor.' + cur;
          const localized = t(enumKey, schema.enum_descriptor[cur]);
          if(localized){
            enumDescEl.text(localized);
            enumDescEl.show();
          } else {
            enumDescEl.text('');
            enumDescEl.hide();
          }
        }
        wrap.append(dropdownContainer);
        wrap.append(enumDescEl);
        updateEnumDesc();
      } else {
        wrap.append(dropdownContainer);
      }

      if(schema.model_type_variants && path === 'training.model_name'){
        state.modelNameField = { dropdown: dropdownContainer, schema, onChange, availableOptions: options, currentValue: () => currentValue };
      }

      actions.registerField(path, dropdownContainer, schema, () => currentValue);
    } else if(type === 'boolean'){
      const toggle = Q('<label class="toggle">');
      const cb = Q('<input type="checkbox">');
      const slider = Q('<span class="slider"></span>');
      cb.prop('checked', !!value);
      cb.on('change', () => onChange(cb.elements[0].checked));
      toggle.append(cb, slider);
      wrap.append(toggle);
      actions.registerField(path, cb, schema, () => cb.elements[0].checked);
    } else if(type === 'number' || type === 'integer'){
      const inp = Q('<input type="number">');
      if(schema.minimum !== undefined) inp.attr('min', schema.minimum);
      if(schema.maximum !== undefined) inp.attr('max', schema.maximum);
      inp.val(value !== undefined ? value : (schema.default !== undefined ? schema.default : ''));
      inp.on('input', () => {
        const raw = inp.val();
        onChange(raw === '' ? null : (type === 'integer' ? parseInt(raw, 10) : parseFloat(raw)));
      });
      wrap.append(inp);
      actions.registerField(path, inp, schema, () => {
        const raw = inp.val();
        if(raw === '') return null;
        return type === 'integer' ? parseInt(raw, 10) : parseFloat(raw);
      });
    } else if(type === 'array'){
      if(schema.items && schema.minItems === schema.maxItems && (schema.items.type === 'number' || schema.items.type === 'integer')){
        const length = schema.minItems;
        const row = Q('<div class="field-row">');
        const arr = Array.isArray(value) ? value.slice() : new Array(length).fill('');
        for(let i = 0; i < length; i += 1){
          const el = Q('<input type="number">');
          if(arr[i] !== undefined) el.val(arr[i]);
          el.on('input', () => {
            const out = [];
            row.find('input').each(function(){
              const val = this.value;
              if(val !== '' && !Number.isNaN(parseFloat(val))) out.push(parseFloat(val));
              else out.push(null);
            });
            onChange(out);
          });
          row.append(el);
        }
        wrap.append(row);
        actions.registerField(path, row, schema, () => {
          const out = [];
          row.find('input').each(function(){
            const val = this.value;
            out.push(val === '' ? null : parseFloat(val));
          });
          return out;
        });
      } else {
        const area = Q('<textarea rows="3">');
        if(Array.isArray(value)) area.text(value.join('\n'));
        area.on('input', () => {
          const lines = area.val().split(/\n+/).filter(item => item.trim().length);
          onChange(lines);
        });
        wrap.append(area);
        if(schema.items && schema.items.type === 'number'){
          wrap.append(Q('<small class="help">').text(t('ui.one_number_per_line', 'One number per line')));
        }
        actions.registerField(path, area, schema, () => area.val().split(/\n+/).filter(item => item.trim().length));
      }
    } else if(type === 'object'){
      const objVal = value && typeof value === 'object' ? value : {};
      const hasProps = schema.properties && Object.keys(schema.properties).length > 0;
      if(!hasProps){
        wrap.append(Q('<small class="help">').text(t('ui.empty_object', 'Empty object')));
      } else {
        const group = Q('<div class="object-group">');
        const header = Q('<div class="object-header" role="button" tabindex="0" aria-expanded="false">');
        const caret = Q('<span class="caret" aria-hidden="true"></span>');
        const title = Q('<span class="object-title">').text(t('field.' + path.replace(/\./g, '_'), path.split('.').slice(-1)[0]));
        header.append(caret, title);
        const inner = Q('<div class="object-inner">');

        function setOpen(open){
          if(open){
            group.addClass('open');
            header.attr('aria-expanded', 'true');
          } else {
            group.removeClass('open');
            header.attr('aria-expanded', 'false');
          }
        }
        header.on('click', () => setOpen(!group.hasClass('open')));
        header.on('keydown', (e) => {
          if(e.key === 'Enter' || e.key === ' '){
            e.preventDefault();
            setOpen(!group.hasClass('open'));
          }
        });

        Object.entries(schema.properties).forEach(([key, subSchema]) => {
          if(subSchema && subSchema.visible === false) return;
          const childPath = `${path}.${key}`;
          const current = objVal[key];
          const child = renderField(childPath, subSchema, current, newValue => {
            objVal[key] = newValue;
            onChange(objVal);
            actions.markDirty();
          });
          inner.append(child);
        });
        group.append(header, inner);
        wrap.append(group);
        actions.registerField(path, group, schema, () => {
          const out = {};
          if(schema.properties){
            Object.keys(schema.properties).forEach(key => {
              const childMeta = state.fieldIndex[`${path}.${key}`];
              if(childMeta){
                try {
                  out[key] = childMeta.getter();
                } catch {}
              }
            });
          }
          return out;
        });
      }
    } else {
      const inp = Q('<input type="text">');
      inp.val(value !== undefined ? value : (schema.default !== undefined ? schema.default : ''));
      inp.on('input', () => onChange(inp.val()));
      wrap.append(inp);
      actions.registerField(path, inp, schema, () => inp.val());
    }

    return wrap.elements[0];
  }

  function ensurePath(section){
    if(!state.config[section]) state.config[section] = {};
    return state.config[section];
  }

  function setConfigValueAtPath(path, value){
    if(!path) return;
    if(!state.config || typeof state.config !== 'object'){
      state.config = {};
    }
    const segments = path.split('.');
    let target = state.config;
    for(let i = 0; i < segments.length - 1; i += 1){
      const key = segments[i];
      if(!target[key] || typeof target[key] !== 'object'){
        target[key] = {};
      }
      target = target[key];
    }
    target[segments[segments.length - 1]] = value;
  }

  function inferTypeFrom(value){
    if(value === null || value === undefined) return 'string';
    if(Array.isArray(value)) return 'array';
    switch(typeof value){
      case 'number':
        return Number.isInteger(value) ? 'integer' : 'number';
      case 'boolean':
        return 'boolean';
      default:
        return 'string';
    }
  }

  function inferPrimitiveType(value){
    if(typeof value === 'number') return Number.isInteger(value) ? 'integer' : 'number';
    if(typeof value === 'boolean') return 'boolean';
    if(typeof value === 'string') return 'string';
    return 'string';
  }

  /**
   * Convert a native <select> element into a visible custom-dropdown while
   * keeping the underlying select for form compatibility and event dispatch.
   * If the select has already been converted, sync() will update the options.
   * @param {HTMLSelectElement} selectEl
   * @returns {Object|null} API object with { container, select, sync, destroy }
   */
  function createCustomDropdownFromSelect(selectEl){
    if(!selectEl || selectEl.tagName.toLowerCase() !== 'select') return null;
    // If already converted, just sync
    if(_selectDropdownMap.has(selectEl)){
      const obj = _selectDropdownMap.get(selectEl);
      obj.sync();
      return obj;
    }

    const container = Q('<div class="custom-dropdown">');
    const button = Q('<div class="custom-dropdown-button">');
    const valueEl = Q('<span class="custom-dropdown-value">');
    const arrow = Q('<span class="custom-dropdown-arrow">▼</span>');
    button.append(valueEl, arrow);
    const list = Q('<div class="custom-dropdown-list">');
    container.append(button, list);

    // Insert the dropdown before the select and hide the select
    const parent = selectEl.parentNode;
    if(parent){
      parent.insertBefore(container.elements[0], selectEl);
    }
    // Hide native select but keep it accessible
    selectEl.style.display = 'none';

    function buildFromSelect(){
      list.html('');
      const children = Array.from(selectEl.children);
      const setValue = (text, val) => { valueEl.text(text); selectEl.value = val; };

      // Helper to create option element
      function createOption(optionNode, appendTo){
        const optionEl = Q('<div class="custom-dropdown-option">').text(optionNode.text).attr('data-value', optionNode.value);
        if(optionNode.disabled) optionEl.addClass('disabled');
        if(optionNode.value === selectEl.value) optionEl.addClass('selected');
        optionEl.on('click', (e) => {
          e.stopPropagation();
          if(optionNode.disabled) return;
          // Update native select and dispatch change
          selectEl.value = optionNode.value;
          // Update UI
          list.find('.custom-dropdown-option').removeClass('selected');
          optionEl.addClass('selected');
          setValue(optionNode.text, optionNode.value);
          container.removeClass('open');
          selectEl.dispatchEvent(new Event('change', { bubbles: true }));
        });
        appendTo.append(optionEl);
      }

      children.forEach((node) => {
        const tag = node.tagName.toLowerCase();
        if(tag === 'optgroup'){
          const groupWrap = Q('<div class="custom-dropdown-optgroup">');
          const groupLabel = Q('<div class="custom-dropdown-optgroup-label">').text(node.label);
          groupWrap.append(groupLabel);
          Array.from(node.children).forEach(opt => createOption(opt, groupWrap));
          list.append(groupWrap);
        } else if(tag === 'option'){
          createOption(node, list);
        }
      });

      const cur = selectEl.selectedOptions && selectEl.selectedOptions[0];
      if(cur) valueEl.text(cur.text);
      else {
        const first = selectEl.options && selectEl.options.length ? selectEl.options[0] : null;
        valueEl.text(first ? first.text : '');
      }
    }

    // Toggle open/close
    button.on('click', (e) => {
      e.stopPropagation();
      const isOpen = container.hasClass('open');
      Q('.custom-dropdown').removeClass('open');
      if(!isOpen) container.addClass('open');
    });

    const api = {
      container: container.elements[0],
      select: selectEl,
      sync: buildFromSelect,
      destroy(){
        container.remove();
        selectEl.style.display = '';
        _selectDropdownMap.delete(selectEl);
      }
    };

    _selectDropdownMap.set(selectEl, api);
    buildFromSelect();

    // Bind label clicks to open/close the custom dropdown
    if(selectEl.id){
      try {
        const lbl = document.querySelector('label[for="' + selectEl.id + '"]');
        if(lbl) {
          lbl.addEventListener('click', (e) => {
            e.preventDefault();
            const isOpen = container.hasClass('open');
            Q('.custom-dropdown').removeClass('open');
            if(!isOpen) container.addClass('open');
          });
        }
      } catch (ex) {
        // ignore.
      }
    }
    return api;
  }

  window.adjustImageSize = function(){
    const mainEl = document.querySelector('main#page-container');
    const heatmapPage = document.querySelector('.heatmap-page');
    const imageWrap = document.getElementById('heatmap-image');
    const imgEl = imageWrap ? imageWrap.querySelector('img') : null;
    if(!mainEl || !heatmapPage || !imageWrap) return;

    const px = value => {
      const num = parseFloat(value || '0');
      return Number.isFinite(num) ? num : 0;
    };

    const vbox = el => {
      const cs = getComputedStyle(el);
      return {
        pt: px(cs.paddingTop), pb: px(cs.paddingBottom),
        mt: px(cs.marginTop), mb: px(cs.marginBottom),
        bt: px(cs.borderTopWidth), bb: px(cs.borderBottomWidth)
      };
    };

    const mainBox = vbox(mainEl);
    const mainContentHeight = Math.max(0, mainEl.clientHeight - mainBox.pt - mainBox.pb);

    const topEl = heatmapPage.querySelector('.heatmap-top');
    const topBox = topEl ? vbox(topEl) : { pt: 0, pb: 0, mt: 0, mb: 0, bt: 0, bb: 0 };
    const topHeight = topEl ? topEl.getBoundingClientRect().height : 0;
    const topOuter = topHeight + topBox.mt + topBox.mb;

    const availableHeight = Math.max(0, mainContentHeight - topOuter);

    const bottomEl = heatmapPage.querySelector('.heatmap-bottom');
    let bottomHeight = availableHeight;
    if(bottomEl){
      const bBox = vbox(bottomEl);
      const extras = bBox.pt + bBox.pb + bBox.bt + bBox.bb + bBox.mt + bBox.mb;
      bottomHeight = Math.max(0, availableHeight - extras);
      bottomEl.style.minHeight = '0px';
      bottomEl.style.height = `${bottomHeight}px`;
    }

    const wrapBox = vbox(imageWrap);
    const wrapExtras = wrapBox.pt + wrapBox.pb + wrapBox.bt + wrapBox.bb + wrapBox.mt + wrapBox.mb;
    const targetHeight = Math.max(0, bottomHeight - wrapExtras);
    imageWrap.style.height = `${targetHeight}px`;
    imageWrap.style.width = '100%';

    if(imgEl){
      imgEl.style.width = '100%';
      imgEl.style.height = '100%';
      imgEl.style.objectFit = 'contain';
      imgEl.style.display = 'block';
    }
  };

  let resizeTimeout;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => window.adjustImageSize(), 100);
  });
  window.addEventListener('orientationchange', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => window.adjustImageSize(), 100);
  });

  hs.components = {
    showPage,
    buildPage,
    renderField,
    createCustomDropdownFromSelect,
    setPageActionsForPage
  };
})();
