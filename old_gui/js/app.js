(function(){
  const hs = window.Hootsight || (window.Hootsight = {});
  const state = hs.state;
  const dom = hs.dom;
  const actions = hs.actions;
  const components = hs.components;
  const text = hs.text || {};
  const t = text.t || ((key, fallback)=> fallback ?? key);
  const applyLocalization = text.applyLocalization || (()=>{});
  const comm = hs.communication;

  function wireHeaderButtons(){
    // Page actions are now handled dynamically when pages are shown
  }

  function setPageActions(actions){
    const container = Q('#page-actions');
    container.empty();
    
    if(actions && actions.length){
      actions.forEach(action => {
        const btn = Q('<button>')
          .addClass(action.type || 'primary')
          .text(action.label)
          .on('click', action.callback);
        
        if(action.id) btn.attr('id', action.id);
        if(action.title) btn.attr('title', action.title);
        if(action.disabled) btn.prop('disabled', true);
        
        container.append(btn);
      });
    }
  }

  async function init(){
    try {
  actions.setStatus(t('status.loading_schema', 'Loading schema & config...'));
      const initial = await comm.loadInitialData();
      state.schema = initial.schema || {};
      state.config = initial.config || {};
      state.i18n = initial.localization || {};
      state.languages = initial.languages || [];
      state.activeLanguage = (initial.activeLanguage || 'en').toLowerCase();

  actions.syncAugmentationsFromConfig();

      applyLocalization();
      wireHeaderButtons();
      actions.navInit();
      actions.initLanguageSelector();

      components.showPage('projects');
      Q('.nav-item[data-page="projects"]').addClass('active');
      actions.pollStatus();
      actions.setStatus(t('footer.ready', 'Ready'));
    } catch (err){
  actions.setStatus(t('status.init_failed', 'Init failed'), true);
      console.error(err);
    }
  }

  function clearPageActions(){
    const container = Q('#page-actions');
    container.empty();
  }

  function refreshCurrentPageActions(){
    const currentPage = document.querySelector('.nav-item.active');
    if(currentPage && currentPage.dataset.page && window.components){
      const pageName = currentPage.dataset.page;
      if(window.components.setPageActionsForPage){
        window.components.setPageActionsForPage(pageName);
      }
    }
  }

  // Expose app functions globally
  window.app = {
    setPageActions,
    clearPageActions,
    refreshCurrentPageActions
  };

  init();
})();
