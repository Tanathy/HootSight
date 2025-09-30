(function(){
  const hs = window.Hootsight || (window.Hootsight = {});

  async function fetchJSON(url, options = {}){
    const response = await fetch(url, options);
    if(!response.ok){
      throw new Error('HTTP ' + response.status);
    }

    const text = await response.text();
    if(!text){
      return {};
    }

    try {
      return JSON.parse(text);
    } catch {
      return { raw: text };
    }
  }

  function buildJSONOptions(method, payload){
    return {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    };
  }

  async function loadInitialData(){
    const [schemaWrap, configWrap, localizationWrap] = await Promise.all([
      fetchJSON('/config/schema'),
      fetchJSON('/config'),
      fetchJSON('/localization')
    ]);

    return {
      schema: schemaWrap?.schema || null,
      config: configWrap?.config || null,
      localization: localizationWrap?.localization || {},
      languages: localizationWrap?.languages || [],
      activeLanguage: localizationWrap?.active || 'en'
    };
  }

  async function fetchLocalization(){
    return fetchJSON('/localization');
  }

  async function switchLanguage(langCode){
    return fetchJSON('/localization/switch', buildJSONOptions('POST', { lang_code: langCode }));
  }

  async function fetchProjects(){
    return fetchJSON('/projects');
  }

  async function createProject(name){
    return fetchJSON('/projects/create', buildJSONOptions('POST', { name }));
  }

  async function fetchProjectConfig(projectName){
    return fetchJSON(`/projects/${projectName}/config`);
  }

  async function fetchConfig(){
    return fetchJSON('/config');
  }

  async function saveProjectConfig(projectName, config){
    return fetchJSON(`/projects/${projectName}/config`, buildJSONOptions('POST', config));
  }

  async function saveSystemConfig(config){
    return fetchJSON('/config', buildJSONOptions('POST', config));
  }

  async function fetchTrainingStatus(trainingId){
    const suffix = trainingId ? `?training_id=${encodeURIComponent(trainingId)}` : '';
    return fetchJSON(`/training/status${suffix}`);
  }

  async function fetchTrainingHistory(trainingId){
    const suffix = trainingId ? `?training_id=${encodeURIComponent(trainingId)}` : '';
    return fetchJSON(`/training/status/all${suffix}`);
  }

  async function startTraining(payload){
    return fetchJSON('/training/start', buildJSONOptions('POST', payload));
  }

  async function stopTraining(trainingId){
    return fetchJSON('/training/stop', buildJSONOptions('POST', { training_id: trainingId }));
  }

  async function evaluateProject(projectName, query = {}){
    const params = new URLSearchParams(query);
    const suffix = params.toString() ? `?${params.toString()}` : '';
    return fetchJSON(`/projects/${projectName}/evaluate${suffix}`);
  }

  async function previewAugmentation(projectName, payload){
    return fetchJSON(`/projects/${projectName}/augmentation/preview`, buildJSONOptions('POST', payload));
  }

  async function checkSystemUpdates(){
    return fetchJSON('/system/updates/check');
  }

  async function applySystemUpdates(paths){
    const payload = Array.isArray(paths) && paths.length ? { paths } : {};
    return fetchJSON('/system/updates/apply', buildJSONOptions('POST', payload));
  }

  async function fetchDocsList(){
    return fetchJSON('/docs/list');
  }

  async function fetchDocPage(path){
    const params = new URLSearchParams();
    if(path){
      params.set('path', path);
    }
    const suffix = params.toString() ? `?${params.toString()}` : '';
    return fetchJSON(`/docs/page${suffix}`);
  }

  hs.communication = {
    fetchJSON,
    loadInitialData,
    fetchLocalization,
    switchLanguage,
    fetchProjects,
    createProject,
    fetchProjectConfig,
    fetchConfig,
  saveProjectConfig,
  saveSystemConfig,
    fetchTrainingStatus,
    fetchTrainingHistory,
    startTraining,
    stopTraining,
    evaluateProject,
    previewAugmentation,
    checkSystemUpdates,
    applySystemUpdates,
    fetchDocsList,
    fetchDocPage
  };
})();
