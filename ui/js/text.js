(function(){
  const hs = window.Hootsight || (window.Hootsight = {});
  const state = hs.state;

  function t(key, fallback){
    const parts = key.split('.');
    let cur = state.i18n;
    for(const part of parts){
      if(cur && typeof cur === 'object' && Object.prototype.hasOwnProperty.call(cur, part)){
        cur = cur[part];
      } else {
        return fallback !== undefined ? fallback : key;
      }
    }
    return typeof cur === 'string' ? cur : (fallback !== undefined ? fallback : key);
  }

  function applyLocalization(){
    Q('[data-i18n]').each(function(){
      const key = this.getAttribute('data-i18n');
      if(!key) return;
      const txt = t(key);
      if(txt) this.textContent = txt;
    });
  }

  hs.text = {
    t,
    applyLocalization
  };
})();
