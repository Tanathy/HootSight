(function(){
  const hs = window.Hootsight || (window.Hootsight = {});
  const state = hs.state;

  function t(key){
    const parts = key.split('.');
    let cur = state.i18n;
    for(const part of parts){
      if(cur && typeof cur === 'object' && Object.prototype.hasOwnProperty.call(cur, part)){
        cur = cur[part];
      } else {
        return key;
      }
    }
    return typeof cur === 'string' ? cur : key;
  }

  function tWithNewlines(key){
    const text = t(key);
    // Split by literal \n string (escaped in JSON as \\n, but comes as \n in the string)
    return text.split('\n').filter(line => line.trim());
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
    tWithNewlines,
    applyLocalization
  };
})();
