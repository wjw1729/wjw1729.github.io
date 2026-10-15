/* global hexo */

'use strict';

// Protect $...$ / $$...$$ from marked before MathJax runs after_post_render.
hexo.extend.filter.register('before_post_render', data => {
  const enabled = data.mathjax || (hexo.config.mathjax && hexo.config.mathjax.every_page);
  if (!enabled) return data;

  const store = [];
  const save = math => {
    const key = `MATHJAXPLACEHOLDER${store.length}END`;
    store.push(math);
    return key;
  };

  let content = data.content;
  content = content.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => save(`$$${tex}$$`));
  content = content.replace(/\$([^$]+?)\$/g, (match, tex) => {
    if (!tex.trim()) return match;
    return save(`$${tex}$`);
  });

  data.content = content;
  data._mathStore = store;
  return data;
});

hexo.extend.filter.register('after_post_render', data => {
  if (!data._mathStore) return data;

  data._mathStore.forEach((math, i) => {
    data.content = data.content.split(`MATHJAXPLACEHOLDER${i}END`).join(math);
  });
  return data;
}, 1);
