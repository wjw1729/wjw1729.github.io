/* global hexo */

'use strict';

// Homepage lists titles only. Full text stays on the post page.
// Registered before theme views load, then applied after they are read.
hexo.theme.on('processAfter', () => {
  hexo.theme.setView(
    'index.ejs',
    "<%- partial('_partial/archive', {expand: false, index: true}) %>\n"
  );
});
