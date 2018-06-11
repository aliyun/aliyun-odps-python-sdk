$(function() {
    var lang_selector = '<div class="rst-versions" data-toggle="rst-versions" role="note" aria-label="versions">\n' +
        '  <span class="rst-current-version" data-toggle="rst-current-version">\n' +
        '    <span class="fa fa-book"> PyODPS Docs</span><span class="fa fa-caret-down"></span>\n' +
        '  </span>\n' +
        '  <div class="rst-other-versions"><!-- Inserted RTD Footer -->\n' +
        '    <div class="injected">\n' +
        '      <dl id="other-versions-languages">\n' +
        '        <dt>Languages</dt>\n' +
        '      </dl>\n' +
        '    </div>\n' +
        '  </div>\n' +
        '</div>';

    $.ajax({
        url: DOCUMENTATION_OPTIONS.URL_ROOT + '../langs.json',
        success: function (data) {
            $('nav.wy-nav-side').append($(lang_selector));

            var curLang = '';
            $.each(data.languages, function (idx, val) {
                if (location.pathname.indexOf('/' + val + '/') >= 0)
                    curLang = val;
            });
            if (!curLang) return;

            var reloadLangs = function() {
                var langStrs = '';
                $('#other-versions-languages dd').remove();
                $.each(data.languages, function (idx, val) {
                    var langStr = '<dd><a href="' + location.href.replace('/' + curLang + '/', '/' + val + '/')
                        + '">' + val + '</a></dd>';
                    if (curLang === val)
                        langStrs += '<strong>' + langStr + '</strong>';
                    else
                        langStrs += langStr;
                });
                $('#other-versions-languages').append($(langStrs));
            };
            reloadLangs();
            $('.reference').click(function() { window.setTimeout(reloadLangs, 100); });
            $('.headerlink').click(function() { window.setTimeout(reloadLangs, 100); });
        },
        dataType: 'json'
    });
});
