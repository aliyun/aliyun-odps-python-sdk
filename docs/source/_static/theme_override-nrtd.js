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
        '      <dl id="other-versions-versions">\n' +
        '        <dt>Versions</dt>\n' +
        '      </dl>\n' +
        '    </div>\n' +
        '  </div>\n' +
        '</div>';

    var render_links = function (data) {
        $('nav.wy-nav-side').append($(lang_selector));

        var curLang = '', curVersion = '';

        var languages = data.languages;
        if (languages === undefined) {
            languages = [];
            $('#other-versions-languages').hide();
        }

        var versions = data.versions;
        if (versions === undefined) {
            versions = [];
            $('#other-versions-versions').hide();
        }

        $.each(languages, function (idx, val) {
            if (location.pathname.indexOf('/' + val + '/') >= 0)
                curLang = val;
        });

        $.each(versions, function (idx, val) {
            if (location.pathname.indexOf('/' + val + '/') >= 0)
                curVersion = val;
        });

        var reloadLinks = function() {
            var langStrs = '', versionStrs = '';
            if (curLang !== '') {
                $('#other-versions-languages dd').remove();
                $.each(languages, function (idx, val) {
                    var langStr = '<dd><a href="' + location.href.replace('/' + curLang + '/', '/' + val + '/')
                        + '">' + val + '</a></dd>';
                    if (curLang === val)
                        langStrs += '<strong>' + langStr + '</strong>';
                    else
                        langStrs += langStr;
                });
                $('#other-versions-languages').append($(langStrs));
            }
            if (curVersion !== '') {
                $('#other-versions-versions dd').remove();
                $.each(versions, function (idx, val) {
                    var versionStr = '<dd><a href="' + location.href.replace('/' + curVersion + '/', '/' + val + '/')
                        + '">' + val + '</a></dd>';
                    if (curVersion === val)
                        versionStrs += '<strong>' + versionStr + '</strong>';
                    else
                        versionStrs += versionStr;
                });
                $('#other-versions-versions').append($(versionStrs));
            }
        };
        reloadLinks();
        $('.reference').click(function() { window.setTimeout(reloadLinks, 100); });
        $('.headerlink').click(function() { window.setTimeout(reloadLinks, 100); });
    };

    $.ajax({
        url: DOCUMENTATION_OPTIONS.URL_ROOT + '../../versions.json',
        success: render_links,
        error: function() {
            $.ajax({
                url: DOCUMENTATION_OPTIONS.URL_ROOT + '../versions.json',
                success: render_links,
                dataType: 'json'
            });
        },
        dataType: 'json'
    });
});
