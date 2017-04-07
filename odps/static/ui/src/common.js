/*
 * Copyright 1999-2017 Alibaba Group Holding Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * Common PyODPS Javascript Module
 */

require(['jupyter-js-widgets'], function (_) {}, function () {
    require.config({
        paths: {
            'jupyter-js-widgets': 'nbextensions/widgets/widgets/js/widget'
        }
    })
});

var pyodps_init_time = new Date();

define('pyodps/common', ['jquery', 'base/js/namespace', 'jupyter-js-widgets'], function ($, Jupyter) {
    "use strict";

    var entityMap = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': '&quot;',
        "'": '&#39;',
        "/": '&#x2F;'
    };

    function escape_html(string) {
        return String(string).replace(/[&<>"'\/]/g, function (s) {
            return entityMap[s];
        });
    }

    var view_prompts = {};

    var register_css = function (url) {
        var nbext_path = require.toUrl('nbextensions');
        var user_base_path = nbext_path.substr(0, nbext_path.indexOf('nbextensions'));
        require.config({
            paths: {
                pyodps: user_base_path + 'nbextensions/pyodps'
            }
        });

        if ($('style[data-pyodps-styles="' + url + '"]').length > 0) return;
        var url_parts = require.toUrl(url).split('?', 1);
        var css_url = url_parts[0] + '.css';
        if (url_parts.length > 1) css_url += '?' + url_parts[1];
        $('head').append('<link type="text/css" rel="stylesheet" href="' + css_url + '" data-pyodps-styles="' + url + '" />');
    };

    /**
     * Install a hook for a widget on termination of cell execution
     * @param _widget The widget object
     * @param func The function to be hooked
     */
    var call_on_executed = function (_widget, func) {
        var view_name = _widget.model.get('_view_name');
        var cell_obj = _widget.options.cell;
        if (undefined === view_prompts[view_name]) {
            view_prompts[view_name] = {
                prompt_num: -1,
                func: func,
            };

            var notifier = function () {
                // only display notifications when the cell stops running.
                if ($(cell_obj.element).hasClass('running'))
                    window.setTimeout(notifier, 100);
                else {
                    // ensure that notifications for a cell appear only once.
                    var prompt_num = cell_obj.input_prompt_number;
                    if (prompt_num !== view_prompts[view_name].prompt_num) {
                        view_prompts[view_name].prompt_num = prompt_num;
                        view_prompts[view_name].func.apply(_widget);
                    }
                    // make sure that notification will reoccur when rerun this cell.
                    view_prompts[view_name] = undefined;
                }
            };
            window.setTimeout(notifier, 100);
        } else {
            view_prompts[view_name].func = func
        }
    };
    $([Jupyter.events]).on('kernel_restarting.Kernel', function () {
        view_prompts = {};
    });

    register_css('nbextensions/pyodps/styles');

    return {
        init_time: pyodps_init_time,
        register_css: register_css,
        call_on_executed: call_on_executed,
        escape_html: escape_html,
    }
});

define('nbextensions/pyodps/main', ['base/js/namespace'], function(Jupyter) {
    return {
        load_ipython_extension: function() {
            if (Jupyter.CodeCell.config_defaults) {
                Jupyter.CodeCell.config_defaults.highlight_modes['magic_sql'] = {'reg':[/^%%sql/]};
            } else {
                Jupyter.CodeCell.options_default.highlight_modes['magic_sql'] = {'reg':[/^%%sql/]};
            }
        }
    }
});
