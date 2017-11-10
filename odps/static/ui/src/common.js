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

require(['base/js/utils'], function(jupyter_utils) {
    jupyter_utils.load_extension('jupyter-js-widgets/extension').then(function () {
        require(['@jupyter-widgets/base'], function (_) {
            window.console.log('ipywidgets later than 7.0.0 detected.');
        }, function () {
            require.config({
                map: {
                    '*': {
                        '@jupyter-widgets/base': 'nbextensions/jupyter-js-widgets/extension',
                        '@jupyter-widgets/controls': 'nbextensions/jupyter-js-widgets/extension',
                        '@jupyter-widgets/output': 'nbextensions/jupyter-js-widgets/extension',
                    }
                }
            });
            window.console.log('ipywidgets between 4.0.0 and 7.0.0 detected.');
        });
    }, function () {
        require.config({
            map: {
                '*': {
                    '@jupyter-widgets/base': 'nbextensions/widgets/widgets/js/widget',
                    '@jupyter-widgets/controls': 'nbextensions/widgets/widgets/js/widget',
                    '@jupyter-widgets/output': 'nbextensions/widgets/widgets/js/widget',
                }
            }
        });
        window.console.log('ipywidgets prior than 4.0.0 detected.');
    });
});

var pyodps_init_time = new Date();

define('pyodps/common', ['jquery', 'base/js/namespace', '@jupyter-widgets/base'], function ($, Jupyter) {
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

    var _instant_widgets = {};

    var _get_widget_cell = function (_widget) {
        if (_widget.options.cell)
            return _widget.options.cell;
        return $(_widget.options.output.element).closest('.cell').data('cell');
    };

    var _auto_close_widgets = function () {
        var cells = Jupyter.notebook.get_cells();
        for (var i in cells) {
            var output_area = cells[i].output_area;
            var new_outputs = [];
            for (var j in output_area.outputs) {
                var o = output_area.outputs[j];
                if (undefined !== o.data) {
                    var o_data = o.data['application/vnd.jupyter.widget-view+json'];
                    if (undefined !== o_data) {
                        if (_instant_widgets[o_data.model_id]) {
                            delete _instant_widgets[o_data.model_id];
                            continue;
                        }
                    }
                }
                new_outputs.push(o);
            }
            output_area.outputs = new_outputs;
        }

        var _hide_empty_widgets = function() {
            for (var i in cells) {
                var cell_el = $(cells[i].element);
                if (!cell_el.hasClass('running')) {
                    cell_el.find('.output_area:has(.jupyter-widgets-view)').each(function () {
                        if ($(this).text() === '') $(this).remove();
                    });
                }
            }
        };
        _hide_empty_widgets();
        window.setTimeout(_hide_empty_widgets, 100);
    };

    $([Jupyter.events]).on('kernel_idle.Kernel', _auto_close_widgets);
    $([Jupyter.events]).on('kernel_ready.Kernel', _auto_close_widgets);

    var add_transient_widgets = function (comm_id) {
        _instant_widgets[comm_id] = true;
    };

    var close_widget = function (widget) {
        var widget_output_area = _get_widget_cell(widget).output_area;
        var widget_area = $(widget.$el).closest('.output_area:has(.jupyter-widgets-view)');
        if (widget_area.length > 0)
            widget_area.hide();
        else {
            $(widget.$el).closest('.widget-area').children('div').map(function(idx, o) {
                if ($(o).text() === '') $(o).hide();
            });
            widget.remove();
        }

        var new_outputs = [];
        for (var i in widget_output_area.outputs) {
            var o = widget_output_area.outputs[i];
            if (o.data === undefined) {
                new_outputs.push(o);
            } else {
                var o_data = o.data['application/vnd.jupyter.widget-view+json'];
                if (undefined === o_data || o_data.model_id !== widget.model.comm.comm_id)
                    new_outputs.push(o);
            }
        }
        widget_output_area.outputs = new_outputs;
    };

    /**
     * Install a hook for a widget on termination of cell execution
     * @param _widget The widget object
     * @param func The function to be hooked
     */
    var call_on_executed = function (_widget, func) {
        var view_name = _widget.model.get('_view_name');
        var cell_obj = _get_widget_cell(_widget);
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
        add_transient_widgets: add_transient_widgets,
        close_widget: close_widget,
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
