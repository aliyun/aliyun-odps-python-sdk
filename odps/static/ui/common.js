/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
 * Common PyODPS Javascript Module
 */
if (!require.defined('pyodps')) {

    define('pyodps', ['jquery', 'base/js/namespace'], function ($, IPython) {
        var load_counter = 0;
        var widget_num = parseInt('##WIDGET_NUM##');
        var last_load_time = new Date();
        var load_secs = parseInt('##MAX_SCRIPT_LOAD_SEC##');
        var loaded_widgets = {};

        var view_prompts = {};

        var loaded = function () {
            load_counter++;
            if (load_counter == widget_num && (new Date() - last_load_time) / 1000.0 <= load_secs - 0.5) {
                IPython.notebook.kernel.interrupt();
            }
        };

        var reset_counter = function (_widget_num) {
            load_counter = 0;
            widget_num = _widget_num;
            last_load_time = new Date();
        };

        var register_css = function (css_text) {
            if ($('style[data-pyodps-styles="_"]').length > 0) return;
            $('head').append('<style type="text/css" data-pyodps-styles="_">' + decodeURIComponent(css_text)
                + '</style>');
        };

        var define_widget = function (name, mods, func) {
            if (undefined !== loaded_widgets[name]) {
                loaded(); return;
            }
            loaded_widgets[name] = 'loaded';

            require(['jupyter-js-widgets'], function (_) {
                // ipywidgets>=5: use new namespace
                var new_mods = $.map(mods, function (v, _) {
                    if (v == 'widgets') return 'jupyter-js-widgets';
                    else return v;
                });
                define(name, new_mods, func);
                loaded();
            }, function () {
                // ipywidgets<=4: use old namespace
                var new_mods = $.map(mods, function (v, _) {
                    if (v == 'widgets') return 'nbextensions/widgets/widgets/js/widget';
                    else return v;
                });
                define(name, new_mods, func);
                loaded();
            });
        };

        /**
         * Install a hook for a widget on termination of cell execution
         * @param _widget The widget object
         * @param func The function to be hooked
         */
        var call_on_executed = function (_widget, func) {
            var cell_element = $(_widget.$el.closest('.cell'));
            var view_name = _widget.model.get('_view_name');
            if (undefined == view_prompts[view_name]) {
                view_prompts[view_name] = -1;
            }

            var notifier = function () {
                // only display notifications when the cell stops running.
                if (cell_element.hasClass('running'))
                    window.setTimeout(notifier, 100);
                else {
                    // ensure that notifications for a cell appear only once.
                    var prompt_num = cell_element.data('cell').input_prompt_number;
                    if (prompt_num !== view_prompts[view_name]) {
                        view_prompts[view_name] = prompt_num;
                        func.apply(_widget);
                    }
                }
            };
            window.setTimeout(notifier, 100);
        };
        $([IPython.events]).on('kernel_restarting.Kernel', function () {
            view_prompts = {};
        });

        return {
            reset_counter: reset_counter,
            register_css: register_css,
            define_widget: define_widget,
            call_on_executed: call_on_executed
        }
    });
}

require(['pyodps'], function(pyodps) {
    pyodps.reset_counter(parseInt('##WIDGET_NUM##'));
});