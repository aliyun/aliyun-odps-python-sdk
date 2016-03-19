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
 * Cell customization for PAI
 */

require(["nbextensions/widgets/widgets/js/widget", "nbextensions/widgets/widgets/js/manager", "jquery"], function(widget, manager, $) {
    var retry_html = '<a class="retry-btn" title="Retry"><i class="fa fa-repeat" /></a>';

    var PAIRetryButton = widget.DOMWidgetView.extend({
        initialize: function(parameters) {
            var that = this;
            that.listenTo(that.model, 'msg:custom', that._handle_route_msg, that);
        },
        render: function() {
            var that = this;
            that.setElement($('<div></div>'))
        },
        _handle_route_msg: function(msg) {
            var that = this;
            var cell_element = that.$el.closest('.cell');
            var cell = cell_element.data('cell');
            var btn_element = $(retry_html);
            btn_element.click(function (e) {
                var old_text = cell.get_text();
                cell.set_text('%retry\n\n' + old_text);
                cell.execute();
                cell.set_text(old_text);
                e.stopPropagation();
            });
            var setter = function() {
                var prompt = cell_element.find('.input_prompt');
                if (cell_element.find('.retry-btn').length == 0) {
                    // only display notifications when the cell stops running.
                    if (cell_element.hasClass('running'))
                        window.setTimeout(setter, 100);
                    else
                        prompt.append(btn_element);
                }
            };
            window.setTimeout(setter, 100);
        }
    });

    manager.WidgetManager.register_widget_view('PAIRetryButton', PAIRetryButton);
    if ('undefined' !== typeof pyodps && pyodps.loaded) {
        pyodps.loaded();
    }
});