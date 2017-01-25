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
 * Cell customization for PyODPS ML
 */
define('pyodps/ml-retry', ["jquery", "pyodps/common", "jupyter-js-widgets"], function ($, common, widget) {
    "use strict"

    var retry_html = '<a class="retry-btn" title="Retry"><i class="fa fa-repeat" /></a>';

    var MLRetryButton = widget.DOMWidgetView.extend({
        initialize: function (parameters) {
            var that = this;
            that.model.on('change:msg', that._msg_changed, that);
        },
        render: function() {
            $(this.$el).closest('.widget-area').find('div').hide();
        },
        update: function () {
            $(this.$el).closest('.widget-area').find('div').hide();
        },
        _msg_changed: function () {
            var that = this;
            var cell_element = $(that.$el.closest('.cell'));
            var cell = cell_element.data('cell');
            var btn_element = $(retry_html);
            btn_element.click(function (e) {
                var old_text = cell.get_text();
                cell.set_text('%retry\n\n' + old_text);
                cell.execute();
                cell.set_text(old_text);
                e.stopPropagation();
            });
            common.call_on_executed(that, function () {
                var prompt = cell_element.find('.input_prompt');
                prompt.append(btn_element);
            });
            that.remove();
        }
    });


    return {
        MLRetryButton: MLRetryButton
    };
});
