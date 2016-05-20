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
 * HTML 5 notification component
 */

require(['pyodps'], function(pyodps) {
    pyodps.define_widget('pyodps/html-notify', ["jquery", "widgets"], function ($, widget) {

        if (Notification && Notification.permission !== "granted") {
            Notification.requestPermission(function (status) {
                if (Notification.permission !== status) {
                    Notification.permission = status;
                }
            });
        }

        var IMAGE_DATA = 'data:image/png;base64,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';

        var HTMLNotifier = widget.WidgetView.extend({
            initialize: function (parameters) {
                var that = this;
                $(that.$el).closest('.widget-area').find('div').hide();
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
                if (Notification) {
                    var notify_config = $.parseJSON(that.model.get('msg'));
                    notify_config.icon = IMAGE_DATA;

                    pyodps.call_on_executed(that, function () {
                        new Notification('PyODPS', notify_config);
                    });
                }
                that.remove();
            }
        });

        return {
            HTMLNotifier: HTMLNotifier
        };
    });
});
