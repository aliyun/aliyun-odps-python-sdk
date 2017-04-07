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
 * InstancesProgress front end
 */
Date.prototype.format = function (fmt) {
    var o = {
        "M+": this.getMonth() + 1,
        "d+": this.getDate(),
        "h+": this.getHours(),
        "m+": this.getMinutes(),
        "s+": this.getSeconds(),
        "q+": Math.floor((this.getMonth() + 3) / 3),
        "S": this.getMilliseconds()
    };
    if (/(y+)/.test(fmt)) fmt = fmt.replace(RegExp.$1, (this.getFullYear() + "").substr(4 - RegExp.$1.length));
    for (var k in o)
        if (new RegExp("(" + k + ")").test(fmt)) fmt = fmt.replace(RegExp.$1, (RegExp.$1.length == 1) ? (o[k]) : (("00" + o[k]).substr(("" + o[k]).length)));
    return fmt;
};

define('pyodps/progress', ["jquery", "pyodps/common", "jupyter-js-widgets"], function ($, _, widget) {
    "use strict";

    var dialog_html = '<div class="modal fade" id="pyodps-progress-viewer" role="dialog">'.trim() +
        ' <div class="modal-dialog modal-sm">'.trim() +
        '  <div class="modal-content">'.trim() +
        '   <div class="modal-header">'.trim() +
        '    <button type="button" class="close pull-right" data-dismiss="modal">&times;</button>'.trim() +
        '    <span class="generated-time pull-right"></span>'.trim() +
        '    <h4 class="modal-title">Modal Header</h4>'.trim() +
        '   </div>'.trim() +
        '   <div class="modal-body">'.trim() +
        '    <div class="instances-holder"></div>'.trim() +
        '   </div>'.trim() +
        '   <div class="modal-footer">'.trim() +
        '    <span class="pull-left progress-legend-panel">'.trim() +
        '     <span class="progress-bar progress-legend"></span><span>Finished</span>'.trim() +
        '     <span class="progress-bar progress-bar-warning progress-bar-striped progress-legend"></span><span>Submitted</span>'.trim() +
        '     <span class="progress progress-legend"></span><span>Waiting</span>'.trim() +
        '    </span>'.trim() +
        '    <button type="button" class="btn btn-default pull-right" data-dismiss="modal">Close</button>'.trim() +
        '   </div>'.trim() +
        '  </div>'.trim() +
        ' </div>'.trim() +
        '</div>';

    var instance_html = '<div class="panel-group">' +
        ' <div class="panel panel-default">' +
        '  <div class="panel-heading">' +
        '   <h4 class="panel-title">' +
        '    <a class="instance_id" data-toggle="collapse" href="#inst_panel_{INSTANCE_ID}"></a> &nbsp; ' +
        '   </h4>' +
        '  </div>' +
        '  <div id="inst_panel_{INSTANCE_ID}" class="panel-collapse in">' +
        '   <ul class="list-group tasks">' +
        '   </ul>' +
        '   <div class="panel-footer"><a class="logview" href="javascript: void(0)">Logview</a></div>' +
        '  </div>' +
        ' </div>' +
        '</div>';

    var task_html = '<li class="list-group-item">' +
        ' <div class="task-name"></div>' +
        ' <div class="stages"></div>' +
        '</li>';

    var stage_html = '<div class="stage">' +
        ' <div class="stage-name"></div>' +
        ' <div class="stage-progress">' +
        '  <div class="progress">' +
        '   <div class="terminated progress-bar" role="progressbar"></div>' +
        '   <div class="running progress-bar progress-bar-warning progress-bar-striped active" role="progressbar"></div>' +
        '  </div>' +
        ' </div>' +
        '</div>';

    var modal;

    var InstancesProgress = widget.DOMWidgetView.extend({
        initialize: function (attributes, options) {
            var that = this;
            InstancesProgress.__super__.initialize.apply(that, arguments);

            that.groupMsgs = {};
            that.groupOrder = []; // Order of groups by time of insertion
            that.instRecords = {};
            that.listenTo(that.model, 'msg:custom', that._handle_route_msg, that);
            that.model.on('change:prefix', that._prefix_changed, that);
            that.model.on('change:suffix', that._suffix_changed, that);
        },

        render: function () {
            var that = this;
            // Render the view.
            modal = $('#pyodps-progress-viewer');
            if (modal.length == 0) {
                modal = $(dialog_html);
                $('body').append(modal);
            }

            that.prefixElement = $('<span style="padding-right: 5px"></span>');
            that.groupsElement = $('<span></span>');
            that.suffixElement = $('<span></span>');
            var rootElement = $('<div></div>')
                .append(that.prefixElement)
                .append(that.groupsElement)
                .append(that.suffixElement);

            try {
                that.$el.append(rootElement);
            } catch(e) {
                that.setElement(rootElement);
            }
        },

        update: function (options) {
            var that = this;
            if (options === undefined || options.updated_view != that) {
                that.prefixElement.text(that.model.get('prefix'));
                that.suffixElement.text(that.model.get('suffix'));
                that._build_link();
            }
            return InstancesProgress.__super__.update.apply(that);
        },

        _prefix_changed: function() {
            var that = this;
            that.prefixElement.text(that.model.get('prefix'));
        },

        _suffix_changed: function() {
            var that = this;
            that.suffixElement.text(that.model.get('suffix'));
        },

        _build_link: function () {
            var widget = this;
            widget.groupsElement.empty();
            $.each(widget.groupOrder, function (id, item) {
                var obj = $('<a class="pyodps-progress-launcher" href="javascript: void(0)"></a>')
                    .data('group', item).text(widget.groupMsgs[item].name);

                obj.click(function (e) {
                    var that = this;
                    var group = $(that).data('group');
                    widget._build_modal(group);
                    widget._show_modal();
                    e.stopPropagation();
                });
                widget.groupsElement.append(obj);
            });
        },

        _build_modal: function (group_key) {
            var widget = this;
            var group = widget.groupMsgs[group_key];
            modal.data('group_key', group_key);
            $('.modal-title', modal).text('Progress of "' + group.name + '"');
            $('.generated-time', modal).text('Generate time: ' + new Date(group.gen_time).format('yyyy-MM-dd hh:mm:ss'));

            var instance_holder = $('.instances-holder', modal);
            instance_holder.empty();

            if (!group.instances) return;
            $.each(group.instances, function (idx, inst_json) {
                if (undefined === widget.instRecords[inst_json.id]) {
                    widget.instRecords[inst_json.id] = {
                        startDate: new Date().valueOf()
                    };
                }
                if (inst_json.status == 'Terminated') {
                    var inst_rec = widget.instRecords[inst_json.id];
                    if (undefined === inst_rec.endDate) {
                        inst_rec.endDate = new Date().valueOf();
                    }
                    // check if the instance returns instantly
                    if (inst_rec.endDate - inst_rec.startDate < 1000 * 5) {
                        var is_success = true;
                        $.each(inst_json.tasks, function (idx, task_json) {
                            if (task_json.status != 'SUCCESS') is_success = false;
                        });
                        // if is successful, we do not render
                        if (is_success) return;
                    }
                }
                var inst_obj = $(instance_html.replace(/\{INSTANCE_ID\}/g, inst_json.id));
                var tasks_holder = $('.tasks', inst_obj);
                $('.instance_id', inst_obj).text('Instance ' + idx + ': ' + inst_json.status);
                $('.logview', inst_obj).attr('href', inst_json.logview).attr('target', '_blank');

                if (!inst_json.tasks) return;
                $.each(inst_json.tasks, function (idx, task_json) {
                    var task_obj = $(task_html);
                    $('.task-name', task_obj).text(task_json.name + ': ' + task_json.status);

                    var stages_holder = $('.stages', task_obj);
                    if (!task_json.stages) {
                        stages_holder.text('No stage information available.')
                    } else {
                        $.each(task_json.stages, function (idx, stage_json) {
                            var stage_obj = $(stage_html);
                            var stage_name_tipper = function (el) {
                                var this_obj = $(el);
                                if (el.scrollWidth > el.offsetWidth) {
                                    this_obj.attr('title', this_obj.text());
                                } else {
                                    this_obj.removeAttr('title');
                                }
                            };
                            $('.stage-name', stage_obj)
                                .text(stage_json.name)
                                .resize(function() { stage_name_tipper(this); });
                            window.setTimeout(function() {
                                stage_name_tipper($('.stage-name', stage_obj).last()[0]);
                            }, 100);

                            if (stage_json.total_workers == 0)
                                stage_json.total_workers = 1;
                            var terminated_perc = (stage_json.terminated_workers * 100 / (stage_json.total_workers + stage_json.backup_workers)).toFixed(2) + '%';
                            $('.terminated', stage_obj).width(terminated_perc).text(terminated_perc);
                            var running_perc = (stage_json.running_workers * 100 / (stage_json.total_workers + stage_json.backup_workers)).toFixed(2) + '%';
                            $('.running', stage_obj).width(running_perc).text(running_perc);
                            stages_holder.append(stage_obj);

                        });
                    }
                    tasks_holder.append(task_obj);
                });
                instance_holder.append(inst_obj);
            });
        },

        _show_modal: function () {
            modal.modal();
        },

        _handle_route_msg: function (msg) {
            var that = this;
            if (msg) {
                // message format: '{"action": "action", content: ["content1", "content2"]}'
                var msg_obj = $.parseJSON(msg);
                var action = msg_obj.action;
                var content = [];
                if (msg_obj.content)
                    content = msg_obj.content;
                if (action == 'update') {
                    $.each(content, function (idx, group_json) {
                        group_json = $.parseJSON(group_json);
                        if (!that.groupMsgs[group_json.key])
                            that.groupOrder.push(group_json.key);
                        that.groupMsgs[group_json.key] = group_json;

                        if (modal.data('group_key') == group_json.key) {
                            that._build_modal(group_json.key);
                        }
                    });
                } else if (msg_obj.action == 'delete') {
                    $.each(content, function (idx, group_key) {
                        if (!that.groupMsgs[group_key])
                            return;
                        var i = that.groupOrder.indexOf(group_key);
                        if (i >= 0) that.groupOrder.splice(i, 1);
                    });
                } else if (msg_obj.action == 'clear') {
                    that.groupMsgs = {};
                    that.groupOrder = [];
                    that._build_link();
                }
            }
            that.update();
        },
    });

    return {
        InstancesProgress: InstancesProgress
    };
});
