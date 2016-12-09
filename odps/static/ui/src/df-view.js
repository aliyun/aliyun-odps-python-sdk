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
 * PyODPS DataFrame Interactive Module
 */
String.prototype.rsplit = function(sep, maxsplit) {
    var split = this.split(sep);
    return maxsplit ? [ split.slice(0, -maxsplit).join(sep) ].concat(split.slice(-maxsplit)) : split;
};

define('pyodps/df-view', ['jquery', 'base/js/namespace', 'jupyter-js-widgets', 'pyodps/common', 'echarts.min', 'westeros.min'],
       function($, IPython, widgets, pyodps, echarts) {
    "use strict";
    var viewer_counter = 1;
    var df_view_pager_sizes = [1, 5, 1];
    var df_view_html = '<div class="row df-view output-suppressor">' +
        '<ul class="df-function-nav nav nav-pills pills-small">' +
        ' <li class="active"><a class="btn-df-table" data-toggle="tab" href="#show-table-{counter}"><i class="fa fa-table" /></a></li>' +
        ' <li><a class="btn-bar-chart" data-toggle="tab" href="#show-bar-chart-{counter}"><i class="fa fa-bar-chart" /></a></li>' +
        ' <li><a class="btn-pie-chart" data-toggle="tab" href="#show-pie-chart-{counter}"><i class="fa fa-pie-chart" /></a></li>' +
        ' <li><a class="btn-line-chart" data-toggle="tab" href="#show-line-chart-{counter}"><i class="fa fa-line-chart" /></a></li>' +
        ' <li><a class="btn-scatter-chart" data-toggle="tab" href="#show-scatter-chart-{counter}"><i class="cf cf-scatter-chart" /></a></li>' +
        ' <li class="pull-right">' +
        '  <div class="btn-group df-graph-toolbar" role="group">' +
        '   <button type="button" class="btn btn-default btn-download-graph">' +
        '    <i class="fa fa-download" />' +
        '   </button>' +
        '   <a type="button" class="btn btn-default btn-config-chart" data-toggle="button" aria-pressed="false" autocomplete="off">' +
        '    <i class="fa fa-gear" />&nbsp;<i class="btn-config-caret fa fa-caret-down" />' +
        '   </a>' +
        '  </div>' +
        ' </li>' +
        '</ul>' +
        '' +
        '</div>' +
        '<div class="row">' +
        '<div class="tab-content">' +
        ' <div id="show-table-{counter}" class="tab-pane show-table fade in active">' +
        '  <div class="show-row row">' +
        '   <div class="show-body">' +
        '    <div class="show-table-body">' +
        '     <i class="fa fa-spinner fa-spin fa-2x" />' +
        '    </div>' +
        '    <ul class="df-table-pagination pagination pagination-sm" style="display: none">' +
        '     <li class="df-pager-first df-pager-top">' +
        '      <a href="#" aria-label="Previous">' +
        '       <span aria-hidden="true">&laquo;</span>' +
        '      </a>' +
        '     </li>' +
        '     <li class="df-pager-last df-pager-top">' +
        '      <a href="#" aria-label="Next">' +
        '       <span aria-hidden="true">&raquo;</span>' +
        '      </a>' +
        '     </li>' +
        '    </ul>' +
        '   </div>' +
        '  </div>' +
        ' </div>' +
        ' <div id="show-bar-chart-{counter}" class="tab-pane fade show-pane show-bar-chart" data-binding="bar">' +
        '  <div class="show-row row">' +
        '   <div class="show-body"><i class="fa fa-spinner fa-spin fa-2x" /></div>' +
        '   <div class="show-options" style="display: none">' +
        '    Groups: <select class="groups-selector field-selector" data-placeholder="Select group columns" style="width: 180px" multiple="multiple"></select>' +
        '    Keys: <select class="keys-selector field-selector" data-placeholder="Select key columns" style="width: 180px" multiple="multiple"></select>' +
        '    Values: <select class="values-selector field-selector df-with-agg" data-placeholder="Select value columns" style="width: 180px" multiple="multiple"></select>' +
        '    <div class="row pull-right" style="margin: 4px 0">' +
        '     <button type="button" class="btn btn-primary btn-refresh-popup">Refresh</button>' +
        '     <button type="button" class="btn btn-default btn-cancel-popup">Cancel</button>' +
        '    </div>' +
        '   </div>' +
        '  </div>' +
        ' </div>' +
        ' <div id="show-pie-chart-{counter}" class="tab-pane fade show-pane show-pie-chart" data-binding="pie">' +
        '  <div class="show-row row">' +
        '   <div class="show-body"><i class="fa fa-spinner fa-spin fa-2x" /></div>' +
        '   <div class="show-options" style="display: none">' +
        '    Keys: <select class="keys-selector field-selector" data-placeholder="Select key columns" style="width: 180px" multiple="multiple"></select>' +
        '    Values: <select class="values-selector field-selector df-with-agg" data-placeholder="Select value columns" style="width: 180px" multiple="multiple"></select>' +
        '    <div class="row pull-right" style="margin: 4px 0">' +
        '     <button type="button" class="btn btn-primary btn-refresh-popup">Refresh</button>' +
        '     <button type="button" class="btn btn-default btn-cancel-popup">Cancel</button>' +
        '    </div>' +
        '   </div>' +
        '  </div>' +
        ' </div>' +
        ' <div id="show-line-chart-{counter}" class="tab-pane fade show-pane show-line-chart" data-binding="line">' +
        '  <div class="show-row row">' +
        '   <div class="show-body"><i class="fa fa-spinner fa-spin fa-2x" /></div>' +
        '   <div class="show-options" style="display: none">' +
        '    Groups: <select class="groups-selector field-selector" data-placeholder="Select group columns" style="width: 180px" multiple="multiple"></select>' +
        '    X Axis: <select class="keys-selector field-selector df-select-with-id" data-placeholder="Select key columns" style="width: 180px"></select>' +
        '    Y Axis: <select class="values-selector field-selector" data-placeholder="Select value columns" style="width: 180px" multiple="multiple"></select>' +
        '    <div class="row pull-right" style="margin: 4px 0">' +
        '     <button type="button" class="btn btn-primary btn-refresh-popup">Refresh</button>' +
        '     <button type="button" class="btn btn-default btn-cancel-popup">Cancel</button>' +
        '    </div>' +
        '   </div>' +
        '  </div>' +
        ' </div>' +
        ' <div id="show-scatter-chart-{counter}" class="tab-pane fade show-pane show-scatter-chart" data-binding="scatter">' +
        '  <div class="show-row row">' +
        '   <div class="show-body"><i class="fa fa-spinner fa-spin fa-2x" /></div>' +
        '   <div class="show-options" style="display: none">' +
        '    Groups: <select class="groups-selector field-selector" data-placeholder="Select group columns" style="width: 180px" multiple="multiple"></select>' +
        '    X Axis: <select class="keys-selector field-selector df-select-with-id" data-placeholder="Select key columns" style="width: 180px"></select>' +
        '    Y Axis: <select class="values-selector field-selector" data-placeholder="Select value columns" style="width: 180px" multiple="multiple"></select>' +
        '    <div class="row pull-right" style="margin: 4px 0">' +
        '     <button type="button" class="btn btn-primary btn-refresh-popup">Refresh</button>' +
        '     <button type="button" class="btn btn-default btn-cancel-popup">Cancel</button>' +
        '    </div>' +
        '   </div>' +
        '  </div>' +
        ' </div>' +
        '</div>' +
        '</div>';
    var df_view_pager_html = '<li class="df-pager df-pager-{page}" data-page="{page}"><a href="#">{page}</a></li>';
    var df_view_elip_pager_html = '<li class="df-elip-pager disabled"><a href="#">...</a></li>';
    var df_view_agg_selector_link_html = '<a class="df-agg-selector">{selectors}</a>';
    var df_view_agg_func_selector = '<div>' +
        ' <select class="df-agg-func-selector" data-placeholder="Select aggregators" style="width: 180px" multiple="multiple">' +
        '  <option value="count">count</option>' +
        '  <option value="nunique">nunique</option>' +
        '  <option value="min">min</option>' +
        '  <option value="max">max</option>' +
        '  <option value="sum">sum</option>' +
        '  <option value="mean">mean</option>' +
        '  <option value="median">median</option>' +
        '  <option value="var">var</option>' +
        '  <option value="std">std</option>' +
        ' </select>' +
        ' <div class="row pull-right" style="margin: 4px 0">' +
        '  <button type="button" class="btn btn-default btn-cancel-agg-popup">Cancel</button>' +
        ' </div>' +
        '</div>';

    pyodps.register_css('pyodps/chosen');
    pyodps.register_css('pyodps/fonts/custom-font');

    $([IPython.events]).on("kernel_busy.Kernel", function() {
        $('.cell').not(':has(.output-suppressor)').each(function () {
            $(this).find('div.output_wrapper').show();
        });
    });

    var DFView = widgets.DOMWidgetView.extend({
        render: function() {
            var that = this;
            that.listenTo(that.model, 'change:start_sign', that._actual_render);
            that.listenTo(that.model, 'change:error_sign', that._render_error);
            that.send({
                action: 'start_widget'
            });
        },

        _actual_render: function() {
            var that = this;
            var cell_obj = $(that.$el).closest('.cell');
            var cell_data = cell_obj.data();
            if (cell_data !== undefined) {
                var last_msg = cell_data.cell.last_msg_id;
                if (!last_msg) {
                    $(that.$el).closest('.widget-area').find('button.close').click();
                    return;
                }
            }
            cell_obj.find('.widget-area .close').click(function() {
                cell_obj.find('div.output_wrapper').show();
            });

            that._chart_bindings = {
                bar: {
                    button: '.btn-bar-chart',
                    panel: '.show-bar-chart',
                    config: '_bar_chart_conf',
                    agg: true
                },
                pie: {
                    button: '.btn-pie-chart',
                    panel: '.show-pie-chart',
                    config: '_pie_chart_conf',
                    agg: true
                },
                line: {
                    button: '.btn-line-chart',
                    panel: '.show-line-chart',
                    config: '_line_chart_conf',
                    agg: false
                },
                scatter: {
                    button: '.btn-scatter-chart',
                    panel: '.show-scatter-chart',
                    config: '_scatter_chart_conf',
                    agg: false
                },
            };

            that.listenTo(that.model, 'change:table_records', that._update_table_records);
            that.listenTo(that.model, 'change:bar_chart_records', that._update_bar_chart);
            that.listenTo(that.model, 'change:pie_chart_records', that._update_pie_chart);
            that.listenTo(that.model, 'change:line_chart_records', that._update_line_chart);
            that.listenTo(that.model, 'change:scatter_chart_records', that._update_scatter_chart);

            var df_view_obj = $(df_view_html.replace(/\{counter\}/g, viewer_counter));
            viewer_counter += 1;
            df_view_obj.on('click', '.btn-refresh-popup', function() {
                that._refresh_popover();
            });
            df_view_obj.on('click', '.btn-cancel-popup', function() {
                $(that.$el.find(".btn-config-chart")).click();
            });
            df_view_obj.on('click', '.btn-df-table', function() {
                that._clear_widget_spaces();

                var cfg_chart_btn = that.$el.find('.btn-config-chart');
                if (cfg_chart_btn.hasClass('active')) {
                    $(cfg_chart_btn).click();
                }
                that.$el.find('.df-graph-toolbar').hide();
            });
            df_view_obj.on('click', '.btn-bar-chart', function() {
                that._clear_widget_spaces();

                that.$el.find('.df-graph-toolbar').show();
                if (!that._bar_chart_conf) {
                    that._gen_default_agg_fields('bar');
                }
                that.send({
                    action: 'aggregate_graph',
                    target: 'bar_chart_records',
                    values: that._bar_chart_conf.values,
                    keys: that._bar_chart_conf.keys,
                    groups: that._bar_chart_conf.groups
                });
                that._bind_popover('bar');
                that._refresh_chart('bar');
            });
            df_view_obj.on('click', '.btn-pie-chart', function() {
                that._clear_widget_spaces();

                that.$el.find('.df-graph-toolbar').show();
                if (!that._pie_chart_conf) {
                    that._gen_default_agg_fields('pie', 'count');
                }
                that.send({
                    action: 'aggregate_graph',
                    target: 'pie_chart_records',
                    values: that._pie_chart_conf.values,
                    keys: that._pie_chart_conf.keys
                });
                that._bind_popover('pie');
                that._refresh_chart('pie');
            });
            df_view_obj.on('click', '.btn-line-chart', function() {
                that._clear_widget_spaces();

                that.$el.find('.df-graph-toolbar').show();
                if (!that._line_chart_conf) {
                    that._gen_default_agg_fields('line');
                }
                that.send({
                    action: 'aggregate_graph',
                    target: 'line_chart_records',
                    values: that._line_chart_conf.values,
                    keys: that._line_chart_conf.keys,
                    groups: that._line_chart_conf.groups
                });
                that._bind_popover('line');
                that._refresh_chart('line');
            });
            df_view_obj.on('click', '.btn-scatter-chart', function() {
                that._clear_widget_spaces();

                that.$el.find('.df-graph-toolbar').show();
                if (!that._scatter_chart_conf) {
                    that._gen_default_agg_fields('scatter');
                }
                that.send({
                    action: 'aggregate_graph',
                    target: 'scatter_chart_records',
                    values: that._scatter_chart_conf.values,
                    keys: that._scatter_chart_conf.keys,
                    groups: that._scatter_chart_conf.groups
                });
                that._bind_popover('scatter');
                that._refresh_chart('scatter');
            });
            df_view_obj.on('click', '.btn-config-chart', function() {
                if (!that.$el.find('.btn-config-chart').hasClass('active')) {
                    that._render_popover();
                    that._bind_popover(':visible');
                }
                if (that.$el.find('.btn-config-chart').hasClass('active')) {
                    that.$el.find(".btn-config-caret").removeClass("fa-caret-up").addClass("fa-caret-down");
                } else {
                    that.$el.find(".btn-config-caret").removeClass("fa-caret-down").addClass("fa-caret-up");
                }
            });
            df_view_obj.on('click', '.df-pager', function() {
                var page_id = parseInt($(this).data('page')) - 1;
                that.send({action: 'fetch_table', page: page_id});
            });
            df_view_obj.on('click', '.df-pager-first', function() {
                that.send({action: 'fetch_table', page: 0});
            });
            df_view_obj.on('click', '.df-pager-last', function() {
                var pagenation_obj = that.$el.find('.show-table .pagination');
                that.send({action: 'fetch_table', page: pagenation_obj.data('pages') - 1});
            });
            df_view_obj.on('click', '.btn-download-graph', function() {
                var chart_container = $(that.$el).find('.show-body:visible');
                var chart = chart_container.data('chart');
                if (!chart) return;
                var uri = chart.getDataURL({backgroundColor: '#ffffff'});
                var binding_name = chart_container.closest('.show-pane').data('binding');

                var downloadLink = document.createElement("a");
                downloadLink.href = uri;
                downloadLink.download = binding_name + '.png';

                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
            });

            var _resize_table_areas = function() {
                $('.show-table .show-body').each(function() {
                    $(this).find('.show-table-body').width($(this).closest('.cell').find('.inner_cell').width());
                });
            };

            $(window).click(function() {
                that.$el.find('.df-popover-agg').each(function(idx, e) {
                    that._destroy_popover(e);
                });
            }).resize(function() {
                $('.show-body').each(function() {
                    var chart = $(this).data('chart');
                    if (chart) chart.resize();
                });
                _resize_table_areas();
            });

            that.$el.append(df_view_obj);
            that.$el.find('.df-graph-toolbar').hide();
            that.send({action: 'fetch_table'});
            _resize_table_areas();
        },
        
        _render_error: function () {
            var that = this;
            var _error_renderer = function () {
                var $that_el = $(that.$el);
                var error_html = $that_el.closest('.cell').find('.output_subarea.output_error').html();
                if (!error_html) window.setTimeout(_error_renderer, 200);
                $that_el.find('.show-body:visible').html(error_html);
            };
            window.setTimeout(_error_renderer, 500);
        },

        _destroy_popover: function (dom) {
            var that = this;
            var pid = $(dom).attr('id');
            var selector_obj = $(that.$el.find('.df-agg-selector[aria-describedby=' + pid + ']'));
            $(dom).remove();
            selector_obj.popover('destroy');
            selector_obj.removeData('popover_id');
        },

        _switch_display: function(show_df) {
            var cell_obj = this.$el.closest('div.cell');
            if (show_df) {
                cell_obj.find('div.output_wrapper').hide();
                cell_obj.find('div.widget-area').show();
                cell_obj.find('div.widget-subarea').show();
            } else {
                cell_obj.find('div.output_wrapper').show();
                cell_obj.find('div.widget-area').hide();
                cell_obj.find('div.widget-subarea').hide();
            }
        },

        _bind_popover: function(selector) {
            var that = this, i;
            if (selector == ':visible') selector = '.show-pane:visible';
            else selector = that._chart_bindings[selector].panel;

            var opt_selector = that.$el.find(selector).find('.show-options');
            var binding = that._chart_bindings[that.$el.find(selector).data('binding')];
            var config = that[binding.config];

            opt_selector.find('.groups-selector option').attr('selected', null);
            for (i in config.groups)
                opt_selector.find('.groups-selector option[value=' + config.groups[i] + ']').attr('selected', true);

            var selected_keys = config.keys;
            if (!selected_keys) selected_keys = ['#ID'];
            opt_selector.find('.keys-selector option').attr('selected', null);
            for (i in selected_keys)
                opt_selector.find('.keys-selector option[value=' + selected_keys[i] + ']').attr('selected', true);

            var values_selector = opt_selector.find('.values-selector');
            values_selector.find('option').attr('selected', null);
            if (values_selector.hasClass('df-with-agg')) {
                for (i in config.values)
                    values_selector.find('option[value=' + i + ']').attr('selected', true);
            } else {
                for (i in config.values)
                    values_selector.find('option[value=' + config.values[i] + ']').attr('selected', true);
            }

            that._cur_pane_selector = selector;

            var chart_btn = that.$el.find('.btn-config-chart');
            if (chart_btn.hasClass('active') && that.$el.find(selector).data('binding') != that.$el.find('.show-pane:visible').data('binding'))
                $(chart_btn).click();
            if (opt_selector.length != 0) {
                $(chart_btn).popover({
                    html: true,
                    placement: 'bottom',
                    viewport: that.$el,
                    content: function() {
                        return that.$el.find(that._cur_pane_selector).find('.show-options').html();
                    }
                });
            }
        },

        _render_popover: function() {
            var that = this;
            var binding = that._chart_bindings[that.$el.find('.show-pane:visible').data('binding')];
            var config = that[binding.config];
            var render_agg_selector = function(e) {
                var choice = $(e).find('span').text(), i;
                var aggregators = config._values_pending[choice];
                if (aggregators) {
                    var selector_obj = $(df_view_agg_selector_link_html.replace(/\{selectors\}/g, aggregators.join(',')));
                    selector_obj.insertBefore($(e).find('.search-choice-close'));
                    selector_obj.click(function (e) {
                        e.stopPropagation();

                        if (selector_obj.data('popover_id')) {
                            $('#' + selector_obj.data('popover_id')).remove();
                            selector_obj.popover('destroy');
                            selector_obj.removeData('popover_id');
                        } else {
                            selector_obj.popover({
                                html: true,
                                placement: 'bottom',
                                trigger: 'manual',
                                viewport: that.$el,
                                content: function() {
                                    var agg_select_obj = $(df_view_agg_func_selector);
                                    var aggregators = config._values_pending[choice];
                                    for (i in aggregators) {
                                        agg_select_obj.find('select option[value=' + aggregators[i] + ']').attr('selected', true);
                                    }
                                    return $(agg_select_obj).html();
                                }
                            });
                            selector_obj.popover('show');
                            var popover_id = selector_obj.attr('aria-describedby');
                            selector_obj.data('popover_id', popover_id);

                            var popover_obj = $('#' + popover_id);
                            var popover_offset = popover_obj.offset();
                            popover_obj.remove();
                            popover_obj
                                .addClass('df-popover-agg')
                                .insertAfter(selector_obj.closest('.popover'))
                                .offset(popover_offset)
                                .click(function(e) { e.stopPropagation(); });
                            var agg_chosen_obj = popover_obj.find('.df-agg-func-selector').chosen();
                            agg_chosen_obj.change(function() {
                                config._values_pending[choice] = agg_chosen_obj.val();
                                selector_obj.text(config._values_pending[choice].join(','));
                            });
                            popover_obj.on('click', '.btn-cancel-agg-popup', function() {
                                that._destroy_popover(popover_obj);
                            });
                        }
                    }).mousedown(function (e) {
                        e.stopPropagation();
                    });
                }
            };

            $(that.$el.find(".popover .field-selector")).chosen({});

            config._values_pending = JSON.parse(JSON.stringify(config.values));

            var values_selector = $(that.$el.find('.popover .values-selector'));
            if (!values_selector.hasClass('df-with-agg'))
                return;
            var chosen_container = $(values_selector.data().chosen.container);
            var chosen_obj = $(that.$el.find(".popover .values-selector")).chosen();
            chosen_obj.change(function() {
                chosen_container.find('.search-choice').each(function(idx, e) {
                    var choice = $(e).find('span').text();
                    $(e).not(':has(.df-agg-selector)').each(function() {
                        config._values_pending[choice] = ['sum'];
                        render_agg_selector(e);
                    });
                });
                var choices_coll = {}, k;
                $.each(chosen_obj.val(), function(idx, v) {
                    choices_coll[v] = true;
                });
                for (k in config._values_pending)
                    if (!choices_coll[k]) delete config._values_pending[k];
            });
            chosen_container.find('.search-choice').each(function(idx, e) {
                render_agg_selector(e);
            });
        },

        _refresh_popover: function() {
            var that = this;
            var visible_pane = that.$el.find('.show-pane:visible');
            var binding = that._chart_bindings[visible_pane.data('binding')];
            var config = that[binding.config];
            var popover = that.$el.find('.popover');

            config.groups = $(popover).find('.groups-selector').chosen().val();

            var keys_selector = $(popover).find('.keys-selector');
            if (keys_selector.attr('multiple')) {
                config.keys = keys_selector.chosen().val();
            } else {
                var key_sel = keys_selector.chosen().val();
                if (key_sel == '#ID')
                    config.keys = [];
                else
                    config.keys = [key_sel];
            }

            var values_selector = $(popover).find('.values-selector');
            if (values_selector.hasClass('df-with-agg')) {
                config.values = JSON.parse(JSON.stringify(config._values_pending));
            } else {
                config.values = values_selector.chosen().val();
            }

            $(that.$el.find(binding.button)).click();
        },

        _gen_default_agg_fields: function(target, agg) {
            var that = this;
            var bindings = that._chart_bindings[target];
            if (!that._columns) return;
            var values = {};
            if (!agg) agg = 'sum';
            if (that._columns.length == 2) {
                that[bindings.config] = {
                    keys: [that._columns[0]],
                    values: [that._columns[1]],
                };
                values[that._columns[1]] = [agg];
            } else {
                that[bindings.config] = {
                    groups: [that._columns[0]],
                    keys: [that._columns[1]],
                    values: [that._columns[2]],
                };
                values[that._columns[2]] = [agg];
            }
            if (bindings.agg) that[bindings.config].values = values;
        },

        _clear_widget_spaces: function() {
            $('.widget-subarea').children('div').map(function(idx, o) {
                if ($(o).text() == '') $(o).remove();
            });
        },

        _update_table_records: function() {
            var that = this;
            var i, j;
            var records = that.model.get('table_records');
            var table_html = '<table class="table table-striped">';

            that._switch_display(true);
            that._clear_widget_spaces();

            that._columns = records.columns;
            var field_selector = that.$el.find(".field-selector");
            field_selector.empty();
            $(that.$el).find('.field-selector.df-select-with-id').append($('<option value="#ID">(Row ID)</option>'));
            for (i = 0; i < records.columns.length; i++) {
                var col_name = records.columns[i];
                field_selector.append($('<option value="' + col_name + '">' + col_name + '</option>'))
            }

            // write table headers
            table_html += '<thead><tr>';
            for (i = 0; i < records.columns.length; i++) {
                table_html += '<th>' + pyodps.escape_html(records.columns[i]) + '</th>';
            }
            table_html += '</tr></thead>';
            // write table body
            table_html += '<tbody>';
            for (i = 0; i < records.data.length; i++) {
                table_html += '<tr>';
                for (j = 0; j < records.data[i].length; j++) {
                    table_html += '<td>' + pyodps.escape_html(records.data[i][j]) + '</td>';
                }
                table_html += '</tr>';
            }
            table_html += '</tbody>';
            table_html += '</table>';
            that.$el.find('.show-table-body').html(table_html);

            // load pagination
            var page_str;
            var pagination_obj = that.$el.find('.show-table .pagination');
            if (!records.pages)
                pagination_obj.hide();
            else {
                var pages = records.pages;
                var lrange_right = df_view_pager_sizes[0];
                var mrange_left = records.page + 1 - Math.floor(df_view_pager_sizes[1] / 2);
                var mrange_right = mrange_left + df_view_pager_sizes[1] - 1;
                var rrange_left = pages - df_view_pager_sizes[2] + 1;
                if (mrange_left < 3) {
                    mrange_left = 3;
                    mrange_right = df_view_pager_sizes[1] + 2;
                }
                if (mrange_right > pages - 2){
                    mrange_left = pages - df_view_pager_sizes[1] - 1;
                    mrange_right = pages - 2;
                }
                if (lrange_right > pages) lrange_right = pages;
                if (mrange_right > pages) mrange_right = pages;
                if (mrange_left < 1) mrange_left = 1;
                if (rrange_left < 1) rrange_left = 1;

                pagination_obj.data('pages', pages);

                that.$el.find('.df-pager').remove();
                that.$el.find('.df-elip-pager').remove();

                for (i = 1; i <= lrange_right; i ++) {
                    page_str = df_view_pager_html.replace(/\{page\}/g, i);
                    $(page_str).insertBefore(that.$el.find('.df-pager-last'));
                }
                if (i < mrange_left - 1) {
                    $(df_view_elip_pager_html).insertBefore(that.$el.find('.df-pager-last'));
                    i = mrange_left;
                }
                for (; i <= mrange_right; i ++) {
                    page_str = df_view_pager_html.replace(/\{page\}/g, i);
                    $(page_str).insertBefore(that.$el.find('.df-pager-last'));
                }
                if (i < rrange_left - 1) {
                    $(df_view_elip_pager_html).insertBefore(that.$el.find('.df-pager-last'));
                    i = rrange_left;
                }
                for (; i <= pages; i ++) {
                    page_str = df_view_pager_html.replace(/\{page\}/g, i);
                    $(page_str).insertBefore(that.$el.find('.df-pager-last'));
                }
                that.$el.find('.pagination').removeClass('active');
                that.$el.find('.df-pager-top').removeClass('disabled');
                that.$el.find('.df-pager-' + (records.page + 1)).addClass('active');
                if (records.page == 0) that.$el.find('.df-pager-first').addClass('disabled');
                if (records.page == records.pages - 1) that.$el.find('.df-pager-last').addClass('disabled');
                that.$el.find('.show-table .pagination').show();
            }
        },

        _build_grouped_data: function(config, records) {
            var i, j, k;
            var group_data, cur_key, obj_key;
            var cur_key_group;

            var x_keys = [];
            var x_key_set = {}, y_value_col_set = {};

            if (!config.keys || config.keys.length == 0) {
                var max_data_length = 0;
                for (i = 0; i < records.data.length; i++) {
                    group_data = records.data[i];
                    if (group_data.$count > max_data_length)
                        max_data_length = group_data.$count;
                }
                for (i = 0; i < max_data_length; i++) {
                    x_keys.push(i);
                    x_key_set[i] = true;
                }
                for (i = 0; i < records.data.length; i++) {
                    group_data = records.data[i];
                    group_data.$keys = [];
                    for (j = 0 ;j < group_data.$count; j++) {
                        group_data.$keys.push(j);
                    }
                }
            } else if (config.keys.length == 1) {
                for (i = 0; i < records.keys.length; i++) {
                    cur_key = records.keys[i][0];
                    x_keys.push(cur_key);
                    x_key_set[cur_key] = true;
                }
                for (i = 0; i < records.data.length; i++) {
                    group_data = records.data[i];
                    group_data.$keys = [];
                    for (j = 0; j < group_data.$count; j++) {
                        cur_key = group_data[config.keys[0]][j];
                        group_data.$keys.push(cur_key);
                    }
                }
            } else {
                for (i = 0; i < records.keys.length; i++) {
                    cur_key_group = [];
                    for (j = 0; j < config.keys.length; j++) {
                        cur_key_group.push(config.keys[j] + '=' + records.keys[i][j]);
                    }
                    cur_key = cur_key_group.join(' ');
                    x_keys.push(cur_key);
                    x_key_set[cur_key] = true;
                }
                for (i = 0; i < records.data.length; i++) {
                    group_data = records.data[i];
                    group_data.$keys = [];
                    for (j = 0 ;j < group_data.$count; j++) {
                        cur_key_group = [];
                        for (k = 0; k < config.keys.length; k++) {
                            cur_key_group.push(config.keys[k] + '=' + group_data[config.keys[k]][j]);
                        }
                        cur_key = cur_key_group.join(' ');
                        group_data.$keys.push(cur_key);
                    }
                }
            }

            if (config.values) {
                if (config.values instanceof Array) {
                    for (i = 0; i < config.values.length; i++) {
                        y_value_col_set[config.values[i]] = true;
                    }
                } else {
                    for (i in config.values) {
                        y_value_col_set[i] = true;
                    }
                }
            }
            var grouped_data = [];
            for (i = 0; i < records.data.length; i++) {
                group_data = records.data[i];
                for (obj_key in group_data) {
                    var key_str = obj_key.toString();
                    if (key_str.startsWith('$')) continue;
                    var splited_key;
                    if (key_str.indexOf('__') >= 0)
                        splited_key = key_str.rsplit('__', 1);
                    else
                        splited_key = [key_str, ''];
                    if (!y_value_col_set[splited_key[0]]) continue;
                    var data_obj = {}, data_map = {};

                    for (j = 0; j < group_data.$count; j++) {
                        data_map[group_data.$keys[j]] = group_data[key_str][j];
                    }
                    if (records.groups)
                        data_obj.group = records.groups[i].join(' ');
                    else
                        data_obj.group = '';
                    data_obj.value = splited_key[0];
                    data_obj.agg = splited_key[1];
                    data_obj.data = [];
                    for (j = 0; j < x_keys.length; j++) {
                        data_obj.data.push(data_map[x_keys[j]]);
                    }
                    grouped_data.push(data_obj);
                }
            }
            return {groups: grouped_data, xkeys: x_keys};
        },

        _refresh_chart: function(binding) {
            var that = this;
            var chart_body = $(that.$el.find(that._chart_bindings[binding].panel).find('.show-body'));

            var resizer = function() {
                var chart = chart_body.data('chart');
                if (!chart) return;
                var old_height = chart_body.height();
                window.setTimeout(function () {
                    chart_body.height(old_height - 5);
                    chart.resize();
                    window.setTimeout(function () {
                        chart_body.height(old_height);
                        chart.resize();
                    }, 50);
                }, 50);
                var canvas_width = chart_body.find('canvas').width();
                if (!canvas_width) {
                    chart.resize();
                    window.setTimeout(resizer, 100);
                }
            };
            window.setTimeout(resizer, 100);
        },

        _update_bar_chart: function() {
            var that = this, i;
            var records = that.model.get('bar_chart_records');
            that._clear_widget_spaces();

            var chart_body = that.$el.find('.show-bar-chart .show-body');
            chart_body.css('height', '400px');
            var chart;
            if ($(chart_body).data('chart'))
                chart = $(chart_body).data('chart');
            else
                chart = echarts.init(chart_body.get(0), 'westeros');

            var series_array = [];
            var grouped_data = that._build_grouped_data(that._bar_chart_conf, records);
            for (i = 0; i < grouped_data.groups.length; i++) {
                var group_datum = grouped_data.groups[i];
                var label;
                if (records.groups)
                    label = 'Group: ' + group_datum.group + ' Data: ' +  group_datum.value + '-' + group_datum.agg;
                else
                    label = 'Data: ' +  group_datum.value + '-' + group_datum.agg;
                series_array.push({
                    name: label,
                    type: 'bar',
                    data: group_datum.data
                });
            }

            var options = {
                tooltip: {
                    trigger: 'axis',
                    axisPointer: { type: 'shadow' }
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '3%',
                    containLabel: true
                },
                xAxis: [
                    {
                        type: 'category',
                        data: grouped_data.xkeys,
                        axisTick: { alignWithLabel: true }
                    }
                ],
                yAxis: [
                    {
                        type : 'value',
                        min: 'auto',
                        max: 'auto'
                    }
                ],
                series: series_array
            };
            chart.setOption(options, true, false);
            $(chart_body).data('chart', chart);
            that._refresh_chart('bar');
        },

        _update_pie_chart: function() {
            var that = this, i;
            var records = that.model.get('pie_chart_records');
            that._clear_widget_spaces();

            var chart_body = that.$el.find('.show-pie-chart .show-body');
            chart_body.css('height', '400px');
            var chart;
            if ($(chart_body).data('chart'))
                chart = $(chart_body).data('chart');
            else
                chart = echarts.init(chart_body.get(0), 'westeros');

            var grouped_data = that._build_grouped_data(that._pie_chart_conf, records);
            var single_data = grouped_data.groups[0].data;

            var pie_data = [];
            for (i = 0; i < grouped_data.xkeys.length; i++)
                pie_data.push({name: grouped_data.xkeys[i], value: single_data[i]});

            var options = {
                tooltip : {
                    trigger: 'item'
                },
                legend: {
                    orient: 'vertical',
                    left: 'left',
                    data: grouped_data.xkeys
                },
                series : [
                    {
                        type: 'pie',
                        data: pie_data,
                        itemStyle: {
                            emphasis: {
                                shadowBlur: 10,
                                shadowOffsetX: 0,
                                shadowColor: 'rgba(0, 0, 0, 0.5)'
                            }
                        }
                    }
                ]
            };
            chart.setOption(options, true, false);
            chart.resize();
            $(chart_body).data('chart', chart);
            that._refresh_chart('pie');
        },

        _update_line_chart: function() {
            var that = this, i, j;
            var records = that.model.get('line_chart_records');
            that._clear_widget_spaces();

            var chart_body = that.$el.find('.show-line-chart .show-body');
            chart_body.css('height', '400px');
            var chart;
            if ($(chart_body).data('chart'))
                chart = $(chart_body).data('chart');
            else
                chart = echarts.init(chart_body.get(0), 'westeros');

            var series_array = [], is_numerical = true, group_datum;
            var grouped_data = that._build_grouped_data(that._line_chart_conf, records);
            for (i = 0; i < grouped_data.groups.length; i++) {
                group_datum = grouped_data.groups[i];
                for (j = 0; j < group_datum.data.length; j++) {
                    if (group_datum.data[j]) {
                        if (isNaN(parseFloat(grouped_data.xkeys[j]))) {
                            is_numerical = false;
                            break;
                        }
                    }
                }
                if (!is_numerical) break;
            }
            for (i = 0; i < grouped_data.groups.length; i++) {
                group_datum = grouped_data.groups[i];
                var label;
                if (records.groups)
                    label = 'Group: ' + group_datum.group + ' Data: ' +  group_datum.value;
                else
                    label = 'Data: ' +  group_datum.value;
                var data_array = [];
                if (is_numerical) {
                    for (j = 0; j < group_datum.data.length; j++) {
                        if (group_datum.data[j])
                            data_array.push([grouped_data.xkeys[j], group_datum.data[j]]);
                    }
                } else {
                    for (j = 0; j < group_datum.data.length; j++) {
                        if (group_datum.data[j])
                            data_array.push(group_datum.data[j]);
                    }
                }
                series_array.push({
                    name: label,
                    type: 'line',
                    smooth: true,
                    data: data_array
                });
            }

            var options = {
                tooltip: {
                    trigger: 'axis',
                    axisPointer: { type: 'shadow' }
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '3%',
                    containLabel: true
                },
                xAxis: {
                    min: 'auto',
                    max: 'auto',
                    type: is_numerical ? 'value' : 'category'
                },
                yAxis: {
                    min: 'auto',
                    max: 'auto'
                },
                series: series_array
            };
            if (!is_numerical)
                options.xAxis.data = grouped_data.xkeys;

            chart.setOption(options, true, false);
            $(chart_body).data('chart', chart);
            that._refresh_chart('line');
        },

        _update_scatter_chart: function() {
            var that = this, i, j;
            var records = that.model.get('scatter_chart_records');
            that._clear_widget_spaces();

            var chart_body = that.$el.find('.show-scatter-chart .show-body');
            chart_body.css('height', '400px');
            var chart;
            if ($(chart_body).data('chart'))
                chart = $(chart_body).data('chart');
            else
                chart = echarts.init(chart_body.get(0), 'westeros');

            var series_array = [], is_numerical = true, group_datum;
            var grouped_data = that._build_grouped_data(that._scatter_chart_conf, records);
            for (i = 0; i < grouped_data.groups.length; i++) {
                group_datum = grouped_data.groups[i];
                for (j = 0; j < group_datum.data.length; j++) {
                    if (group_datum.data[j]) {
                        if (isNaN(parseFloat(grouped_data.xkeys[j]))) {
                            is_numerical = false;
                            break;
                        }
                    }
                }
                if (!is_numerical) break;
            }
            for (i = 0; i < grouped_data.groups.length; i++) {
                group_datum = grouped_data.groups[i];
                var label;
                if (records.groups)
                    label = 'Group: ' + group_datum.group + ' Data: ' +  group_datum.value;
                else
                    label = 'Data: ' +  group_datum.value;
                var data_array = [];
                if (is_numerical) {
                    for (j = 0; j < group_datum.data.length; j++) {
                        if (group_datum.data[j])
                            data_array.push([grouped_data.xkeys[j], group_datum.data[j]]);
                    }
                } else {
                    for (j = 0; j < group_datum.data.length; j++) {
                        if (group_datum.data[j])
                            data_array.push(group_datum.data[j]);
                    }
                }
                series_array.push({
                    name: label,
                    type: 'scatter',
                    smooth: true,
                    data: data_array
                });
            }

            var options = {
                tooltip: {
                    trigger: 'axis',
                    axisPointer: { type: 'shadow' }
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '3%',
                    containLabel: true
                },
                xAxis: {
                    min: 'auto',
                    max: 'auto',
                    type: is_numerical ? 'value' : 'category'
                },
                yAxis: {
                    min: 'auto',
                    max: 'auto'
                },
                series: series_array
            };
            if (!is_numerical)
                options.xAxis.data = grouped_data.xkeys;

            chart.setOption(options, true, false);
            $(chart_body).data('chart', chart);
            that._refresh_chart('scatter');
        },
    });

    return {
        DFView: DFView
    }
});
