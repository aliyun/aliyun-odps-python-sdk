/*
 * Copyright 1999-2026 Alibaba Group Holding Ltd.
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
import React, { useState, useEffect, useCallback, useRef } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import ReactECharts from "echarts-for-react";
import "echarts";
import "../westeros.js";
import { ChartConfigPopover } from "./ChartConfigPopover.jsx";
import { buildChartOption, createDefaultConfig } from "../df-charts.js";

const MAX_TABLE_FETCH_SIZE = 10;
const CHART_TABS = ["bar", "pie", "line", "scatter"];
// The chart is re-initialised for every configuration, so records of the
// previous configuration can never stay on screen, and its annotations are
// rendered as SVG text: crisp when the panel is resized and selectable.
const CHART_OPTS = { renderer: "svg" };

export function DataFrameView({
  model,
  tableRecords,
  lastError,
  barChartRecords,
  pieChartRecords,
  lineChartRecords,
  scatterChartRecords,
}) {
  const [activeTab, setActiveTab] = useState("table");
  const [page, setPage] = useState(0);
  const [configs, setConfigs] = useState({});
  // Charts whose request has been answered neither by records nor an error.
  const inflight = useRef(new Set());
  // Charts whose displayed records come from an earlier configuration.
  const seenRecords = useRef({});
  const [staleCharts, setStaleCharts] = useState([]);

  // A fresh widget starts with an empty table_records ({}) rather than null;
  // `data` is present (possibly empty) only once the kernel answered.
  const columns = (tableRecords && tableRecords.columns) || [];
  const data = (tableRecords && tableRecords.data) || [];
  const tableLoaded = !!(tableRecords && tableRecords.data);
  const totalPages = tableRecords && tableRecords.size
    ? Math.ceil(tableRecords.size / MAX_TABLE_FETCH_SIZE)
    : 0;
  const chartRecords = {
    bar: barChartRecords,
    pie: pieChartRecords,
    line: lineChartRecords,
    scatter: scatterChartRecords,
  };

  useEffect(() => {
    // Ask the kernel for the first page of the table.
    if (model) {
      model.send({ action: "start_widget" });
    }
  }, [model]);

  const sendChartRequest = useCallback(
    (chart, chartConfig, force) => {
      if (!model || !chartConfig) return;
      if (force) inflight.current.delete(chart);
      if (inflight.current.has(chart)) return;
      inflight.current.add(chart);
      // Whatever is on screen now came from an older configuration: it must
      // not be shown next to the new fields while the answer is on its way.
      setStaleCharts((prev) => (prev.includes(chart) ? prev : [...prev, chart]));
      model.send({
        action: "aggregate_graph",
        groups: chartConfig.groups || [],
        keys: chartConfig.keys || [],
        values: chartConfig.values,
        target: `${chart}_chart_records`,
      });
    },
    [model]
  );

  useEffect(() => {
    // A chart is current again once a payload arrives after its last request.
    const arrived = CHART_TABS.filter(
      (chart) => chartRecords[chart] && seenRecords.current[chart] !== chartRecords[chart]
    );
    if (arrived.length === 0) return;
    arrived.forEach((chart) => {
      seenRecords.current[chart] = chartRecords[chart];
    });
    setStaleCharts((prev) => prev.filter((chart) => !arrived.includes(chart)));
  }, [barChartRecords, pieChartRecords, lineChartRecords, scatterChartRecords]);

  useEffect(() => {
    // A chart is ready once its records arrived; a failed request stays
    // retryable through Refresh in the configuration popover.
    CHART_TABS.forEach((chart) => {
      if (chartRecords[chart]) {
        inflight.current.delete(chart);
      }
    });
  }, [barChartRecords, pieChartRecords, lineChartRecords, scatterChartRecords]);

  useEffect(() => {
    // Chart fields cannot be defaulted before the table delivered its columns.
    if (activeTab === "table" || columns.length === 0) return;
    if (configs[activeTab] && chartRecords[activeTab]) return;
    const chartConfig = configs[activeTab] || createDefaultConfig(columns, activeTab);
    if (!chartConfig) return;
    if (!configs[activeTab]) {
      setConfigs((prev) => ({ ...prev, [activeTab]: chartConfig }));
    }
    sendChartRequest(activeTab, chartConfig, false);
  }, [activeTab, columns.length, configs, chartRecords, sendChartRequest]);

  const handleFetchTable = useCallback(
    (newPage) => {
      if (model) {
        model.send({ action: "fetch_table", page: newPage });
      }
      setPage(newPage);
    },
    [model]
  );

  const handleConfigApply = useCallback(
    (newConfig) => {
      if (activeTab === "table") return;
      setConfigs((prev) => ({ ...prev, [activeTab]: newConfig }));
      sendChartRequest(activeTab, newConfig, true);
    },
    [activeTab, sendChartRequest]
  );

  return (
    <div className="pyodps-df-view">
      <Tabs.Root value={activeTab} onValueChange={setActiveTab}>
        <Tabs.List className="pyodps-tab-list">
          <Tabs.Trigger value="table" className="pyodps-tab-trigger">
            <i className="fa fa-table"></i>
          </Tabs.Trigger>
          <Tabs.Trigger value="bar" className="pyodps-tab-trigger">
            <i className="fa fa-bar-chart"></i>
          </Tabs.Trigger>
          <Tabs.Trigger value="pie" className="pyodps-tab-trigger">
            <i className="fa fa-pie-chart"></i>
          </Tabs.Trigger>
          <Tabs.Trigger value="line" className="pyodps-tab-trigger">
            <i className="fa fa-line-chart"></i>
          </Tabs.Trigger>
          <Tabs.Trigger value="scatter" className="pyodps-tab-trigger">
            <i className="cf cf-scatter-chart"></i>
          </Tabs.Trigger>
          <div className="pyodps-chart-toolbar">
            {activeTab !== "table" && (
              <ChartConfigPopover
                columns={columns}
                chart={activeTab}
                config={configs[activeTab]}
                onApply={handleConfigApply}
              />
            )}
            <button
              className="pyodps-df-close"
              title="Close view and show the table representation"
              onClick={() => model.send({ action: "close_widget" })}
            >
              <i className="fa fa-times"></i>
            </button>
          </div>
        </Tabs.List>

        {lastError ? <div className="pyodps-df-error">{lastError}</div> : null}

        <Tabs.Content value="table" className="pyodps-tab-content">
          <DataTable
            columns={columns}
            data={data}
            loaded={tableLoaded}
            page={page}
            totalPages={totalPages}
            onPageChange={handleFetchTable}
          />
        </Tabs.Content>

        {CHART_TABS.map((chart) => (
          <Tabs.Content
            key={chart}
            value={chart}
            className="pyodps-tab-content"
          >
            <ChartView
              chart={chart}
              records={chartRecords[chart]}
              config={configs[chart]}
              stale={staleCharts.includes(chart)}
            />
          </Tabs.Content>
        ))}
      </Tabs.Root>
    </div>
  );
}

function DataTable({ columns, data, loaded, page, totalPages, onPageChange }) {
  if (!loaded) {
    return <div className="pyodps-table-loading">Loading...</div>;
  }
  if (data.length === 0) {
    return <div className="pyodps-table-empty">No rows</div>;
  }

  return (
    <div className="pyodps-table-container">
      <table className="pyodps-table">
        <thead>
          <tr>
            {columns.map((col, i) => (
              <th key={i}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, ri) => (
            <tr key={ri}>
              {row.map((cell, ci) => (
                <td key={ci}>{String(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {totalPages > 1 && (
        <div className="pyodps-pagination">
          <button onClick={() => onPageChange(0)} disabled={page === 0}>
            &laquo;
          </button>
          <button onClick={() => onPageChange(page - 1)} disabled={page === 0}>
            &lsaquo;
          </button>
          <span>
            Page {page + 1} of {totalPages}
          </span>
          <button
            onClick={() => onPageChange(page + 1)}
            disabled={page >= totalPages - 1}
          >
            &rsaquo;
          </button>
          <button
            onClick={() => onPageChange(totalPages - 1)}
            disabled={page >= totalPages - 1}
          >
            &raquo;
          </button>
        </div>
      )}
    </div>
  );
}

function ChartView({ chart, records, config, stale }) {
  if (stale || !records || !records.data) {
    // Never draw a payload that the current fields did not produce.
    return (
      <div className="pyodps-chart-loading">
        {stale && records && records.data ? "Updating..." : "Loading..."}
      </div>
    );
  }

  return (
    <div className="pyodps-chart-container">
      <ReactECharts
        option={buildChartOption(chart, records, config)}
        theme="westeros"
        notMerge
        lazyUpdate={false}
        opts={CHART_OPTS}
        style={{ height: "400px", width: "100%" }}
      />
    </div>
  );
}

export default DataFrameView;