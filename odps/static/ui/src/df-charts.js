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
/*
 * Chart fields and chart options of the DataFrame view.
 *
 * The three field families keep the roles they have in the classic notebook
 * front end, only reimplemented without jQuery/chosen:
 *
 *   keys   -> x-axis categories
 *   groups -> one series per group value
 *   values -> measures; the kernel aggregates them and names the resulting
 *             columns "<column>__<method>", or selects them unchanged when the
 *             config asks for no aggregation (line and scatter charts)
 *
 * One record per request arrives from the kernel as
 *
 *   {groups: [[key, ...], ...] | null,        one entry per group
 *    data:   [{"column": [values], "$count": n}, ...],
 *    keys:   [[key, ...], ...] | null,        distinct key tuples, sorted
 *    group_cols: [...], key_cols: [...]}
 */

/** Aggregations offered for value columns, as before. */
export const AGG_METHODS = [
  "count",
  "nunique",
  "min",
  "max",
  "sum",
  "mean",
  "median",
  "var",
  "std",
];

/** "(Row ID)" entry of the single x-axis selector: no key column. */
export const NO_KEY = "#ID";

/**
 * Per-chart controls:
 *   bar/pie   aggregate their value columns and can edit the methods
 *   line/scatter plot the selected columns as they are, with a single x axis
 */
export const CHART_BINDINGS = {
  bar: { groups: true, agg: true, keysMultiple: true, defaultAgg: "sum" },
  pie: { groups: false, agg: true, keysMultiple: true, defaultAgg: "count" },
  line: { groups: true, agg: false, keysMultiple: false, defaultAgg: "sum" },
  scatter: { groups: true, agg: false, keysMultiple: false, defaultAgg: "sum" },
};

export function valueColumnNames(values) {
  if (!values) return [];
  return Array.isArray(values) ? values : Object.keys(values);
}

/**
 * Default field selection: the leading columns take the group/key/value roles
 * they took in the classic front end. Returns null when there are too few
 * columns to derive anything sensible.
 */
export function createDefaultConfig(columns, chart) {
  const binding = CHART_BINDINGS[chart];
  if (!binding || columns.length < 2) {
    return null;
  }
  const asValue = (name) => (binding.agg ? { [name]: [binding.defaultAgg] } : [name]);
  if (columns.length === 2) {
    return { groups: [], keys: [columns[0]], values: asValue(columns[1]) };
  }
  return {
    groups: binding.groups ? [columns[0]] : [],
    keys: [columns[1]],
    values: asValue(columns[2]),
  };
}

/**
 * Resolve a payload column to its metric: either a configured column used
 * as-is, or an aggregated "<column>__<method>".
 */
function resolveMetric(column, configured) {
  if (configured.has(column)) {
    return { metric: column, method: "" };
  }
  const pos = column.lastIndexOf("__");
  if (pos > 0) {
    const metric = column.slice(0, pos);
    if (configured.has(metric)) {
      return { metric, method: column.slice(pos + 2) };
    }
  }
  return null;
}

/**
 * Turn the kernel payload into series aligned with the x-axis categories.
 *
 * A group holds only the keys it actually has, so each of its values is looked
 * up by key; keys that a group lacks become gaps instead of shifting the
 * following points.
 */
export function buildGroupedData(config, records) {
  const rows = (records && records.data) || [];
  const keyRows = (records && records.keys) || null;
  const keys = (config && config.keys) || [];
  const configured = new Set(valueColumnNames(config && config.values));

  const formatKey = (values) => {
    // Several key columns are labelled "column=value" pairs, as before.
    if (values.length > 1) {
      return values.map((value, i) => `${keys[i]}=${value}`).join(" ");
    }
    return String(values[0]);
  };

  let xkeys;
  if (keys.length === 0) {
    // No key column: plot against the position inside each group.
    let longest = 0;
    rows.forEach((row) => {
      longest = Math.max(longest, row.$count || 0);
    });
    xkeys = [];
    for (let i = 0; i < longest; i++) {
      xkeys.push(i);
    }
  } else {
    xkeys = (keyRows || []).map((row) =>
      formatKey(Array.isArray(row) ? row : [row])
    );
  }

  const groups = [];
  rows.forEach((row, index) => {
    const count = row.$count || 0;
    const rowKeys = [];
    for (let i = 0; i < count; i++) {
      if (keys.length === 0) {
        rowKeys.push(i);
      } else {
        rowKeys.push(
          formatKey(keys.map((key) => (row[key] ? row[key][i] : undefined)))
        );
      }
    }
    const label = records.groups ? records.groups[index].join(" ") : "";
    Object.keys(row).forEach((column) => {
      if (column.charAt(0) === "$") {
        return;
      }
      const metric = resolveMetric(column, configured);
      if (!metric) {
        return;
      }
      const byKey = new Map();
      rowKeys.forEach((key, i) => {
        byKey.set(key, row[column][i]);
      });
      groups.push({
        group: label,
        value: metric.metric,
        method: metric.method,
        data: xkeys.map((key) => byKey.get(key)),
      });
    });
  });

  return { xkeys, groups };
}

// Series identity and label. The method is always part of both, so two series
// of the same column can never read alike, whichever chart renders them.
function seriesId(entry) {
  return [entry.group, entry.value, entry.method].filter(Boolean).join("|");
}

function seriesName(entry) {
  const measure = entry.method ? `${entry.value}-${entry.method}` : entry.value;
  return entry.group ? `Group: ${entry.group} Data: ${measure}` : `Data: ${measure}`;
}

function withSeries(groups, type, points) {
  const series = groups.map((entry) => ({
    id: seriesId(entry),
    name: seriesName(entry),
    type: type,
    ...(points ? { smooth: type === "line" } : {}),
    data: points ? points(entry) : entry.data,
  }));
  // List the names explicitly: the legend then always matches the series the
  // current configuration produced.
  return {
    series,
    legend: { type: "scroll", data: series.map((s) => s.name) },
  };
}

function barOption({ xkeys, groups }) {
  const { series, legend } = withSeries(groups, "bar");
  return {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, transitionDuration: 0 },
    legend,
    grid: { left: "3%", right: "4%", bottom: "3%", containLabel: true },
    xAxis: [{ type: "category", data: xkeys, axisTick: { alignWithLabel: true } }],
    // No min/max: "auto" is not a valid axis bound (it parses to NaN and the
    // axis then renders without any tick labels).
    yAxis: [{ type: "value" }],
    series,
  };
}

function pieOption({ xkeys, groups }) {
  // A pie shows a single measure; the first configured value column is it.
  const first = groups[0];
  const values = first ? first.data : [];
  return {
    tooltip: { trigger: "item", transitionDuration: 0 },
    legend: { type: "scroll", orient: "vertical", left: "left", data: xkeys },
    series: [
      {
        id: first ? seriesId(first) : "values",
        name: first ? seriesName(first) : "value",
        type: "pie",
        // Slices carry name, value and share, so a slice is identifiable
        // without hovering it.
        label: { show: true, formatter: "{b}: {c} ({d}%)" },
        data: xkeys.map((name, i) => ({ name, value: values[i] })),
      },
    ],
  };
}

function lineScatterOption(type, { xkeys, groups }) {
  // Numeric categories become a value axis so the spacing is proportional.
  const numerical = xkeys.every((key) => !isNaN(parseFloat(key)));
  const { series, legend } = withSeries(groups, type, (entry) => {
    if (numerical) {
      // x rides along in every point, so points without a value can go.
      return entry.data
        .map((value, i) => [xkeys[i], value])
        .filter((point) => point[1] != null);
    }
    // On a category axis the positions carry the x values: leaving the gaps in
    // place keeps a group that misses keys from shifting.
    return entry.data;
  });

  const xAxis = { type: numerical ? "value" : "category" };
  if (!numerical) {
    xAxis.data = xkeys;
  }

  return {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, transitionDuration: 0 },
    legend,
    grid: { left: "3%", right: "4%", bottom: "3%", containLabel: true },
    xAxis,
    // No min/max: "auto" parses to NaN and suppresses every tick label.
    yAxis: { type: "value" },
    series,
  };
}

export function buildChartOption(chart, records, config) {
  const grouped = buildGroupedData(config, records);
  if (chart === "pie") {
    return pieOption(grouped);
  }
  if (chart === "bar") {
    return barOption(grouped);
  }
  return lineScatterOption(chart, grouped);
}