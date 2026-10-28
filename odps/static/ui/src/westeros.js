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
 * Westeros ECharts Theme (ES Module)
 * Registers the "westeros" theme with echarts.
 * Converted from AMD module to ES module.
 */

import * as echarts from "echarts";

if (!echarts) {
  console.error("ECharts is not loaded");
} else {
  echarts.registerTheme("westeros", {
    color: [
      "#516b91",
      "#59c4e6",
      "#edafda",
      "#93b7e3",
      "#a5e7f0",
      "#cbb0e3",
    ],
    backgroundColor: "#ffffff",
    textStyle: {
      fontFamily: "sans-serif",
    },
    title: {
      textStyle: { color: "#516b91", fontWeight: "normal" },
      subtextStyle: { color: "#93b7e3" },
    },
    legend: {
      textStyle: { color: "#516b91" },
    },
    xAxis: {
      axisLine: { lineStyle: { color: "#516b91" } },
      axisTick: { lineStyle: { color: "#516b91" } },
      axisLabel: { color: "#516b91" },
      splitLine: { lineStyle: { color: "#e3e6ef" } },
    },
    yAxis: {
      axisLine: { lineStyle: { color: "#516b91" } },
      axisTick: { lineStyle: { color: "#516b91" } },
      axisLabel: { color: "#516b91" },
      splitLine: { lineStyle: { color: "#e3e6ef" } },
    },
    tooltip: {
      // Dark enough for white text (8.8:1) and outlined, so the box and its
      // text stay readable when it floats over a bar of the same hue.
      backgroundColor: "#3c4b66",
      borderColor: "#ffffff",
      textStyle: { color: "#ffffff" },
      axisPointer: { lineStyle: { color: "#93b7e3" } },
    },
    categoryAxis: {
      axisLine: { lineStyle: { color: "#516b91" } },
      axisTick: { lineStyle: { color: "#516b91" } },
      axisLabel: { color: "#516b91" },
      splitLine: { lineStyle: { color: "#e3e6ef" } },
    },
    valueAxis: {
      axisLine: { lineStyle: { color: "#516b91" } },
      axisTick: { lineStyle: { color: "#516b91" } },
      axisLabel: { color: "#516b91" },
      splitLine: { lineStyle: { color: "#e3e6ef" } },
    },
    line: {
      smooth: true,
      itemStyle: { normal: { borderWidth: 2 } },
      lineStyle: { normal: { width: 2 } },
    },
    bar: {
      itemStyle: { normal: { barBorderWidth: 0, barBorderColor: "#ccc" } },
    },
    pie: {
      itemStyle: { normal: { borderWidth: 2, borderColor: "#fff" } },
    },
    radar: {
      symbol: "emptyCircle",
      symbolSize: 4,
    },
  });
}

export default echarts;
