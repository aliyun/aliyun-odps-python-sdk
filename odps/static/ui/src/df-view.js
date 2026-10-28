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
 * PyODPS DataFrame Interactive Module (ES Module)
 * Mounts a React component (DataFrameView) into this.el.
 */

import { DOMWidgetView } from "@jupyter-widgets/base";
import React from "react";
import { DataFrameView } from "./components/DataFrameView.jsx";
import { createReactRoot } from "./react-root.js";
import "./westeros.js";

export class DFView extends DOMWidgetView {
  render() {
    this._root = createReactRoot(this.el);
    this._renderReact();

    this.model.on("change:table_records", this._renderReact, this);
    this.model.on("change:bar_chart_records", this._renderReact, this);
    this.model.on("change:pie_chart_records", this._renderReact, this);
    this.model.on("change:line_chart_records", this._renderReact, this);
    this.model.on("change:scatter_chart_records", this._renderReact, this);
    this.model.on("change:start_sign", this._handleStartSign, this);
    this.model.on("change:error_sign", this._handleErrorSign, this);
    this.model.on("change:last_error", this._renderReact, this);
    this.model.on("change:closed", this._syncReprVisibility, this);
    this.model.on("msg:custom", this._handleCustomMsg, this);

    this._syncReprVisibility();
  }

  processLuminoMessage(msg) {
    super.processLuminoMessage(msg);
    if (msg.type === "after-attach") {
      // The element is in the document now, so the surrounding output area
      // can finally be located.
      this._syncReprVisibility();
    }
  }

  /**
   * Output child that holds this widget. JupyterLab wraps every output in a
   * `.jp-OutputArea-child`; classic Notebook nests the widget in a
   * `.widget-area` inside its own `.output_area`.
   */
  _ownOutput() {
    return (
      this.el.closest(".jp-OutputArea-child") ||
      this.el.closest(".widget-area") ||
      this.el.closest(".output_area")
    );
  }

  /** Closest ancestor whose (descendant) children are the cell's outputs. */
  _outputContainer(own) {
    if (!own) return null;
    if (own.classList.contains("jp-OutputArea-child")) {
      return own.parentElement;
    }
    // Classic Notebook: the cell's outputs share a single `.output` box.
    return own.closest(".output") || own.parentElement;
  }

  /**
   * Rendered repr outputs belonging to the same cell, i.e. the pandas HTML
   * that `DataFrame._repr_html_` produced alongside this widget. Widget
   * outputs are excluded so other views are never touched.
   */
  _reprOutputs() {
    const own = this._ownOutput();
    if (!own) return [];
    if (own.classList.contains("jp-OutputArea-child")) {
      const area = own.parentElement;
      if (!area) return [];
      return Array.from(area.children).filter((child) => {
        if (child === own) return false;
        // The repr renders straight into the output child as
        // .jp-RenderedHTMLCommon (or .jp-RenderedText for the plain-text
        // fallback); widget outputs never contain those.
        return !!child.querySelector(".jp-RenderedHTMLCommon, .jp-RenderedText");
      });
    }
    // Classic Notebook: the widget occupies its own `.output_area` and the
    // kernel appends the pandas repr *after* it (ResultFrame._repr_html_
    // displays the widget before returning the HTML). Hide only that nearest
    // following repr so unrelated displays in the same cell stay visible.
    const ownArea = own.closest(".output_area") || own;
    const reprArea = this._classicReprOutputArea(ownArea);
    return reprArea ? [reprArea] : [];
  }

  /** Nearest following sibling output area that holds a rendered repr. */
  _classicReprOutputArea(ownArea) {
    let node = ownArea.nextElementSibling;
    while (node && node.classList && node.classList.contains("output_area")) {
      if (
        node.querySelector(
          ".output_subarea.output_result, .output_subarea.output_html, .output_subarea.rendered_html"
        )
      ) {
        return node;
      }
      // An adjacent non-repr output (e.g. stdout) is not the repr we own.
      break;
    }
    return null;
  }

  /**
   * Hide the pandas repr while the interactive view is open, and put it back
   * once the view is closed or removed. Both outputs stay in the notebook
   * document, so nothing is lost by hiding.
   */
  _syncReprVisibility() {
    const own = this._ownOutput();
    if (!own) {
      // Not attached yet: retry briefly.
      if ((this._syncAttempts || 0) < 40) {
        this._syncAttempts = (this._syncAttempts || 0) + 1;
        this._syncTimer = setTimeout(() => this._syncReprVisibility(), 50);
      }
      return;
    }
    this._syncAttempts = 0;

    const area = this._outputContainer(own);
    if (area && !this._observer) {
      // The repr output is appended after the widget, so watch for it.
      this._observer = new MutationObserver(() => this._syncReprVisibility());
      this._observer.observe(area, { childList: true });
    }

    const closed = !!this.model.get("closed");
    own.style.display = closed ? "none" : "";
    this._reprOutputs().forEach((child) => {
      child.style.display = closed ? "" : "none";
    });
  }

  _renderReact() {
    this._root.render(
      React.createElement(DataFrameView, {
        model: this.model,
        tableRecords: this.model.get("table_records"),
        lastError: this.model.get("last_error"),
        barChartRecords: this.model.get("bar_chart_records"),
        pieChartRecords: this.model.get("pie_chart_records"),
        lineChartRecords: this.model.get("line_chart_records"),
        scatterChartRecords: this.model.get("scatter_chart_records"),
      })
    );
  }

  _handleStartSign() {
    this._renderReact();
  }

  _handleErrorSign() {
    this._renderReact();
  }

  _handleCustomMsg() {
    this._renderReact();
  }

  remove() {
    if (this._syncTimer) {
      clearTimeout(this._syncTimer);
      this._syncTimer = null;
    }
    if (this._observer) {
      this._observer.disconnect();
      this._observer = null;
    }
    // Never leave the pandas repr hidden behind a widget that is going away.
    this._reprOutputs().forEach((child) => {
      child.style.display = "";
    });
    if (this._root) {
      this._root.unmount();
    }
    super.remove();
  }
}

export default { DFView };
