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
 * InstancesProgress front end (ES Module)
 * Mounts a React component (ProgressViewer with Radix Dialog) into this.el.
 *
 * The group list is kept here rather than in React state: the kernel sends
 * group updates immediately after `display(widget)`, which can be before React
 * finished mounting the component. A listener installed by a React effect
 * silently drops those first updates.
 */

import { DOMWidgetView } from "@jupyter-widgets/base";
import React from "react";
import { ProgressViewer } from "./components/ProgressViewer.jsx";
import { createReactRoot } from "./react-root.js";

export class InstancesProgressView extends DOMWidgetView {
  render() {
    this._groups = {};
    this._groupOrder = [];
    this._root = createReactRoot(this.el);

    this.model.on("msg:custom", this._handleCustomMsg, this);
    this.model.on("change:prefix", this._renderReact, this);
    this.model.on("change:suffix", this._renderReact, this);

    this._renderReact();
  }

  _handleCustomMsg(msg) {
    let msgObj = msg;
    try {
      if (typeof msg === "string") {
        msgObj = JSON.parse(msg);
      }
      if (!msgObj) {
        return;
      }
      const action = msgObj.action;
      const content = msgObj.content || [];

      if (action === "update") {
        const groups = { ...this._groups };
        const order = [...this._groupOrder];
        content.forEach((groupJson) => {
          const group =
            typeof groupJson === "string" ? JSON.parse(groupJson) : groupJson;
          if (!groups[group.key]) {
            order.push(group.key);
          }
          groups[group.key] = group;
        });
        this._groups = groups;
        this._groupOrder = order;
      } else if (action === "delete") {
        const groups = { ...this._groups };
        content.forEach((key) => delete groups[key]);
        this._groups = groups;
        this._groupOrder = this._groupOrder.filter(
          (key) => !content.includes(key)
        );
      } else if (action === "clear") {
        this._groups = {};
        this._groupOrder = [];
      }
    } catch (e) {
      // Ignore malformed messages, matching the kernel sender, which never
      // raises on serialization errors.
      return;
    }
    this._renderReact();
  }

  _renderReact() {
    if (!this._root) {
      return;
    }
    this._root.render(
      React.createElement(ProgressViewer, {
        groups: this._groupOrder.map((key) => this._groups[key]),
        prefix: this.model.get("prefix"),
        suffix: this.model.get("suffix"),
      })
    );
  }

  update(options) {
    if (options === undefined || options.updated_view !== this) {
      this._renderReact();
    }
    super.update.apply(this);
  }

  remove() {
    if (this._root) {
      this._root.unmount();
      this._root = null;
    }
    super.remove();
  }
}