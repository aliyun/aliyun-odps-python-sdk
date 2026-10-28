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
 * PyODPS Jupyter extension entry point.
 * Registers widget view modules for both JupyterLab and classic Notebook.
 */

import { IJupyterWidgetRegistry } from "@jupyter-widgets/base";
import { InstancesProgressView } from "./progress.js";
import { DFView } from "./df-view.js";
import "./styles.css";

// Widget module protocol version. The kernel side declares the value it
// expects in `_view_module_version` (odps/ui/progress.py and odps/df/ui.py),
// and WidgetManager resolves registered modules with semver, so both sides
// must agree or the views fail to load in the browser.
const WIDGET_MODULE_VERSION = "2.0.0";

const WIDGET_MODULES = [
  {
    name: "pyodps/progress",
    version: WIDGET_MODULE_VERSION,
    exports: {
      InstancesProgress: InstancesProgressView,
    },
  },
  {
    name: "pyodps/df-view",
    version: WIDGET_MODULE_VERSION,
    exports: {
      DFView: DFView,
    },
  },
];

// --- JupyterLab labextension plugin ---
// Registers PyODPS widget view modules with the JupyterLab widget registry
// via the IJupyterWidgetRegistry token from @jupyter-widgets/base.
// This is the canonical pattern used by @jupyter-widgets/jupyterlab-manager:
// the extension consumes the registry and calls registerWidget() for each
// widget module so WidgetManager.loadClass can resolve "pyodps/progress"
// and "pyodps/df-view" by name.

const PYODPS_PLUGIN_ID = "pyodps:widget-registry";

const widgetRegistryPlugin = {
  id: PYODPS_PLUGIN_ID,
  description: "PyODPS interactive widget views",
  autoStart: true,
  requires: [IJupyterWidgetRegistry],
  activate: function (app, registry) {
    WIDGET_MODULES.forEach(function (m) {
      registry.registerWidget(m);
    });
  },
};

export default widgetRegistryPlugin;

// --- Classic Notebook nbextension entry ---
// The kernel declares `_view_module` as the exact AMD module names
// "pyodps/progress" and "pyodps/df-view" (see odps/ui/progress.py and
// odps/df/ui.py), and the classic widget manager resolves them through
// require([_view_module]). @jupyter-widgets/base has no registry-based
// fallback for classic Notebook, so the bundle must define those names.
let _classicRegistered = false;

function _registerClassicModules() {
  if (_classicRegistered || typeof define !== "function" || !define.amd) {
    return;
  }
  _classicRegistered = true;
  define("pyodps/progress", [], function () {
    return {
      InstancesProgress: InstancesProgressView,
      version: WIDGET_MODULE_VERSION,
    };
  });
  define("pyodps/df-view", [], function () {
    return {
      DFView: DFView,
      version: WIDGET_MODULE_VERSION,
    };
  });
}

// Classic Notebook may invoke this entry point explicitly; registration is
// idempotent, so both this and the auto-registration below are safe.
export function load_ipython_extension() {
  _registerClassicModules();
}

// Register as soon as the bundle is evaluated under RequireJS.
if (typeof window !== "undefined") {
  _registerClassicModules();
}
