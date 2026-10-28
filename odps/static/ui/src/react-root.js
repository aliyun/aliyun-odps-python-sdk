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
 * React root helper.
 *
 * react-dom 18 provides createRoot(), react-dom 17 only the legacy render().
 * react-dom is a shared singleton, so the renderer that ends up in the widget
 * depends on the host: JupyterLab 4 provides react-dom 18, while JupyterLab 3
 * provides react-dom 17. Both must mount the same React components.
 */

import ReactDOM from "react-dom";

export function createReactRoot(el) {
  if (typeof ReactDOM.createRoot === "function") {
    const root = ReactDOM.createRoot(el);
    return {
      render: (node) => root.render(node),
      unmount: () => root.unmount(),
    };
  }

  return {
    render: (node) => ReactDOM.render(node, el),
    unmount: () => ReactDOM.unmountComponentAtNode(el),
  };
}