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
 * Classic Notebook nbextension build — UMD output for notebook <7.
 *
 * The JupyterLab labextension is built separately by `jupyter-builder`
 * (invoked via `npm run build:labextension`), which uses @rspack/core with
 * module federation. This webpack config only produces the classic Notebook
 * bundle (nbextension/main.js) that registers widget views via RequireJS.
 *
 * React/react-dom are bundled inline here because classic Notebook has no
 * shared module registry.
 */
const path = require("path");

module.exports = {
  mode: "production",
  entry: "./src/extension.js",
  output: {
    path: path.resolve(__dirname, "nbextension"),
    filename: "main.js",
    library: {
      type: "umd",
      name: "pyodps",
    },
  },
  resolve: {
    extensions: [".js", ".jsx"],
  },
  externals: {
    // @jupyter-widgets/base is provided by widgetsnbextension's AMD modules
    // in classic Notebook; do not bundle it.
    "@jupyter-widgets/base": "@jupyter-widgets/base",
    "@jupyter-widgets/controls": "@jupyter-widgets/controls",
  },
  module: {
    rules: [
      {
        test: /\.jsx?$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
          options: {
            presets: [
              "@babel/preset-env",
              ["@babel/preset-react", { runtime: "automatic" }],
            ],
          },
        },
      },
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"],
      },
      // Fonts referenced by @font-face (the scatter-chart icon font).
      {
        test: /\.(woff2?|ttf|eot|svg)$/,
        type: "asset/resource",
      },
    ],
  },
};
