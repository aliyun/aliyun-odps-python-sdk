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
 * Build the prebuilt JupyterLab extension.
 *
 * `@jupyterlab/core-meta` ships the JupyterLab core package metadata as
 * `core.package.json`, but `@jupyterlab/builder`'s CLI reads `package.json`
 * from the directory given to `--core-path`. Stage that metadata under a
 * temporary directory and point the builder at it.
 *
 * The whole build stays inside node on purpose. Python entry points such as
 * `jupyter-builder` or `jupyter labextension` cannot be used here: pip's
 * isolated PEP 517 build environment replaces sys.path, so packages installed
 * in the surrounding environment (including jupyterlab itself) are not
 * importable while a wheel is being built.
 */

const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const uiDir = path.resolve(__dirname, "..");

function resolveFromNodeModules(pkg, relative) {
  try {
    return require.resolve(`${pkg}/${relative}`);
  } catch (e) {
    // Fall back to the usual layout when the package restricts `exports`.
    return path.join(uiDir, "node_modules", pkg, relative);
  }
}

const coreMetaDir = path.dirname(
  resolveFromNodeModules("@jupyterlab/core-meta", "package.json")
);
const coreMetaFile = path.join(coreMetaDir, "core.package.json");
if (!fs.existsSync(coreMetaFile)) {
  console.error(`Cannot find JupyterLab core metadata at ${coreMetaFile}`);
  process.exit(1);
}

const stagedCoreDir = fs.mkdtempSync(path.join(os.tmpdir(), "pyodps-core-meta-"));
fs.copyFileSync(coreMetaFile, path.join(stagedCoreDir, "package.json"));

const builder = resolveFromNodeModules(
  "@jupyterlab/builder",
  "lib/build-labextension.js"
);
if (!fs.existsSync(builder)) {
  console.error(`Cannot find @jupyterlab/builder at ${builder}`);
  process.exit(1);
}

const result = spawnSync(
  process.execPath,
  [
    builder,
    "--core-path",
    stagedCoreDir,
    uiDir, // positional: the extension package directory
    ...process.argv.slice(2),
  ],
  { stdio: "inherit", cwd: uiDir }
);

try {
  fs.rmSync(stagedCoreDir, { recursive: true, force: true });
} catch (e) {
  // Leaving a temp directory behind must not fail the build.
}

if (result.status !== 0) {
  process.exit(result.status === null ? 1 : result.status);
}

// The built extension needs the install.json marker alongside it.
const installJson = path.join(uiDir, "install.json");
const targetDir = path.join(uiDir, "labextension");
if (fs.existsSync(installJson) && fs.existsSync(targetDir)) {
  fs.copyFileSync(installJson, path.join(targetDir, "install.json"));
}