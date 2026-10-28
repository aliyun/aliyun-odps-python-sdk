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
import React, { useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { AGG_METHODS, CHART_BINDINGS, NO_KEY, valueColumnNames } from "../df-charts.js";

/**
 * Floating item list of a field select.
 *
 * The menu is portalled to the document body and positioned from the trigger's
 * rectangle, so it is drawn *over* the configuration box without enlarging it,
 * and no ancestor of the widget (the config box, the output area, the notebook
 * panel — the latter two clip) can cut it off.
 */
function useFloatingMenu(open, onClose) {
  const triggerRef = useRef(null);
  const menuRef = useRef(null);
  const [style, setStyle] = useState(null);

  const close = useCallback(() => onClose(), [onClose]);

  useEffect(() => {
    if (!open) {
      setStyle(null);
      return;
    }
    const place = () => {
      const el = triggerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const below = window.innerHeight - rect.bottom - 8;
      const above = rect.top - 8;
      const flipUp = below < 140 && above > below;
      setStyle({
        position: "fixed",
        left: `${rect.left}px`,
        width: `${rect.width}px`,
        top: flipUp ? "auto" : `${rect.bottom + 2}px`,
        bottom: flipUp ? `${window.innerHeight - rect.top + 2}px` : "auto",
        maxHeight: `${Math.max(120, Math.min(240, flipUp ? above : below))}px`,
      });
    };
    place();
    // A menu pinned to the viewport must not drift when the page moves.
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", place);
    };
  }, [open, close]);

  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      const target = e.target;
      if (triggerRef.current && triggerRef.current.contains(target)) return;
      // The menu lives in the body, so it is not covered by the trigger check.
      if (menuRef.current && menuRef.current.contains(target)) return;
      close();
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open, close]);

  const menu = (children) =>
    open && style
      ? createPortal(
          <div className="pyodps-multiselect-dropdown" ref={menuRef} style={style}>
            {children}
          </div>,
          document.body
        )
      : null;

  return { triggerRef, menu };
}

/**
 * Chart field configuration.
 *
 * Which controls a chart gets, and what the three field families mean, is
 * described in df-charts.js. Aggregating charts (bar, pie) let every selected
 * value column choose its own aggregation methods; line and scatter plot the
 * selected columns as they are and use a single x-axis column.
 */
export function ChartConfigPopover({ columns, chart, config, onApply }) {
  const binding = CHART_BINDINGS[chart] || CHART_BINDINGS.bar;
  const [open, setOpen] = useState(false);
  const [groups, setGroups] = useState([]);
  const [keys, setKeys] = useState([]);
  const [values, setValues] = useState([]);
  const [methods, setMethods] = useState({});
  // At most one field dropdown is open: two open menus overlap each other and
  // the Refresh button.
  const [openSelect, setOpenSelect] = useState(null);
  const ref = useRef(null);

  const selectToggle = (id) => setOpenSelect(openSelect === id ? null : id);

  useEffect(() => {
    if (!open) {
      setOpenSelect(null);
      return;
    }
    const handler = (e) => {
      const target = e.target;
      if (ref.current && ref.current.contains(target)) {
        return;
      }
      // A field menu is portalled to the body, i.e. outside this subtree, so a
      // click inside it must not close the configuration box.
      if (target instanceof Element && target.closest(".pyodps-multiselect-dropdown")) {
        return;
      }
      setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  // Reopen on the applied configuration: unapplied edits are dropped.
  useEffect(() => {
    if (!open) return;
    const applied = config || {};
    setGroups(applied.groups || []);
    setKeys(applied.keys || []);
    setValues(valueColumnNames(applied.values));
    setMethods(
      binding.agg && applied.values && !Array.isArray(applied.values)
        ? { ...applied.values }
        : {}
    );
  }, [open, config, binding.agg]);

  const toggle = (list, setList, col, single) => {
    if (single) {
      setList(list.includes(col) ? [] : [col]);
    } else if (list.includes(col)) {
      setList(list.filter((c) => c !== col));
    } else {
      setList([...list, col]);
    }
  };

  const toggleValue = (col) => {
    if (values.includes(col)) {
      setValues(values.filter((c) => c !== col));
    } else {
      setValues([...values, col]);
      if (binding.agg && !methods[col]) {
        // Newly picked value columns start aggregated with sum, as before.
        setMethods({ ...methods, [col]: ["sum"] });
      }
    }
  };

  const setColumnMethods = (col, selected) => {
    setMethods({ ...methods, [col]: selected });
  };

  const handleApply = () => {
    const appliedValues = binding.agg
      ? Object.fromEntries(
          values.map((col) => [
            col,
            methods[col] && methods[col].length > 0 ? methods[col] : ["sum"],
          ])
        )
      : [...values];
    onApply({
      groups: binding.groups ? groups : [],
      keys: binding.keysMultiple
        ? keys
        : keys.length > 0 && keys[0] !== NO_KEY
        ? [keys[0]]
        : [],
      values: appliedValues,
    });
    setOpen(false);
  };

  return (
    <div className="pyodps-config-wrapper" ref={ref}>
      <button
        className="pyodps-config-btn"
        onClick={() => setOpen(!open)}
        title="Chart configuration"
      >
        <i className="fa fa-gear"></i>&nbsp;
        <i className="fa fa-caret-down"></i>
      </button>
      {open && (
        <div className="pyodps-config-dropdown">
          {binding.groups && (
            <div className="pyodps-config-row">
              <label>Groups:</label>
              <MultiSelect
                columns={columns}
                selected={groups}
                onToggle={(col) => toggle(groups, setGroups, col, false)}
                placeholder="Select group columns"
                open={openSelect === "groups"}
                onToggleOpen={() => selectToggle("groups")}
              />
            </div>
          )}
          <div className="pyodps-config-row">
            <label>{binding.keysMultiple ? "Keys:" : "X Axis:"}</label>
            <MultiSelect
              columns={columns}
              selected={keys}
              onToggle={(col) => toggle(keys, setKeys, col, !binding.keysMultiple)}
              single={!binding.keysMultiple}
              extra={binding.keysMultiple ? [] : [{ value: NO_KEY, label: "(Row ID)" }]}
              placeholder={binding.keysMultiple ? "Select key columns" : "Select a key column"}
              open={openSelect === "keys"}
              onToggleOpen={() => selectToggle("keys")}
            />
          </div>
          <div className="pyodps-config-row">
            <label>{binding.agg ? "Values:" : "Y Axis:"}</label>
            <MultiSelect
              columns={columns}
              selected={values}
              onToggle={toggleValue}
              placeholder="Select value columns"
              open={openSelect === "values"}
              onToggleOpen={() => selectToggle("values")}
            />
          </div>
          {binding.agg &&
            values.map((col) => (
              <div className="pyodps-config-row pyodps-agg-row" key={col}>
                <label>{col}:</label>
                <MethodSelect
                  selected={methods[col] || ["sum"]}
                  onChange={(selected) => setColumnMethods(col, selected)}
                  open={openSelect === `method:${col}`}
                  onToggleOpen={() => selectToggle(`method:${col}`)}
                />
              </div>
            ))}
          <button className="pyodps-refresh-btn" onClick={handleApply}>
            Refresh
          </button>
        </div>
      )}
    </div>
  );
}

function MethodSelect({ selected, onChange, open, onToggleOpen }) {
  const { triggerRef, menu } = useFloatingMenu(open, onToggleOpen);

  const toggleMethod = (method) => {
    if (selected.includes(method)) {
      onChange(selected.filter((m) => m !== method));
    } else {
      onChange([...selected, method]);
    }
  };

  return (
    <div className="pyodps-multiselect" ref={triggerRef}>
      <button
        className="pyodps-multiselect-trigger"
        onClick={onToggleOpen}
      >
        {selected.length > 0 ? selected.join(", ") : "sum"}
      </button>
      {menu(
        AGG_METHODS.map((method) => (
          <label
            key={method}
            className={
              selected.includes(method)
                ? "pyodps-multiselect-item selected"
                : "pyodps-multiselect-item"
            }
          >
            <input
              type="checkbox"
              checked={selected.includes(method)}
              onChange={() => toggleMethod(method)}
            />
            {method}
          </label>
        ))
      )}
    </div>
  );
}

function MultiSelect({ columns, selected, onToggle, placeholder, single, extra, open, onToggleOpen }) {
  const { triggerRef, menu } = useFloatingMenu(open, onToggleOpen);
  const entries = (extra || []).concat(columns.map((col) => ({ value: col, label: col })));

  return (
    <div className="pyodps-multiselect" ref={triggerRef}>
      <button
        className="pyodps-multiselect-trigger"
        onClick={onToggleOpen}
      >
        {selected.length > 0
          ? entries
              .filter((entry) => selected.includes(entry.value))
              .map((entry) => entry.label)
              .join(", ")
          : placeholder}
      </button>
      {menu(
        entries.map((entry) => (
          <label
            key={entry.value}
            className={
              selected.includes(entry.value)
                ? "pyodps-multiselect-item selected"
                : "pyodps-multiselect-item"
            }
          >
            <input
              type={single ? "radio" : "checkbox"}
              checked={selected.includes(entry.value)}
              onChange={() => onToggle(entry.value)}
            />
            {entry.label}
          </label>
        ))
      )}
    </div>
  );
}

export default ChartConfigPopover;