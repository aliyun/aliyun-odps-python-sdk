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
import React, { useState, useEffect, useRef } from "react";
import * as Dialog from "@radix-ui/react-dialog";

function formatDate(dateStr) {
  const d = new Date(dateStr);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function ProgressViewer({ groups = [], prefix, suffix }) {
  const [selectedGroup, setSelectedGroup] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const stop = (e) => {
      const target = e.target;
      if (target && target.closest && target.closest(".pyodps-progress-launcher")) {
        e.stopPropagation();
      }
    };
    el.addEventListener("mousedown", stop, true);
    return () => el.removeEventListener("mousedown", stop, true);
  }, []);

  const handleGroupClick = (groupKey) => {
    setSelectedGroup(groupKey);
    setDialogOpen(true);
  };

  const selectedGroupData =
    groups.find((group) => group.key === selectedGroup) || null;

  return (
    <div className="pyodps-progress-container" ref={containerRef}>
      <span className="pyodps-progress-prefix" style={{ paddingRight: "5px" }}>
        {prefix}
      </span>
      <span className="pyodps-progress-groups">
        {groups.map((group) => (
          <a
            key={group.key}
            className="pyodps-progress-launcher"
            href="javascript:void(0)"
            style={{ marginRight: "5px" }}
            onClick={(e) => {
              e.preventDefault();
              handleGroupClick(group.key);
            }}
          >
            {group.name}
          </a>
        ))}
      </span>
      <span className="pyodps-progress-suffix">{suffix}</span>

      <Dialog.Root open={dialogOpen} onOpenChange={setDialogOpen} modal={false}>
        <Dialog.Portal>
          <div
            className="pyodps-dialog-overlay"
            onClick={() => setDialogOpen(false)}
          />
          <Dialog.Content
            className="pyodps-dialog-content"
            onInteractOutside={(e) => e.preventDefault()}
          >
            <div className="pyodps-dialog-header">
              <Dialog.Title className="pyodps-dialog-title">
                {selectedGroupData
                  ? `Progress of "${selectedGroupData.name}"`
                  : "Progress Viewer"}
              </Dialog.Title>
              <span className="pyodps-dialog-time">
                {selectedGroupData
                  ? `Generate time: ${formatDate(selectedGroupData.gen_time)}`
                  : ""}
              </span>
              <Dialog.Close className="pyodps-dialog-close">&times;</Dialog.Close>
            </div>
            <div className="pyodps-dialog-body">
              {selectedGroupData && selectedGroupData.instances
                ? selectedGroupData.instances.map((inst, idx) => (
                    <InstanceView key={inst.id} inst={inst} idx={idx} />
                  ))
                : null}
            </div>
            <div className="pyodps-dialog-footer">
              <div className="pyodps-progress-legend">
                <span className="pyodps-legend-item">
                  <span className="pyodps-legend-swatch pyodps-legend-swatch-finished" />
                  Finished
                </span>
                <span className="pyodps-legend-item">
                  <span className="pyodps-legend-swatch pyodps-legend-swatch-submitted" />
                  Submitted
                </span>
                <span className="pyodps-legend-item">
                  <span className="pyodps-legend-swatch pyodps-legend-swatch-waiting" />
                  Waiting
                </span>
              </div>
              <Dialog.Close asChild>
                <button className="pyodps-dialog-btn">Close</button>
              </Dialog.Close>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}

function InstanceView({ inst, idx }) {
  return (
    <div className="pyodps-instance" key={inst.id}>
      <div className="pyodps-instance-header">
        Instance {idx}: {inst.status}
        {inst.logview ? (
          <a
            className="pyodps-logview"
            href={inst.logview}
            target="_blank"
            rel="noreferrer"
          >
            Logview
          </a>
        ) : null}
      </div>
      {inst.tasks
        ? inst.tasks.map((task, tidx) => <TaskView key={tidx} task={task} />)
        : null}
    </div>
  );
}

function TaskView({ task }) {
  return (
    <div className="pyodps-task">
      <div className="pyodps-task-name">
        {task.name}: {task.status}
      </div>
      <div className="pyodps-stages">
        {task.stages && task.stages.length > 0
          ? task.stages.map((stage, sidx) => (
              <StageView key={sidx} stage={stage} />
            ))
          : "No stage information available."}
      </div>
    </div>
  );
}

function StageView({ stage }) {
  const total = stage.total_workers || 1;
  const backups = stage.backup_workers || 0;
  const denominator = total + backups;
  const terminatedPerc =
    ((stage.terminated_workers || 0) * 100 / denominator).toFixed(2) + "%";
  const runningPerc =
    ((stage.running_workers || 0) * 100 / denominator).toFixed(2) + "%";

  return (
    <div className="pyodps-stage">
      <div className="pyodps-stage-name">{stage.name}</div>
      <div className="pyodps-stage-progress">
        <div
          className="pyodps-progress-bar-terminated"
          style={{ width: terminatedPerc }}
        >
          {terminatedPerc}
        </div>
        <div
          className="pyodps-progress-bar-running"
          style={{ width: runningPerc }}
        >
          {runningPerc}
        </div>
      </div>
    </div>
  );
}

export default ProgressViewer;
