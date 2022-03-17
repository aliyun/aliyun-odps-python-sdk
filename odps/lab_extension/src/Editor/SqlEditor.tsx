import { CodeCell } from '@jupyterlab/cells';
import React from 'react';
import ReactDOM from 'react-dom';
import { INotebookTracker } from '@jupyterlab/notebook';
import { Signal } from '@lumino/signaling';
import { debounce } from 'lodash';
import {
  ODPS_CONFIGURE_PYTHON_CODE,
  ODPS_EXECUTE_PYTHON_CODE
} from './Template';
import Axios from 'axios';
import { LSPEditorConfigBuilder } from './ConfigBuilder/LSPEditorConfigBuilder';
import { isInner } from '../Utils/isInner';

/**
 * 1. Insert code cell for ak and odps configuration
 * 2. Mount SQL Editor on current cell
 *    bind sql change to code cell model value change
 * @param tracker
 */
export const onSqlCellTypeSelected = (tracker: INotebookTracker) => {
  Private.insertOdpsConfigureCell(tracker);
  const cell = tracker.activeCell as CodeCell;
  Private.mountSqlEditor(cell);
};

/**
 * Traversal all cells and reverse state for sql cell
 * @param tracker
 */
export const reverseAllEditor = (tracker: INotebookTracker) => {
  const notebook = tracker.currentWidget;
  const model = notebook.model;

  for (let i = 0; i <= model.cells.length; i++) {
    const cellModel = model.cells.get(i);
    if (!cellModel) {
      break;
    }
    if (cellModel.metadata.get('odps_sql_cell')) {
      const find = notebook.content.widgets.find(
        item => item.model === cellModel
      ) as CodeCell;
      Private.mountSqlEditor(find);
    }
  }
};

namespace Private {
  /**
   * judge if there is any configure cell already exist
   * if not, this function will
   * insert an OdpsConfigure python code before SQL Editor
   * @param tracker
   */
  export const insertOdpsConfigureCell = (tracker: INotebookTracker) => {
    const notebook = tracker.currentWidget!;
    const model = notebook.model;

    for (let i = 0; i <= model.cells.length; i++) {
      const cellModel = model.cells.get(i);
      if (!cellModel) {
        break;
      }
      if (cellModel.metadata.get('odps_configure')) {
        return;
      }
    }

    const factory = model.contentFactory;
    const codeCellModel = factory.createCodeCell({});
    codeCellModel.metadata.set('odps_configure', true);

    if (isInner()) {
    } else {
      codeCellModel.value.text = ODPS_CONFIGURE_PYTHON_CODE();
      model.cells.insert(notebook.content.activeCellIndex, codeCellModel);
    }
  };

  /**
   * insert SQL Editor and mount
   * @param cell
   */
  export const mountSqlEditor = (cell: CodeCell) => {
    const codeCellModel = cell.model;

    codeCellModel.mimeType = 'ipython/sql';
    cell.editor.host.style.height = '400px';
    cell.editor.host.oncontextmenu = e => {
      e.stopPropagation();
    };

    const defaultSql: string =
      (codeCellModel.metadata.get('sql_value') as string | undefined) || '';
    codeCellModel.value.text = ODPS_EXECUTE_PYTHON_CODE(defaultSql);
    codeCellModel.metadata.set('odps_sql_cell', true);

    const builder = new LSPEditorConfigBuilder();

    window.getLspEditor.then(() => {
      const reactEle = React.createElement(window.LSPEditor, {
        ...builder
          .autoTheme()
          .autoLSPUrl()
          .content(defaultSql)
          .uri(codeCellModel.id)
          .build(),
        // @ts-ignore
        onChange: (content: string) => {
          signal.emit({
            content
          });
        }
      });

      const signal = new Signal<
        typeof reactEle,
        {
          content: string;
        }
      >(reactEle);
      signal.connect(
        debounce((sender, args) => {
          codeCellModel.value.text = ODPS_EXECUTE_PYTHON_CODE(args.content);
          codeCellModel.metadata.set('sql_value', args.content);
        }, 100)
      );

      ReactDOM.render(reactEle, cell.editor.host);
    });
  };
}
