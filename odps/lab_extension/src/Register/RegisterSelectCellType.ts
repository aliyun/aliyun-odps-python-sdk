import { INotebookTracker } from '@jupyterlab/notebook';
import { onSqlCellTypeSelected } from '../Editor/SqlEditor';

export const CELL_VALUE = 'code';
export const CELL_NAME = 'ODPS SQL';

export const registerSelectCellType = (tracker: INotebookTracker) => {
  tracker.currentChanged.connect((_, notebook) => {
    const selector = notebook.node.querySelector<HTMLSelectElement>(
      ':scope .jp-Notebook-toolbarCellTypeDropdown select'
    );
    if (selector.querySelector(':scope option[odps=true]')) {
      return;
    }
    const ele = document.createElement('option');
    ele.value = CELL_VALUE;
    ele.text = CELL_NAME;
    ele.setAttribute('odps', 'true');
    selector?.appendChild(ele);
    selector.addEventListener('change', evt => {
      const target = evt.target as HTMLSelectElement;
      if (target.value === CELL_VALUE) {
        onSqlCellTypeSelected(tracker);
      }
    });
  });
};
