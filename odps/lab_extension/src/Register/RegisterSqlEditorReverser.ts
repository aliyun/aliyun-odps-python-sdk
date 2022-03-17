import { INotebookModel, INotebookTracker } from '@jupyterlab/notebook';
import { reverseAllEditor } from '../Editor/SqlEditor';
import { IChangedArgs } from '@jupyterlab/coreutils';

export const registerSqlEditorReverser = (tracker: INotebookTracker) => {
  const slot = (sender: INotebookModel, args: IChangedArgs<any>) => {
    if (args.name === 'dirty' && !args.newValue) {
      reverseAllEditor(tracker);
    }
  };
  if (tracker.currentWidget?.isAttached) {
    slot(tracker.currentWidget.model, {
      name: 'dirty',
      newValue: false,
      oldValue: false
    });
  }
  tracker.currentChanged.connect(() => {
    if (tracker.currentWidget.model.dirty) {
      // fixed when page already mount, but not ready
      tracker.currentWidget.model.stateChanged.disconnect(slot);
      tracker.currentWidget.model.stateChanged.connect(slot);
    } else {
      slot(tracker.currentWidget.model, {
        name: 'dirty',
        newValue: false,
        oldValue: false
      });
    }
  });
};
