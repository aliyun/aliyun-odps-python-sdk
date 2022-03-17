import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { INotebookTracker } from '@jupyterlab/notebook';
import { registerSelectCellType } from './Register/RegisterSelectCellType';
import { registerSqlEditorReverser } from './Register/RegisterSqlEditorReverser';
import { container } from './Container';
import { registerDataVisualization } from './Register/RegisterDataVisualization';
import { injectCDN } from './Utils/injectCDN';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { IThemeManager } from '@jupyterlab/apputils';

/**
 * Initialization data for the pyodps-lab-extension extension.
 */
const extension: JupyterFrontEndPlugin<void> = {
  id: 'pyodps-lab-extension:plugin',
  autoStart: true,
  requires: [INotebookTracker, IThemeManager, IRenderMimeRegistry],
  activate: (
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    themeManager: IThemeManager,
    renderMimeRegistry: IRenderMimeRegistry
  ) => {
    container.set(INotebookTracker, tracker);
    container.set(IThemeManager, themeManager);
    container.set(IRenderMimeRegistry, renderMimeRegistry);
    injectCDN();
    registerDataVisualization(renderMimeRegistry);
    registerSelectCellType(tracker);
    registerSqlEditorReverser(tracker);
    console.log('JupyterLab extension pyodps-lab-extension is activated!');
  }
};

export default extension;
