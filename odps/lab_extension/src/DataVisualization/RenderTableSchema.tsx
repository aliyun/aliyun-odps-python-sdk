import { Widget } from '@lumino/widgets';
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';
import { Message } from '@lumino/messaging';
import ReactDOM from 'react-dom';
import React from 'react';
import { DataExplorer, Toolbar, Viz } from '@nteract/data-explorer';
import { container } from '../Container';
import { IThemeManager } from '@jupyterlab/apputils';

export class RenderTableSchema extends Widget implements IRenderMime.IRenderer {
  private readonly _mimeType: string;
  private readonly _themeManager: IThemeManager;
  private get isLightTheme() {
    return this._themeManager.isLight(this._themeManager.theme);
  }

  /**
   * Create a new widget for rendering JSON.
   */
  constructor(options: IRenderMime.IRendererOptions) {
    super();
    this._mimeType = options.mimeType;
    this._themeManager = container.get(IThemeManager);
  }

  /**
   * Render JSON into this widget's node.
   */
  public renderModel(model: IRenderMime.IMimeModel): Promise<void> {
    const data = model.data[this._mimeType] || ({} as any);
    return new Promise<void>(resolve => {
      ReactDOM.render(
        <div
          style={{
            background: '#fff'
          }}
        >
          <DataExplorer
            theme={this.isLightTheme ? 'light' : 'dark'}
            data={data}
          >
            <Toolbar />
            <Viz />
          </DataExplorer>
        </div>,
        this.node,
        resolve
      );
    });
  }

  /**
   * Called before the widget is detached from the DOM.
   */
  protected onBeforeDetach(msg: Message): void {
    // Unmount the component so it can tear down.
    ReactDOM.unmountComponentAtNode(this.node);
  }
}
