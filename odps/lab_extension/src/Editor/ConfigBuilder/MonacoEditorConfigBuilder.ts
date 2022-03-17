import { container } from '../../Container';
import { IThemeManager } from '@jupyterlab/apputils';
import { IEditorConfigBuilder } from './EditorConfigBuilder';
import { MonacoEditorProps } from 'react-monaco-editor/lib/types';

/**
 * @deprecated use LSPEditorConfigBuilder first
 */
export class MonacoEditorConfigBuilder implements IEditorConfigBuilder {
  private config: MonacoEditorProps = {
    language: 'sql',
    options: {
      theme: 'vs-dark',
      readOnly: false,
      minimap: { enabled: true },
      fontSize: 14,
      wordWrap: 'wordWrapColumn',
      wordWrapColumn: 80
    },
    value: ''
  };

  /**
   * set Theme automatically
   * dependents on IThemeManager
   */
  public autoTheme() {
    const theme = container.get<IThemeManager>(IThemeManager);
    this.config.options.theme = theme.isLight(theme.theme) ? 'vs' : 'vs-dark';
    return this;
  }

  /**
   * the Editor uri, make sure uri is unique
   * if you wants multiple editor instance
   * @param uri
   */
  public uri(uri: string) {
    // empty implement
    return this;
  }

  /**
   * the sql content
   * @param content
   */
  public content(content: string) {
    this.config.value = content;
    return this;
  }

  /**
   * generate LSP Url automatically
   * dependents on window.location.host
   */
  public autoLSPUrl() {
    // empty implement
    return this;
  }

  /**
   * build configs
   */
  public build() {
    return this.config;
  }
}
