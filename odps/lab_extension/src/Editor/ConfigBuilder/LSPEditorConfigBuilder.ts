import { container } from '../../Container';
import { IThemeManager } from '@jupyterlab/apputils';
import { IEditorConfigBuilder } from './EditorConfigBuilder';
import { isInner } from '../../Utils/isInner';

export class LSPEditorConfigBuilder implements IEditorConfigBuilder {
  private config = {
    language: 'odps',
    uri: '123',
    editorOptions: {
      theme: 'vs-dark',
      readOnly: false,
      minimap: { enabled: true },
      fontSize: 14,
      wordWrap: 'wordWrapColumn',
      wordWrapColumn: 80
    },
    useLsp: true,
    lspOptions: {
      wsUrl: 'wss://lsp-cn-shanghai.data.aliyun.com/lsp',
      projectInfo: {
        projectId: 123,
        projectIdentifier: 'ots_etl'
      },
      settings: {
        autoComplete: [
          'keyword',
          'white',
          'snippet',
          'project',
          'table',
          'column'
        ],
        codeStyle: 1,
        faultCheck: true
      }
    },
    content: ''
  };

  /**
   * set Theme automatically
   * dependents on IThemeManager
   */
  public autoTheme() {
    const theme = container.get<IThemeManager>(IThemeManager);
    this.config.editorOptions.theme = theme.isLight(theme.theme)
      ? 'vs'
      : 'vs-dark';
    return this;
  }

  /**
   * the Editor uri, make sure uri is unique
   * if you wants multiple editor instance
   * @param uri
   */
  public uri(uri: string) {
    this.config.uri = uri;
    return this;
  }

  /**
   * the sql content
   * @param content
   */
  public content(content: string) {
    this.config.content = content;
    return this;
  }

  /**
   * generate LSP Url automatically
   * dependents on window.location.host
   */
  public autoLSPUrl() {
    const regions = [
      'cn-beijing',
      'cn-shanghai',
      'cn-hangzhou',
      'cn-shenzhen',
      'ap-southeast'
    ];
    const host = window.location.host;

    for (let i = 0; i < regions.length; i = i + 1) {
      if (host.includes(regions[i])) {
        this.config.lspOptions.wsUrl = `wss://lsp-${regions[i]}.data.aliyun.com/lsp`;
        return this;
      }
    }


    return this;
  }

  /**
   * build configs
   */
  public build() {
    return {
      options: this.config
    };
  }
}
