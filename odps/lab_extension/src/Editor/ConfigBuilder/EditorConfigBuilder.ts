export interface IEditorConfigBuilder {
  /**
   * set Theme automatically
   * dependents on IThemeManager
   */
  autoTheme(): this;

  /**
   * the Editor uri, make sure uri is unique
   * if you wants multiple editor instance
   * @param uri
   */
  uri(uri: string): this;

  /**
   * the sql content
   * @param content
   */
  content(content: string): this;

  /**
   * generate LSP Url automatically
   * dependents on window.location.host
   */
  autoLSPUrl(): this;

  /**
   * build configs
   */
  build(): object;
}
