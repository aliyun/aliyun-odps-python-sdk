import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';
import { RenderTableSchema } from '../DataVisualization/RenderTableSchema';

export const registerDataVisualization = (
  renderMimeRegistry: IRenderMimeRegistry
) => {
  renderMimeRegistry.addFactory({
    safe: true,
    defaultRank: 0,
    mimeTypes: ['application/vnd.dataresource+json'],
    createRenderer(
      options: IRenderMime.IRendererOptions
    ): IRenderMime.IRenderer {
      return new RenderTableSchema(options);
    }
  });
};
