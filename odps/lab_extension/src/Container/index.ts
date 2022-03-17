import { Token } from '@lumino/coreutils';

export class Container {
  private map = new WeakMap<Token<any>, any>();

  public set(token: Token<any>, impl: any) {
    this.map.set(token, impl);
  }

  public get<T>(token: Token<T>): T {
    if (this.map.has(token)) {
      return this.map.get(token);
    } else {
      throw ReferenceError();
    }
  }
}

export const container = new Container();
