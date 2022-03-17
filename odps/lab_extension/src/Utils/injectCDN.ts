import async from 'async';

export const injectCDN = () => {
  const url = [
    '//g.alicdn.com/code/lib/react/16.6.1/umd/react.production.min.js',
    '//g.alicdn.com/code/lib/react-dom/16.6.1/umd/react-dom.production.min.js',
    '//alifd.alicdn.com/npm/@alifd/next/1.11.6/next.min.js',
    '//f.alicdn.com/lodash.js/4.17.4/lodash.min.js',
    '//g.alicdn.com/LSP/LSP-Editor/0.4.15/index.js'
  ];
  async.eachOfSeries(url, (item, key, callback) => {
    const ele = document.createElement('script');
    ele.src = item;
    ele.type = 'text/javascript';
    ele.onload = () => callback();
    document.head.appendChild(ele);
  });
};
