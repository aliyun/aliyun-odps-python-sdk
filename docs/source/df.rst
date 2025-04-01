.. _df:

**********
DataFrame
**********


PyODPS 提供了 DataFrame API，它提供了类似 pandas 的接口，但是能充分利用 ODPS 的计算能力；
同时能在本地使用同样的接口，用 pandas 进行计算。

.. note::

   PyODPS DataFrame 未来将停止维护。对于新项目，建议使用 `MaxFrame
   <https://maxframe.readthedocs.io/en/latest/index.html>`_\ 。

   PyODPS DataFrame 尽管看起来和 pandas 形似，但并不是 pandas。pandas 的功能，例如完整的
   Series 支持、Index 支持、按行读取数据、多 DataFrame 按 iloc 横向合并等，PyODPS DataFrame
   并不支持。因而使用前请参考文档确定你的写法是否被支持。



.. toctree::
   :maxdepth: 1

   df-quickstart
   df-basic
   df-element
   df-agg
   df-sort-distinct-apply
   df-merge
   df-window
   df-plot
   df-debug-instruction
