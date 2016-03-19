.. _pai_customize:

===================
引入私有 XFlow 算法
===================

PAI SDK 允许用户引入私有 XFlow 算法，新引入的算法和 PAI SDK 固有的算法一样，可以利用 DataSet 等内部对象。下面的代码演示了如何
创建一个自己的 NaiveBayes 算法引用：

.. code-block:: python

    import sys
    from odps.pai.algorithms import *
    from odps.pai.algorithms.loader import load_classifiers

    algo_def = XflowAlgorithmDef('MyNaiveBayes', project='algo_public', xflow_name='NaiveBayes')

    algo_def.add_port(PortDef.build_data_input()).add_port(PortDef.build_model_output())

    algo_def.add_param(ParamDef.build_input_table()).add_param(ParamDef.build_input_partitions())
    algo_def.add_param(ParamDef.build_model_name())
    algo_def.add_param(ParamDef.build_feature_col_names())
    algo_def.add_param(ParamDef.build_label_col_name())

    load_classifiers(algo_def, sys.modules[__name__])

此后，用户即可在当前 module 中利用 MyNaiveBayes 的类名调用 NaiveBayes 算法。如果载入的方法为 load_process_algorithm 而非
load_classifiers，则生成的算法类为具备 transform 方法的算法。

关于引入私有 XFlow 算法的详细信息可参考 :ref:`API 文档 <pai_api_customization>`。
