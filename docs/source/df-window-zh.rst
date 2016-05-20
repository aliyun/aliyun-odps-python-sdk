.. _dfwindow:

.. code:: python

    from odps.df import DataFrame

.. code:: python

    iris = DataFrame(o.get_table('pyodps_iris'))

窗口函数
========

DataFrame API也支持使用窗口函数：

.. code:: python

    grouped = iris.groupby('name')
    grouped.mutate(grouped.sepallength.cumsum(), grouped.sort('sepallength').row_number()).head(10)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>sepallength_sum</th>
          <th>row_number</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Iris-setosa</td>
          <td>250.3</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-setosa</td>
          <td>250.3</td>
          <td>2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-setosa</td>
          <td>250.3</td>
          <td>3</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Iris-setosa</td>
          <td>250.3</td>
          <td>4</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Iris-setosa</td>
          <td>250.3</td>
          <td>5</td>
        </tr>
        <tr>
          <th>5</th>
          <td>Iris-setosa</td>
          <td>250.3</td>
          <td>6</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Iris-setosa</td>
          <td>250.3</td>
          <td>7</td>
        </tr>
        <tr>
          <th>7</th>
          <td>Iris-setosa</td>
          <td>250.3</td>
          <td>8</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Iris-setosa</td>
          <td>250.3</td>
          <td>9</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Iris-setosa</td>
          <td>250.3</td>
          <td>10</td>
        </tr>
      </tbody>
    </table>
    </div>



窗口函数可以使用在列选择中：

.. code:: python

    iris['name', 'sepallength', iris.groupby('name').sort('sepallength').sepallength.cumcount()].head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>sepallength</th>
          <th>sepallength_count</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Iris-setosa</td>
          <td>4.3</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-setosa</td>
          <td>4.4</td>
          <td>2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-setosa</td>
          <td>4.4</td>
          <td>3</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Iris-setosa</td>
          <td>4.4</td>
          <td>4</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Iris-setosa</td>
          <td>4.5</td>
          <td>5</td>
        </tr>
      </tbody>
    </table>
    </div>



窗口函数按标量聚合时，和分组聚合的处理方式一致。


.. code:: python

    from odps.df import Scalar
    iris.groupby(Scalar(1)).sort('sepallength').sepallength.cumcount()



DataFrame API支持的窗口函数包括：

.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <tr>
        <th>窗口函数</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>cumsum</td>
        <td></td>
      </tr>
      <tr>
        <td>cummean</td>
        <td></td>
      </tr>
      <tr>
        <td>cummedian</td>
        <td></td>
      </tr>
      <tr>
        <td>cumstd</td>
        <td></td>
      </tr>
      <tr>
        <td>cummax</td>
        <td></td>
      </tr>
      <tr>
        <td>cummin</td>
        <td></td>
      </tr>
      <tr>
        <td>cumcount</td>
        <td></td>
      </tr>
      <tr>
        <td>lag</td>
        <td>按偏移量取当前行之前第几行的值，如当前行号为rn，则取行号为rn-offset的值。</td>
      </tr>
      <tr>
        <td>lead</td>
        <td>按偏移量取当前行之后第几行的值，如当前行号为rn则取行号为rn+offset的值。</td>
      </tr>
      <tr>
        <td>rank</td>
        <td>计算排名</td>
      </tr>
      <tr>
        <td>dense_rank</td>
        <td>计算连续排名</td>
      </tr>
      <tr>
        <td>percent_rank</td>
        <td>计算一组数据中某行的相对排名</td>
      </tr>
      <tr>
        <td>row_number</td>
        <td>计算行号，从1开始</td>
      </tr>
    </table>
    </div>
