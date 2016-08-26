.. _dfjoinunion:

.. code:: python

    from odps.df import DataFrame

.. code:: python

    movies = DataFrame(o.get_table('pyodps_ml_100k_movies'))
    ratings = DataFrame(o.get_table('pyodps_ml_100k_ratings'))

.. code:: python

    movies.dtypes




.. parsed-literal::

    odps.Schema {
      movie_id                            int64       
      title                               string      
      release_date                        string      
      video_release_date                  string      
      imdb_url                            string      
    }



.. code:: python

    ratings.dtypes




.. parsed-literal::

    odps.Schema {
      user_id                     int64     
      movie_id                    int64     
      rating                      int64     
      unix_timestamp              int64     
    }



Join操作
========

DataFrame也支持对两个Collection执行join的操作，如果不指定join的条件，那么DataFrame
API会寻找名字相同的列，并作为join的条件。

.. code:: python

    movies.join(ratings).head(3)




.. raw:: html

    <div style='padding-bottom: 30px; overflow:scroll;'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movie_id</th>
          <th>title</th>
          <th>release_date</th>
          <th>video_release_date</th>
          <th>imdb_url</th>
          <th>user_id</th>
          <th>rating</th>
          <th>unix_timestamp</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>49</td>
          <td>3</td>
          <td>888068877</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>621</td>
          <td>5</td>
          <td>881444887</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>291</td>
          <td>3</td>
          <td>874833936</td>
        </tr>
      </tbody>
    </table>
    </div>



我们也可以显式指定join的条件。有以下几种方式：

.. code:: python

    movies.join(ratings, on='movie_id').head(3)



.. raw:: html

    <div style='padding-bottom: 30px; overflow:scroll;'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movie_id</th>
          <th>title</th>
          <th>release_date</th>
          <th>video_release_date</th>
          <th>imdb_url</th>
          <th>user_id</th>
          <th>rating</th>
          <th>unix_timestamp</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>49</td>
          <td>3</td>
          <td>888068877</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>621</td>
          <td>5</td>
          <td>881444887</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>291</td>
          <td>3</td>
          <td>874833936</td>
        </tr>
      </tbody>
    </table>
    </div>



在join时，on条件两边的字段名称相同时，只会选择一个，其他类型的join则会被重命名。


.. code:: python

    movies.left_join(ratings, on='movie_id').head(3)



.. raw:: html

    <div style='padding-bottom: 30px; overflow:scroll;'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movie_id_x</th>
          <th>title</th>
          <th>release_date</th>
          <th>video_release_date</th>
          <th>imdb_url</th>
          <th>user_id</th>
          <th>movie_id_y</th>
          <th>rating</th>
          <th>unix_timestamp</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>49</td>
          <td>3</td>
          <td>3</td>
          <td>888068877</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>621</td>
          <td>3</td>
          <td>5</td>
          <td>881444887</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>291</td>
          <td>3</td>
          <td>3</td>
          <td>874833936</td>
        </tr>
      </tbody>
    </table>
    </div>



可以看到，\ ``movie_id``\ 被重命名为movie\_id\_x，以及movie\_id\_y，这和\ ``suffixes``\ 参数有关（默认是\ ``('_x', '_y')``\ ），
当遇到重名的列时，就会被重命名为指定的后缀。

.. code:: python

    ratings2 = ratings[ratings.exclude('movie_id'), ratings.movie_id.rename('movie_id2')]
    ratings2.dtypes




.. parsed-literal::

    odps.Schema {
      user_id                     int64     
      rating                      int64     
      unix_timestamp              int64     
      movie_id2                   int64     
    }



.. code:: python

    movies.join(ratings2, on=[('movie_id', 'movie_id2')]).head(3)




.. raw:: html

    <div style='padding-bottom: 30px; overflow:scroll;'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movie_id</th>
          <th>title</th>
          <th>release_date</th>
          <th>video_release_date</th>
          <th>imdb_url</th>
          <th>user_id</th>
          <th>rating</th>
          <th>unix_timestamp</th>
          <th>movie_id2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>49</td>
          <td>3</td>
          <td>888068877</td>
          <td>3</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>621</td>
          <td>5</td>
          <td>881444887</td>
          <td>3</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>291</td>
          <td>3</td>
          <td>874833936</td>
          <td>3</td>
        </tr>
      </tbody>
    </table>
    </div>



也可以直接写等于表达式。

.. code:: python

    movies.join(ratings2, on=[movies.movie_id == ratings2.movie_id2]).head(3)




.. raw:: html

    <div style='padding-bottom: 30px; overflow:scroll;'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movie_id</th>
          <th>title</th>
          <th>release_date</th>
          <th>video_release_date</th>
          <th>imdb_url</th>
          <th>user_id</th>
          <th>rating</th>
          <th>unix_timestamp</th>
          <th>movie_id2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>49</td>
          <td>3</td>
          <td>888068877</td>
          <td>3</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>621</td>
          <td>5</td>
          <td>881444887</td>
          <td>3</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>291</td>
          <td>3</td>
          <td>874833936</td>
          <td>3</td>
        </tr>
      </tbody>
    </table>
    </div>



self-join的时候，可以调用\ ``view``\ 方法，这样就可以分别取字段。

.. code:: python

    movies2 = movies.view()
    movies.join(movies2, movies.movie_id == movies2.movie_id)[movies, movies2.movie_id].head(3)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movie_id_x</th>
          <th>title_x</th>
          <th>release_date_x</th>
          <th>video_release_date_x</th>
          <th>imdb_url_x</th>
          <th>movie_id_y</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>Toy Story (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Toy%20Story%2...</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>GoldenEye (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?GoldenEye%20(...</td>
          <td>2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Four Rooms (1995)</td>
          <td>01-Jan-1995</td>
          <td></td>
          <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
          <td>3</td>
        </tr>
      </tbody>
    </table>
    </div>



除了\ ``join``\ 以外，DataFrame还支持\ ``left_join``\ ，\ ``right_join``\ ，和\ ``outer_join``\ 。

要使用 **mapjoin**\ 也很简单，只需将mapjoin设为True，执行时会对右表做mapjoin操作。

用户也能join分别来自ODPS和pandas的Collection，此时计算会在ODPS上执行。

Union操作
=========

现在有两张表，字段和类型都一致（可以顺序不同），我们可以使用union或者concat来把它们合并成一张表。

.. code:: python

    mov1 = movies[movies.movie_id < 3]['movie_id', 'title']
    mov2 = movies[(movies.movie_id > 3) & (movies.movie_id < 6)]['title', 'movie_id']
    mov1.union(mov2)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movie_id</th>
          <th>title</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>Toy Story (1995)</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>GoldenEye (1995)</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4</td>
          <td>Get Shorty (1995)</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5</td>
          <td>Copycat (1995)</td>
        </tr>
      </tbody>
    </table>
    </div>

用户也能union分别来自ODPS和pandas的Collection，此时计算会在ODPS上执行。
