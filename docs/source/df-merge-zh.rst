.. _dfmerge:

数据合并
========

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



Join 操作
----------

DataFrame 也支持对两个 Collection 执行 join 的操作，如果不指定 join 的条件，那么 DataFrame
API会寻找名字相同的列，并作为 join 的条件。

.. code:: python

    >>> movies.join(ratings).head(3)
       movie_id              title  release_date  video_release_date                                           imdb_url  user_id  rating  unix_timestamp
    0         3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...       49       3       888068877
    1         3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...      621       5       881444887
    2         3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...      291       3       874833936

我们也可以显式指定join的条件。有以下几种方式：

.. code:: python

    >>> movies.join(ratings, on='movie_id').head(3)
       movie_id              title  release_date  video_release_date                                           imdb_url  user_id  rating  unix_timestamp
    0         3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...       49       3       888068877
    1         3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...      621       5       881444887
    2         3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...      291       3       874833936

在join时，on条件两边的字段名称相同时，只会选择一个，其他类型的join则会被重命名。


.. code:: python

    >>> movies.left_join(ratings, on='movie_id').head(3)
       movie_id_x              title  release_date  video_release_date                                           imdb_url  user_id  movie_id_y  rating  unix_timestamp
    0           3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...       49           3       3       888068877
    1           3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...      621           3       5       881444887
    2           3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...      291           3       3       874833936

可以看到，\ ``movie_id``\ 被重命名为movie\_id\_x，以及movie\_id\_y，这和\ ``suffixes``\ 参数有关（默认是\ ``('_x', '_y')``\ ），
当遇到重名的列时，就会被重命名为指定的后缀。

.. code:: python

    >>> ratings2 = ratings[ratings.exclude('movie_id'), ratings.movie_id.rename('movie_id2')]
    >>> ratings2.dtypes
    odps.Schema {
      user_id                     int64     
      rating                      int64     
      unix_timestamp              int64     
      movie_id2                   int64     
    }
    >>> movies.join(ratings2, on=[('movie_id', 'movie_id2')]).head(3)
       movie_id              title  release_date  video_release_date                                           imdb_url  user_id  rating  unix_timestamp  movie_id2
    0         3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...       49       3       888068877          3
    1         3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...      621       5       881444887          3
    2         3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...      291       3       874833936          3

也可以直接写等于表达式。

.. code:: python

    >>> movies.join(ratings2, on=[movies.movie_id == ratings2.movie_id2]).head(3)
       movie_id              title  release_date  video_release_date                                           imdb_url  user_id  rating  unix_timestamp  movie_id2
    0         3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...       49       3       888068877          3
    1         3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...      621       5       881444887          3
    2         3  Four Rooms (1995)   01-Jan-1995                      http://us.imdb.com/M/title-exact?Four%20Rooms%...      291       3       874833936          3

self-join的时候，可以调用\ ``view``\ 方法，这样就可以分别取字段。

.. code:: python

    >>> movies2 = movies.view()
    >>> movies.join(movies2, movies.movie_id == movies2.movie_id)[movies, movies2.movie_id.rename('movie_id2')].head(3)
       movie_id            title_x release_date_x video_release_date_x  \
    0         2   GoldenEye (1995)    01-Jan-1995                 True
    1         3  Four Rooms (1995)    01-Jan-1995                 True
    2         4  Get Shorty (1995)    01-Jan-1995                 True

                                              imdb_url_x  movie_id2
    0  http://us.imdb.com/M/title-exact?GoldenEye%20(...          2
    1  http://us.imdb.com/M/title-exact?Four%20Rooms%...          3
    2  http://us.imdb.com/M/title-exact?Get%20Shorty%...          4

除了\ ``join``\ 以外，DataFrame还支持\ ``left_join``\ ，\ ``right_join``\ ，和\ ``outer_join``\ 。在执行上述外连接操作时，
默认会将重名列加上 _x 和 _y 后缀，可通过在 suffixes 参数中传入一个二元 tuple 来自定义后缀。

如果需要在外连接中避免对谓词中相等的列取重复列，可以指定 merge_columns 选项，该选项会自动选择两列中的非空值作为新列的值：

.. code:: python

    >>> movies.left_join(ratings, on='movie_id', merge_columns=True)

要使用 **mapjoin**\ 也很简单，只需将mapjoin设为True，执行时会对右表做mapjoin操作。

用户也能join分别来自ODPS和pandas的Collection，或者join分别来自ODPS和数据库的Collection，此时计算会在ODPS上执行。

Union操作
----------

现在有两张表，字段和类型都一致（可以顺序不同），我们可以使用union或者concat来把它们合并成一张表。

.. code:: python

    >>> mov1 = movies[movies.movie_id < 3]['movie_id', 'title']
    >>> mov2 = movies[(movies.movie_id > 3) & (movies.movie_id < 6)]['title', 'movie_id']
    >>> mov1.union(mov2)
       movie_id              title
    0         1   Toy Story (1995)
    1         2   GoldenEye (1995)
    2         4  Get Shorty (1995)
    3         5     Copycat (1995)

用户也能union分别来自ODPS和pandas的Collection，或者union分别来自ODPS和数据库的Collection，此时计算会在ODPS上执行。
