# encoding: utf-8
# Copyright 1999-2022 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import uuid

from .core import DataFrame
from .. import options
from ..tempobj import register_temp_table
from ..examples import tables
from ..utils import TEMP_TABLE_PREFIX


def _base_example_loader(odps, method, **kwargs):
    roles = kwargs.pop('roles', None)

    table_name = TEMP_TABLE_PREFIX + 'df_' + method + '_' + str(uuid.uuid4()).lower().replace('-', '_')
    getattr(tables, method)(odps, table_name, **kwargs)
    register_temp_table(odps, table_name)
    odps.run_sql('alter table %s set lifecycle %d' % (table_name, options.temp_lifecycle))

    df = DataFrame(odps.get_table(table_name))
    if hasattr(df, 'roles') and roles:
        df = df.roles(**roles)
    return df


def create_ionosphere(odps, tunnel=None):
    """
    Create a PyODPS DataFrame with Johns Hopkins University ionosphere database.

    1. Title: Johns Hopkins University Ionosphere database

    2. Source Information:
       -- Donor: Vince Sigillito (vgs@aplcen.apl.jhu.edu)
       -- Date: 1989
       -- Source: Space Physics Group
                  Applied Physics Laboratory
                  Johns Hopkins University
                  Johns Hopkins Road
                  Laurel, MD 20723

    3. Relevant Information:
       This radar data was collected by a system in Goose Bay, Labrador.  This
       system consists of a phased array of 16 high-frequency antennas with a
       total transmitted power on the order of 6.4 kilowatts.  See the paper
       for more details.  The targets were free electrons in the ionosphere.
       "Good" radar returns are those showing evidence of some type of structure
       in the ionosphere.  "Bad" returns are those that do not; their signals pass
       through the ionosphere.

       Received signals were processed using an autocorrelation function whose
       arguments are the time of a pulse and the pulse number.  There were 17
       pulse numbers for the Goose Bay system.  Instances in this databse are
       described by 2 attributes per pulse number, corresponding to the complex
       values returned by the function resulting from the complex electromagnetic
       signal.

    4. Number of Instances: 351

    5. Number of Attributes: 34 plus the class attribute
       -- All 34 predictor attributes are continuous

    6. Attribute Information:
       -- All 34 are continuous, as described above
       -- The 35th attribute is either "good" or "bad" according to the definition
          summarized above.  This is a binary classification task.

    7. Missing Values: None

    :param odps: ODPS Object Reference
    :param tunnel: ODPS Tunnel Object Reference
    :return: Generated DataFrame
    """
    return _base_example_loader(odps, 'create_ionosphere', tunnel=tunnel, roles={'label': 'class'})


def create_iris(odps, tunnel=None):
    """
    Create a PyODPS DataFrame with iris plants database.

    1. Title: Iris Plants Database

    2. Sources:
         (a) Creator: R.A. Fisher
         (b) Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
         (c) Date: July, 1988

    3. Relevant Information:
       --- This is perhaps the best known database to be found in the pattern
           recognition literature.  Fisher's paper is a classic in the field
           and is referenced frequently to this day.  (See Duda & Hart, for
           example.)  The DataFrame contains 3 classes of 50 instances each,
           where each class refers to a type of iris plant.  One class is
           linearly separable from the other 2; the latter are NOT linearly
           separable from each other.
       --- Predicted attribute: class of iris plant.
       --- This is an exceedingly simple domain.

    4. Number of Instances: 150 (50 in each of three classes)

    5. Number of Attributes: 4 numeric, predictive attributes and the class

    6. Attribute Information:
       1. sepal length in cm
       2. sepal width in cm
       3. petal length in cm
       4. petal width in cm
       5. class:
          -- Iris-setosa
          -- Iris-versicolour
          -- Iris-virginica

    7. Missing Attribute Values: None

    Summary Statistics:

    =============== ===== ===== ===== ===== ==================
                    Min   Max   Mean  SD    Class Correlation
    =============== ===== ===== ===== ===== ==================
    sepal length    4.3   7.9   5.84  0.83  0.7826
    sepal width     2.0   4.4   3.05  0.43  -0.4194
    petal length    1.0   6.9   3.76  1.76  0.9490  (high!)
    petal width     0.1   2.5   1.20  0.76  0.9565  (high!)
    =============== ===== ===== ===== ===== ==================

    8. Class Distribution: 33.3% for each of 3 classes.

    :param odps: ODPS Object Reference
    :param tunnel: ODPS Tunnel Object Reference
    :return: Generated DataFrame
    """
    return _base_example_loader(odps, 'create_iris', tunnel=tunnel, roles={'label': 'category'})
