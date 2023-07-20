# Copyright 2023 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from IngeoDash.app import mock_data, table_next, progress, user, update_row, table_component, table_prev, labels_proportion
from dash.exceptions import PreventUpdate
from IngeoDash.annotate import label_column
from IngeoDash.config import Config
from IngeoDash.config import CONFIG
from EvoMSA.tests.test_base import TWEETS
from microtc.utils import tweet_iterator
import numpy as np


def test_mock_data():
    config = Config()
    D = mock_data()
    assert isinstance(D, list)
    assert isinstance(D[0], dict)
    assert 'text' in D[0]


def test_user():
    mem = CONFIG({})
    username, db = user(mem)
    db['hola'] = 1
    assert username in CONFIG.db
    assert 'hola' in CONFIG.db[username]
    mem = CONFIG({CONFIG.username: username})
    username, db = user(mem)
    assert 'hola' in CONFIG.db[username]


def test_table_next():
    D = mock_data()[:15]
    mem = CONFIG({CONFIG.username: 'xxx'})
    CONFIG.db['xxx'] = {mem.data: D[:mem.n_value],
                        mem.original: D[mem.n_value:]}
    db = CONFIG.db['xxx']
    label_column(mem)
    size = len(D)
    _ = table_next(mem)
    assert len(db[mem.permanent]) == mem.n_value
    assert len(db[mem.data]) == 5
    _ = table_next(mem)
    assert len(db[mem.data]) == 0
    assert len(db[mem.original]) == 0


def test_table_next_no_predict():
    D = list(tweet_iterator(TWEETS))
    for x in D[2*CONFIG.n_value:]:
        x['klass'] = 'NA'
    labels = np.unique([x['klass'] for x in D])    
    mem = CONFIG({CONFIG.username: 'xxx',
                  CONFIG.lang: 'xx',
                  CONFIG.labels: labels.tolist()})
    CONFIG.db['xxx'] = {mem.permanent: D[:mem.n_value],
                        mem.data: D[mem.n_value:2*mem.n_value],
                        mem.original: D[2*mem.n_value:]}
    table_next(mem)


def test_table_prev():
    mem = CONFIG({CONFIG.username: 'xxx', 'n_value': 10})
    CONFIG.db['xxx'] = {mem.permanent: [0] * 11,
                        mem.data: [1] * 10,
                        mem.original: [2] * 10}
    table_prev(mem)
    db = CONFIG.db['xxx']
    assert len(db[mem.permanent]) == 1
    assert db[mem.permanent] == [0]
    assert len(db[mem.data]) ==  10 and db[mem.data] == [0] * 10
    assert len(db[mem.original]) == 20
    assert db[mem.original] == ([1] * 10) + ([2] * 10)
    CONFIG.db['xxx'].update({mem.permanent: []})
    table_prev(mem)
    assert len(db[mem.permanent]) == 0
    CONFIG.db['xxx'].update({mem.permanent: [0] * 5,
                             mem.data: [1] * 10,
                             mem.original: [2] * 10})
    table_prev(mem)    
    assert len(db[mem.permanent]) == 0
    assert len(db[mem.data]) ==  5 and db[mem.data] == [0] * 5


def test_table_component():
    import dash_bootstrap_components as dbc
    element = table_component()
    assert isinstance(element, dbc.Stack)


def test_progress():
    mem = CONFIG({})
    assert progress(mem) == 0
    mem.mem.update({CONFIG.username: 'xxx'})
    CONFIG.db['xxx'] = {mem.data: [None] * 10,
                        mem.original: [None] * 10}    
    assert progress(mem) == 50


def test_progress_active_learning():
    mem = CONFIG({CONFIG.active_learning: True, CONFIG.size: 10})
    assert progress(mem) == 0
    mem.mem.update({CONFIG.username: 'xxx'})
    CONFIG.db['xxx'] = {mem.permanent: [None] * 6,
                        mem.data: [None] * 1,
                        mem.original: [None] * 10}    
    assert progress(mem) == 70


def test_update_row():
    from dash import Patch
    D = mock_data()
    mem = CONFIG({CONFIG.username: 'xxx'})
    CONFIG.db['xxx'] = {mem.data: D[:10]}
    label_column(mem)    
    _ = update_row(mem, dict(row=0))
    assert isinstance(_, Patch)


def test_labels_proportion():
    try:
        labels_proportion(CONFIG)
    except PreventUpdate:
        pass
    _ = {CONFIG.username: 'xxx'}
    mem = CONFIG(_)
    CONFIG.db['xxx'] = dict()
    try:
        labels_proportion(mem)
    except PreventUpdate:
        pass
    CONFIG.db['xxx'].update({mem.permanent: [dict(klass=0)] * 3 + [dict(klass=1)] * 5})
    v = np.array([3, 5])
    v = v / v.sum() * 100
    for ele, p in zip(labels_proportion(mem), v):
        assert ele.value == p
