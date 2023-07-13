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
from IngeoDash.app import mock_data, table_next, download, progress, user, update_row, download_component, table_component
from IngeoDash.annotate import label_column
from IngeoDash.config import Config
from IngeoDash.config import CONFIG
from EvoMSA.tests.test_base import TWEETS


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
    mem = CONFIG({CONFIG.username: 'xxx',
                  CONFIG.n: CONFIG.n_value})
    CONFIG.db['xxx'] = {mem.data: D[:mem.n_value],
                        mem.original: D[mem.n_value:]}
    db = CONFIG.db['xxx']
    label_column(mem)
    size = len(D)
    _ = table_next(mem)
    assert len(db[mem.permanent]) == mem.n_value
    assert mem[mem.n] == 2 * mem.n_value
    assert len(db[mem.data]) == 5
    _ = table_next(mem)
    assert len(db[mem.data]) == 0
    assert len(db[mem.original]) == 0


def test_download(): 
    D = mock_data()
    mem = CONFIG({CONFIG.username: 'xxx'})
    CONFIG.db['xxx'] = {mem.permanent: D[:11]}
    _ = download(mem, 'tmp.json')
    assert _['filename'] == 'tmp.json'
    assert len(_['content'].split('\n')) == 11


def test_download_component():
    import dash_bootstrap_components as dbc
    element = download_component()
    assert isinstance(element, dbc.InputGroup)


def test_table_component():
    import dash_bootstrap_components as dbc
    element = table_component()
    assert isinstance(element, dbc.Stack)


def test_progress():
    mem = Config()
    assert progress(mem) == 0
    mem[mem.size] = 10
    mem[mem.n] = 1
    assert progress(mem) == 10


def test_update_row():
    from dash import Patch
    D = mock_data()
    mem = CONFIG({CONFIG.username: 'xxx'})
    CONFIG.db['xxx'] = {mem.data: D[:10]}
    label_column(mem)    
    _ = update_row(mem, dict(row=0))
    assert isinstance(_, Patch)    