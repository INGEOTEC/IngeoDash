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
from IngeoDash.app import mock_data, process_manager, download, progress, upload, user
from IngeoDash.annotate import label_column
from IngeoDash.config import Config
from IngeoDash.config import CONFIG
from EvoMSA.tests.test_base import TWEETS
from microtc.utils import tweet_iterator
import base64
import json
import string


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


def test_process_manager_upload():
    mem = CONFIG({})
    D = mock_data()
    _ = [json.dumps(x) for x in D]
    content_str = str(base64.b64encode(bytes('\n'.join(_),
                                       encoding='utf-8')),
                      encoding='utf-8')
    _ = process_manager(mem,
                        mem.upload,
                        None,
                        f'NA,{content_str}')
    info = json.loads(_)
    db = CONFIG.db[info[mem.username]]
    for a, b in zip(db[mem.data], D):
        assert a['text'] == b['text'] and mem.label_header in a


def test_upload():
    mem = CONFIG({CONFIG.username: 'xxx'})
    mem.label_header = 'klass'
    D = list(tweet_iterator(TWEETS))
    CONFIG.db['xxx'] = {mem.permanent: [D[-1]]}
    D1 = [dict(text=x['text']) for x in D[50:]]
    _ = [json.dumps(x) for x in D[:50] + D1]
    content_str = str(base64.b64encode(bytes('\n'.join(_),
                                       encoding='utf-8')),
                      encoding='utf-8')
    content = f'NA,{content_str}'
    upload(mem, content)
    db = CONFIG.db[mem[mem.username]]
    assert len(db[mem.data]) == mem.n_value
    assert len(db[mem.permanent]) == 51
    assert len(db[mem.data]) + len(db[mem.original]) + len(db[mem.permanent]) == len(D) + 1


def test_process_manager_next():
    D = mock_data()[:15]
    table = dict(row=0)
    mem = CONFIG({CONFIG.username: 'xxx',
                  CONFIG.n: CONFIG.n_value})
    CONFIG.db['xxx'] = {mem.data: D[:mem.n_value],
                        mem.original: D[mem.n_value:]}
    db = CONFIG.db['xxx']
    label_column(mem)
    size = len(D)
    _ = process_manager(mem, triggered_id=mem.next)
    assert len(db[mem.permanent]) == mem.n_value
    assert mem[mem.n] == 2 * mem.n_value
    assert len(db[mem.data]) == 5
    _ = process_manager(mem, triggered_id=mem.next)
    assert len(db[mem.data]) == 0
    assert len(db[mem.original]) == 0


def test_download(): 
    D = mock_data()
    mem = CONFIG({CONFIG.username: 'xxx'})
    CONFIG.db['xxx'] = {mem.permanent: D[:11]}
    _ = download(mem, 'tmp.json')
    assert _['filename'] == 'tmp.json'
    assert len(_['content'].split('\n')) == 11


def test_progress():
    mem = Config()
    assert progress(mem) == 0
    mem[mem.size] = 10
    mem[mem.n] = 1
    assert progress(mem) == 10
    