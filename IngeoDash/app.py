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
from IngeoDash.annotate import flip_label, label_column, store
from IngeoDash.config import CONFIG
from dash.exceptions import PreventUpdate
from dash import Patch
import string
import json
import numpy as np


def mock_data():
    from EvoMSA.tests.test_base import TWEETS
    from microtc.utils import tweet_iterator
    return [{'text': x['text']}
            for x in tweet_iterator(TWEETS) if x['klass'] in ['P', 'N']]


def process_manager(mem,
                    triggered_id=None,
                    next=None):
    if triggered_id == mem.next:
        store(mem)
        db = CONFIG.db[mem[mem.username]]
        init = mem[mem.n]
        data = db[mem.original]
        if len(data):
            rest = data[mem.n_value:]
            data = data[:mem.n_value]
            mem[mem.n] = init + mem.n_value
        else:
            data = []
            rest = []
        db[mem.data] = data
        db[mem.original] = rest
        label_column(mem)        
        return json.dumps(mem.mem)
    raise PreventUpdate


def user(mem):
    try:
        username = mem[mem.username]
    except KeyError:
        for i in range(10):
            cdn = np.array([x for x in string.ascii_uppercase])
            _ = np.random.randint(cdn.shape[0], size=20)
            username = ''.join(cdn[_])
            if username not in CONFIG.db:
                break
    try:
        db = CONFIG.db[username]
    except KeyError:
        db = dict()
        CONFIG.db[username] = db
    return username, db
           

def download(mem, filename):
    db = CONFIG.db[mem[mem.username]]
    permanent = db.get(mem.permanent, list())
    data = db.get(mem.data, list())
    _ = [json.dumps(x) for x in permanent + data]
    return dict(content='\n'.join(_), filename=filename)


def progress(mem):
    if mem.size not in mem:
        return 0
    tot = mem[mem.size]
    if tot == 0:
        return 100
    n = mem[mem.n]
    return np.ceil(100 * n / tot)


def update_row(mem, table):
    data = flip_label(mem, k=table['row'])
    patch = Patch()
    del patch[table['row']]
    patch.insert(table['row'], data)
    return patch
