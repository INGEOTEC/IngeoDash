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
from IngeoDash.config import CONFIG, Config
from dash import dash_table, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import Patch
import string
import json
import numpy as np


def mock_data():
    from EvoMSA.tests.test_base import TWEETS
    from microtc.utils import tweet_iterator
    return [{'text': x['text']}
            for x in tweet_iterator(TWEETS) if x['klass'] in ['P', 'N']]


def table_next(mem: Config, call_next=label_column):
    store(mem)
    db = CONFIG.db[mem[mem.username]]
    data = db[mem.original]
    if len(data):
        rest = data[mem.n_value:]
        data = data[:mem.n_value]
    else:
        data = []
        rest = []
    db[mem.data] = data
    db[mem.original] = rest
    if call_next is not None:
        call_next(mem)       
    return json.dumps(mem.mem)


def table_prev(mem: Config):
    db = CONFIG.db[mem[mem.username]]
    nvalue = mem.n_value
    if mem.permanent not in db:
        return json.dumps(mem.mem)
    permanent = db[mem.permanent]
    if len(permanent) == 0:
        return json.dumps(mem.mem)
    nvalue = nvalue if len(permanent) >= nvalue else len(permanent)
    data = db[mem.data] if mem.data in db else []
    original = db[mem.original] if mem.original in db else []
    db[mem.original] = data + original
    db[mem.data] = permanent[-nvalue:]
    db[mem.permanent] = permanent[:-nvalue]
    return json.dumps(mem.mem)


def table(mem: Config):
    if mem.username in mem:
        data = CONFIG.db[mem[mem.username]][mem.data]
    else:
        data = [{}]
    data = [{k: f'{v}'for k, v in x.items()} for x in data]
    return dash_table.DataTable(data if len(data) else [{}],
                                style_data={'whiteSpace': 'normal',
                                            'textAlign': 'left',
                                            'height': 'auto'},
                                style_header={'fontWeight': 'bold',
                                              'textAlign': 'left'},
                                id=CONFIG.data)


def table_component():
    buttons = dbc.ButtonGroup([dbc.Button('Previous',
                                 color='secondary',
                                 id=CONFIG.prev),
                      dbc.Button('Next', 
                                 color='primary', 
                                 id=CONFIG.next,
                                 n_clicks=0)])
    labels_proportion = dbc.Progress(id=CONFIG.labels_proportion)
    return dbc.Stack([dbc.Progress(value=0, label='Progress',
                                   id=CONFIG.progress,
                                   color='info'),
                      labels_proportion,
                      html.Div(id=CONFIG.center,
                               children=table(CONFIG)),
                      buttons])


def user(mem: Config):
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
           

def progress(mem: Config):
    if mem.username not in mem:
        return 0
    db = CONFIG.db[mem[mem.username]]        
    data = len(db[mem.data]) if mem.data in db else 0
    ori = len(db[mem.original]) if mem.original in db else 0
    tot = mem[mem.size] if mem.size in mem else data + ori 
    if mem.active_learning in mem and mem[mem.active_learning]:
        num = len(db[mem.permanent]) if mem.permanent in db else 0
        num += data
        return np.ceil(100 * num / tot)
    if tot == 0:
        return 0
    return np.ceil(100 * (tot - ori) / tot)


def update_row(mem: Config, table: dict):
    data = flip_label(mem, k=table['row'])
    patch = Patch()
    del patch[table['row']]
    patch.insert(table['row'], {k: f'{v}'for k, v in data.items()})
    return patch


def labels_proportion(mem: Config):
    if mem.username not in mem:
        raise PreventUpdate
    db = CONFIG.db[mem[mem.username]]
    if mem.permanent not in db:
        raise PreventUpdate
    D = db[mem.permanent]
    labels, cnt = np.unique([x[mem.label_header] for x in D], return_counts=True)
    proportions = 100 * cnt / cnt.sum()
    colors = ['primary', 'secondary', 'success']
    output = []
    for label, amount, prop, k in zip(labels, cnt, proportions,
                                      range(len(labels))):
        _ = dbc.Progress(value=prop, color=colors[k % len(colors)],
                         label=f'{label} (#{amount})', bar=True)
        output.append(_)
    return output


if __name__ == '__main__':
    from IngeoDash.__main__ import test_component
    test_component(table_component())