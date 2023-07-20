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
from IngeoDash.config import CONFIG, Config
from IngeoDash.app import user, table_next
from IngeoDash.annotate import has_label
from EvoMSA.utils import MODEL_LANG
from dash import dcc, html, callback, Output, Input
import numpy as np
import dash_bootstrap_components as dbc
import random
import base64
import json
import io


def read_json(mem: Config, data):
    _ = io.StringIO(data.decode('utf-8'))
    return [json.loads(x) for x in _]


def upload(mem: Config, content, lang='es', 
           type='json', text='text', 
           label='klass', n_value=CONFIG.n_value,
           shuffle=False, size=None, active_learning=False,
           call_next=table_next):
    mem.mem.update(dict(label_header=label, text=text, n_value=n_value))
    if active_learning:
        mem.mem.update({mem.active_learning: active_learning, mem.size: size})
    mem.label_header = label
    mem.text = text
    mem.n_value = n_value
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    data = globals()[f'read_{type}'](mem, decoded)
    if shuffle:
        random.shuffle(data)
    username, db = user(mem)
    labels = np.unique([x[mem.label_header]
                        for x in data if has_label(mem, x)])
    permanent = db.get(mem.permanent, list())    
    if labels.shape[0] > 1:
        original = [x for x in data if not has_label(mem, x)]
        permanent.extend([x for x in data if has_label(mem, x)])
    else:
        original = data
    db[mem.permanent] = permanent
    db[mem.original] = original
    size = len(data) if not active_learning else size
    mem.mem.update({mem.lang: lang,
                    mem.size: size,
                    mem.username: username})
    if call_next is not None:
        call_next(mem)
    return json.dumps(mem.mem)


@callback(
    Output(CONFIG.size, 'disabled', allow_duplicate=True),
    Input(CONFIG.checklist, 'value'),
    prevent_initial_call=True,
)
def active_learning_callback(checklist):
    checklist = checklist if checklist is not None else []
    if CONFIG.active_learning in checklist:
        return False
    return True


def upload_component():
    langs = {'ar': 'Arabic', 'ca': 'Catalan', 'de': 'German', 'en': 'English',
             'es': 'Spanish', 'fr': 'French', 'hi': 'Hindi', 'in': 'Indonesian',
             'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean', 'nl': 'Dutch',
             'pl': 'Polish', 'pt': 'Portuguese', 'ru': 'Russian', 'tl': 'Tagalog',
             'tr': 'Turkish', 'zh': 'Chinese'}
    
    lang_grp = dbc.InputGroup([dbc.InputGroupText('Language:'),
                               dbc.Select(id=CONFIG.lang, value='es',
                                          options=[dict(label=langs.get(x, x),
                                                        value=x)
                                                   for x in MODEL_LANG])])

    data_grp = dbc.InputGroup([dbc.InputGroupText('Text Column:'),
                               dcc.Input(id=CONFIG.text,
                                         value='text',
                                         type='text'),
                               dbc.InputGroupText('Text Label:'),
                               dcc.Input(id=CONFIG.label_header,
                                         value='klass',
                                         type='text'),
                               dbc.InputGroupText('Batch Size:'),
                               dcc.Input(id=CONFIG.batch_size,
                                         value=10,
                                         type='number'),
                               dbc.Checklist(id=CONFIG.checklist,
                                             options=[dict(label='Shuffle', value=CONFIG.shuffle),
                                                      dict(label='Active Learning',
                                                           value=CONFIG.active_learning)],
                                             switch=True)])
    al_grp = dbc.InputGroup([dbc.InputGroupText('Number of labeled elements:'),
                             dcc.Input(id=CONFIG.size,
                                       value=1000,
                                       type='number',
                                       disabled=True)])
    upload_button = dbc.Button(dcc.Upload(id=CONFIG.upload,
                                          children=html.Div('Drop or Select File')))
    return dbc.Col(dbc.Stack([lang_grp, data_grp,
                              al_grp,
                              upload_button]))


if __name__ == '__main__':
    from IngeoDash.__main__ import test_component
    test_component(upload_component())
    # from dash import Dash
    # component = upload_component()
    # app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],
    #            suppress_callback_exceptions=True)
    # app.layout = dbc.Container([dbc.Row(component)])
    # app.run_server(debug=True)    