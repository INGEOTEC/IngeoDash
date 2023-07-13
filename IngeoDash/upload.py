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
from IngeoDash.config import CONFIG
from IngeoDash.app import user, label_column
from EvoMSA import DenseBoW
from EvoMSA.utils import MODEL_LANG
from dash import Output, Input, callback, State, dcc
import dash_bootstrap_components as dbc
import base64
import io
import json


def upload(mem, content, lang='es'):
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    _ = io.StringIO(decoded.decode('utf-8'))
    data = [json.loads(x) for x in _]
    username, db = user(mem)

    original = [x for x in data if mem.label_header not in x]
    permanent = db.get(mem.permanent, list())
    permanent.extend([x for x in data if mem.label_header in x])
    db[mem.data] = original[:mem.n_value]
    db[mem.permanent] = permanent
    db[mem.original] = original[mem.n_value:]
    mem.mem = {mem.n: mem.n_value,
               mem.lang: lang,
               mem.size: len(original), 
               mem.username: username}
    if lang not in mem.denseBoW:
        dense = DenseBoW(lang=lang, voc_size_exponent=15,
                         n_jobs=mem.n_jobs,
                         dataset=False)
        CONFIG.denseBoW[lang] = dense.text_representations
    label_column(mem)
    return json.dumps(mem.mem)


@callback(
    Output('store', 'data', allow_duplicate=True),
    Input(CONFIG.upload, 'contents'),
    State(CONFIG.lang, 'value'),
    State('store', 'data'),
    prevent_initial_call=True
)
def upload_callback(content, lang, mem):
    mem = CONFIG(mem)
    return upload(mem, content, lang)


def upload_component():
    lang = dbc.Select(id=CONFIG.lang, value='es',
                      options=[dict(label=x, value=x) for x in MODEL_LANG])

    upload = dbc.InputGroup([dbc.InputGroupText('Language:'),
                             lang, dcc.Upload(id=CONFIG.upload, 
                                              children=dbc.Button('Upload'))])
    return upload

