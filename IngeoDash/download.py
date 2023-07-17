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
import json
import dash_bootstrap_components as dbc
from dash import dcc


def download(mem: Config, filename: str):
    db = CONFIG.db[mem[mem.username]]
    permanent = db.get(mem.permanent, list())
    data = db.get(mem.data, list())
    original = db.get(mem.original, list())
    _ = [json.dumps(x) for x in permanent + data + original]
    return dict(content='\n'.join(_), filename=filename)


def download_component():
    return dbc.InputGroup([dcc.Download(id=CONFIG.download),
                           dbc.InputGroupText('Filename:'),
                           dbc.Input(placeholder='output.json',
                                     value='output.json',
                                     type='text',
                                     id=CONFIG.filename),
                           dbc.Button('Download',
                                      color='success',
                                      id=CONFIG.save)])


if __name__ == '__main__':
    from IngeoDash.__main__ import test_component
    test_component(download_component())