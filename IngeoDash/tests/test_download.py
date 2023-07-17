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
from IngeoDash.app import mock_data
from IngeoDash.download import download, download_component
from IngeoDash.config import CONFIG
import json
import io


def test_download(): 
    mem = CONFIG({CONFIG.username: 'xxx'})
    CONFIG.db['xxx'] = {mem.permanent: [0] * 3,
                        mem.data: [1, 1],
                        mem.original: [2] * 4}
    _ = download(mem, 'tmp.json')
    assert _['filename'] == 'tmp.json'
    _ = io.StringIO(_['content'])
    D = [json.loads(x) for x in _]    
    assert D == [0, 0, 0, 1, 1, 2, 2, 2, 2]


def test_download_component():
    import dash_bootstrap_components as dbc
    element = download_component()
    assert isinstance(element, dbc.InputGroup)