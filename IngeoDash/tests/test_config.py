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
from IngeoDash.config import Config
from IngeoDash.config import CONFIG


def test_Config():
    conf = Config()
    default = dict(store='store',
                   labels='labels',
                   label_header='klass',
                   data='corpus',
                   permanent='permanent',
                   original='original',
                   next='next',
                   next_label='Next',
                   n_value=10,
                   n='N',
                   size='size',
                   progress='progress',
                   download='download',
                   filename='filename',
                   save='save',
                   center='center',
                   upload='upload',
                   lang='lang',
                   n_jobs=1,
                   denseBoW={},
                   db={},
                   username='username',
                   text='text',
                   mem={})
    for k, v in default.items():
        assert v == getattr(conf, k)


def test_Config_mem():
    config = Config()

    config['hola'] = 12
    assert config['hola'] == 12
    assert config.mem['hola'] == 12


def test_Config_call():
    config = Config()
    mem = dict(hola=12)
    xxx = config(mem)
    assert xxx['hola'] == 12
    config['adios'] = 2
    assert 'adios' not in xxx
    xxx['xxx'] = 3
    assert mem['xxx'] == 3


def test_Config_call2():
    mem = CONFIG(dict(label_header='label',
                      text='texto'))
    assert mem.label_header == 'label'
    assert mem.text == 'texto'


def test_CONFIG():
    from IngeoDash.config import CONFIG
    assert isinstance(CONFIG, Config)