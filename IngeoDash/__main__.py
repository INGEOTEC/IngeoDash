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
from IngeoDash.app import process_manager, download, progress, update_row
from IngeoDash.config import CONFIG
from IngeoDash.annotate import flip_label
from EvoMSA.utils import MODEL_LANG
from dash import dcc, Output, Input, callback, ctx, Dash, State, dash_table, html, Patch
import dash_bootstrap_components as dbc


@callback(
    Output('store', 'data'),
    Input(CONFIG.next, 'n_clicks'),
    Input(CONFIG.upload, 'contents'),
    State(CONFIG.lang, 'value'),
    State('store', 'data'),
    prevent_initial_call=True)
def process_manager_callback(next,
                             content,
                             lang,
                             mem):
    mem = CONFIG(mem)
    return process_manager(mem, ctx.triggered_id,
                           next, content, lang)


@callback(
    Output(CONFIG.center, 'children'),
    Input('store', 'data'),
    prevent_initial_call=True    
)
def table(mem):
    mem = CONFIG(mem)
    data = mem.db[mem[mem.username]][mem.data]
    return create_table(data)


def create_table(data):
    return dash_table.DataTable(data if len(data) else [{}],
                                style_data={'whiteSpace': 'normal',
                                            'textAlign': 'left',
                                            'height': 'auto'},
                                style_header={'fontWeight': 'bold',
                                              'textAlign': 'left'},
                                id=CONFIG.data)    


@callback(Output(CONFIG.progress, 'value'),
          Input('store', 'data'))
def progress_callback(mem):
    mem = CONFIG(mem)
    return progress(mem)


@callback(
    Output(CONFIG.data, 'data'),
    Input(CONFIG.data, 'active_cell'),
    State('store', 'data'),
    prevent_initial_call=True
)
def update_row_callback(table, mem):
    mem = CONFIG(mem)
    return update_row(mem, table)


@callback(Output(CONFIG.download, 'data'),
          Input(CONFIG.save, 'n_clicks'),
          State(CONFIG.filename, 'value'),
          State('store', 'data'),
          prevent_initial_call=True)
def download_callback(_, filename, mem):
    mem = CONFIG(mem)
    return download(mem, filename)


def run():
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],
               suppress_callback_exceptions=True)
    lang = dbc.Select(id=CONFIG.lang, value='es',
                      options=[dict(label=x, value=x) for x in MODEL_LANG])

    download_grp = dbc.InputGroup([dbc.InputGroupText('Filename:'),
                                   dbc.Input(placeholder='output.json',
                                             value='output.json',
                                             type='text',
                                             id=CONFIG.filename),
                                   dbc.Button('Download',
                                              color='success',
                                              id=CONFIG.save)])
    upload = dbc.InputGroup([dbc.InputGroupText('Language:'),
                             lang, dcc.Upload(id=CONFIG.upload, 
                                              children=dbc.Button('Upload'))])
    app.layout = dbc.Container([dcc.Loading(children=dcc.Store('store'),
                                            fullscreen=True), 
                                dcc.Download(id=CONFIG.download),
                                dbc.Row(dbc.Stack([dbc.Progress(value=0,
                                                                id=CONFIG.progress),
                                                   html.Div(id=CONFIG.center,
                                                            children=create_table([{}])),
                                                   dbc.Button('Next', 
                                                              color='primary', 
                                                              id=CONFIG.next,
                                                              n_clicks=0)])),
                                dbc.Row(download_grp),
                                dbc.Row(upload)])
    app.run_server(debug=True)


if __name__ == '__main__':
    run()