# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import os
import h5py
import numpy as np
import json
import plotly.graph_objects as go
from contextlib import contextmanager
import time

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

__data_dir__ = r'Y:\Data'
#__data_dir__ = '/Users/jerome/Documents/Code/sandbox'

@contextmanager
def h5file(filename, mode, **kwargs):
    retry = 10
    while retry:
        try:
            resource = h5py.File(filename, mode, **kwargs)
            break
        except OSError:
            print(f"Unable to open {filename}, retrying...")
            time.sleep(1)
            retry -= 1
    else:
        raise Exception(f"Unable to open {filename} after 10 retries.")
    try:
        yield resource
    finally:
        resource.close()

def listdir(thedir):
    if thedir and os.path.isdir(thedir):
        res = [ os.path.join(thedir, name) for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name)) ]
        res.sort(reverse=True)
        return res
    return []

def listfiles(thedir):
    if thedir and os.path.isdir(thedir):
        return [ os.path.join(thedir, name) for name in os.listdir(thedir) if os.path.isfile(os.path.join(thedir, name)) ]
    return []


app.layout = html.Div([
    html.H1(children='HDF5 Viewer'),
    html.Button('Refresh', id='refresh'),
    html.Div([
    html.Label('Year'),
    dcc.Dropdown(
        id='dropdown-year',
        options = [{'label': os.path.basename(d), 'value': d} for d in listdir(__data_dir__)],
        value = listdir(__data_dir__)[0],
        multi=False,
        clearable=False,
    ),
    html.Label('Day'),
    dcc.Dropdown(
        id='dropdown-day',
        multi=False,
    ),
    ], style={'columnCount': 2}),
    dcc.RadioItems(
    id='radio-plottype',
    options=[
        {'label': 'Line plot', 'value': 'line'},
        {'label': '2D map', 'value': 'map'},
    ],
    value='line',
    labelStyle={'display': 'inline-block'}
    ),  
    html.Label('File'),
        dcc.Dropdown(
            id='dropdown-file',
            multi=True,
    ),
    html.Label('Dataset'),
    dcc.Dropdown(
        id='dropdown-dataset',
        multi=True
    ),
    html.Div([
    html.Label('X variable'),
    dcc.Dropdown(
        id='dropdown-x',
        multi=False
    ),
    html.Label('Y variable'),
    dcc.Dropdown(
        id='dropdown-y',
        multi=False
    ),
    html.Label('Map line index'),
    dcc.Dropdown(
        id='dropdown-z',
        multi=False
    ),
    ], style={'columnCount': 3}),
    dcc.Graph(id="main-graph"),
    dcc.ConfirmDialogProvider(
        children=html.Button(
            'Code',
        ),
        id='clipboard',
        message=''
    ),
    html.H3(id='attrs'),
    dash_table.DataTable(id='table'),
    html.H3('')
])

@app.callback(Output('dropdown-year', 'options'),
              Output('dropdown-year', 'value'),
              Input('refresh', 'n_clicks'),
              State('dropdown-year', 'value'))
def refresh(n_clicks, year_value):
    return [{'label': os.path.basename(d), 'value': d} for d in listdir(__data_dir__)], year_value

@app.callback(Output('dropdown-day', 'options'),
              Output('dropdown-day', 'value'),
              Input('dropdown-year', 'value'),
              State('dropdown-day', 'value')
              )
def update_day(year_dir, day_value):
    options = [{'label': os.path.basename(d), 'value': d} for d in listdir(year_dir)]
    vals = [x['value'] for x in options]
    return options, day_value if day_value in vals else None

@app.callback(Output('dropdown-file', 'multi'),
               Output('dropdown-dataset', 'disabled'),
               Output('dropdown-file', 'value'),
               Output('dropdown-z', 'disabled'),
               Output('dropdown-file', 'options'),
               Input('dropdown-day', 'value'),
               Input('radio-plottype', 'value'),
               State('dropdown-file', 'value')
              )
def update_file(day_dir, val, prev_file):
    options = [{'label': os.path.basename(f), 'value': f} for f in listfiles(day_dir) if f.endswith('.h5')]
    vals = [x['value'] for x in options]
    if not type(prev_file) is list:
        prev_file = [prev_file]
    prev_file = [x for x in prev_file if x in vals]
    if val=='line':
        return True, False, prev_file, True, options
    else:
        return False, True, prev_file[0] if prev_file else None, False, options

@app.callback(Output('dropdown-dataset', 'options'),
              Output('dropdown-dataset', 'value'),
              Input('dropdown-file', 'value'),
              State('dropdown-dataset', 'value'),
              State('radio-plottype', 'value')
              )
def update_dataset(files, prev_dataset, plot_type):
    if plot_type=='line':
        if files is None or len(files)==0 or files==[None]:
            return [], []
        file_dataset = []
        for f in files:    
            with h5file(f,'r') as hf:
                file_dataset.extend([ (f,k) for k in hf.keys() ])
        if len(files)==1:
            options = [{'label': ds, 'value': json.dumps((f,ds))} for f,ds in file_dataset]
        else:
            options = [{'label': '{0}:{1}'.format(os.path.splitext(os.path.basename(f))[0],ds), 'value': json.dumps((f,ds))} for f,ds in file_dataset]
        new_datasets = [ds['value'] for ds in options]
        value = [p for p in prev_dataset if p in new_datasets]
        return options, value
    else:
        return [], []


@app.callback(Output('dropdown-x','options'),
              Output('dropdown-y','options'),
              Output('dropdown-z','options'),
              Output('dropdown-x','value'), 
              Output('dropdown-y','value'),
              Output('dropdown-z','value'),
              Input('dropdown-dataset', 'value'),
              State('dropdown-x', 'value'),
              State('dropdown-y', 'value'),
              State('dropdown-z', 'value'),
              State('radio-plottype', 'value'),
              State('dropdown-file', 'value')
              )
def update_vars(file_dataset, prev_x, prev_y, prev_z, plot_type, file):
    if plot_type=='line':
        if file_dataset is None or len(file_dataset)==0:
            return [], [], [], None, None, None
        file_dataset = [json.loads(f) for f in file_dataset]
        v = []
        files = list(set([ f for f,ds in file_dataset]))
        for f1 in files:
            with h5file(f1,'r') as hf:
                for f2,ds in file_dataset:
                    if f1==f2:
                        v.append(set(hf[ds].dtype.names))
        if len(v)==0:
            return [], [], [], None, None, None
        v = set.intersection(*v)
        options = [{'label': x, 'value': x} for x in v]
        new_options = [o['value'] for o in options]
        value_x = prev_x if prev_x in new_options else None
        value_y = prev_y if prev_y in new_options else None
        return options, options, [], value_x, value_y, None
    else:
        if file is not None:
            with h5file(file,'r') as hf:
                ds0 = list(hf.keys())[0]
                options = [{'label': x, 'value': x} for x in hf[ds0].dtype.names]
                map_options = [{'label': x, 'value': x} for x in hf[ds0].attrs.keys()]
            new_options = [o['value'] for o in options]
            new_map_options = [o['value'] for o in map_options]
            value_x = prev_x if prev_x in new_options else None
            value_y = prev_y if prev_y in new_options else None
            value_z = prev_z if prev_z in new_map_options else None
            return options, options, map_options, value_x, value_y, value_z
        else:
            return [], [], [], None, None, None


@app.callback(Output('main-graph','figure'),
              Output('clipboard', 'message'),
              Input('dropdown-x', 'value'),
              Input('dropdown-y', 'value'),
              Input('dropdown-z', 'value'),
              State('dropdown-dataset', 'value'),
              State('radio-plottype', 'value'),
              State('dropdown-file', 'value')
              )
def update_fig(x, y, z, file_dataset, plot_type, file):
    if plot_type=='line':
        if file_dataset is None or len(file_dataset)==0 or y is None:
            return {'data':[{'x': [], 'y': [], 'type': 'line', 'name': None}] , 'layout' : {'title' : '' , 'xaxis' : {'title': ''}, 'yaxis' : { 'title':  ''}}}, ""
        file_dataset = [json.loads(f) for f in file_dataset]
        data = []
        files = list(set([ f for f,ds in file_dataset]))
        cmd_plot = []
        cmd = ""
        for f1 in files:
            with h5file(f1,'r') as hf:
                cmd += f"with h5py.File(r'{f1}','r') as hf:\n"
                for f2,ds in file_dataset:
                    if f1==f2:
                        name = ds if len(files)==1 else '{0} {1}'.format(os.path.splitext(os.path.basename(f1))[0],ds)
                        cmd_name = ds if len(files)==1 else '{0}_{1}'.format(os.path.splitext(os.path.basename(f1))[0],ds)
                        cmd += f"    {cmd_name}_{y} = array(hf['{ds}']['{y}'])\n"
                        cmd += f"    {cmd_name}_x = arange(len({cmd_name}_{y}))\n" if x is None else f"    {cmd_name}_{x} = array(hf['{ds}']['{x}'])\n"
                        if x is None:
                            cmd_plot.append({'x': f"{cmd_name}_x", 'y': f"{cmd_name}_{y}", 'label': f"{name}"})
                            data.append( {'x': np.arange(len(hf[ds][y])), 'y': np.array(hf[ds][y]), 'type': 'line', 'name': name} )
                        else:
                            cmd_plot.append({'x': f"{cmd_name}_{x}", 'y': f"{cmd_name}_{y}", 'label': f"{name}"})
                            data.append( {'x': np.array(hf[ds][x]), 'y': np.array(hf[ds][y]), 'type': 'line', 'name': name} )
        cmd += "fig,ax = subplots()\n"
        for plot in cmd_plot:
            cmd += "ax.plot({x},{y},label='{label}')\n".format(**plot)
        title =  os.path.relpath(files[0],__data_dir__) + ' ' + ' '.join([os.path.basename(f) for f in files[1:]])
        #cmd += f"ax.set_xlabel('{x}')\nax.set_ylabel('{y}')\nlegend()\nax.set_title(r'{title}')\nfig.tight_layout()"
        cmd += f"ax.set_ylabel('{y}')\nlegend()" if x is None else f"ax.set_xlabel('{x}')\nax.set_ylabel('{y}')\nlegend()"
        ret = {'data':data , 'layout' : {'title' : title, 'yaxis' : { 'title':  y}}} if x is None else {'data':data , 'layout' : {'title' : title , 'xaxis' : {'title': x}, 'yaxis' : { 'title':  y}}}
        return ret, cmd
    else:
        if file is None or y is None or z is None:
            return {'data':[{'x': [], 'y': [], 'type': 'line', 'name': None}] , 'layout' : {'title' : '' , 'xaxis' : {'title': ''}, 'yaxis' : { 'title':  ''}}}, ""
        else:
            with h5file(file,'r') as hf:
                l = len(hf)
                k = list(hf.keys())
                ds0 = k[0]
                y_ = np.array(hf[ds0][y])
                xdata = np.arange(len(y_)) if x is None else np.array(hf[ds0][x])
                ydata = np.empty((l,len(y_)),dtype=y_.dtype)
                zdata = np.empty(l)
                for i in range(l):
                    ydata[i,:] = hf[k[i]][y]
                    zdata[i] = hf[k[i]].attrs[z]
            title =  os.path.relpath(file,__data_dir__) 
            fname = os.path.splitext(os.path.basename(file))[0]
            fig = go.Figure(data=go.Heatmap(x=xdata, y=zdata, z=ydata, type = 'heatmap', colorscale = 'Viridis', colorbar={"title": y}))
            if x is None:
                cmd = f"fig,ax = subplots()\n{fname}=loadh5(r'{file}')\nm=ax.pcolormesh(arange(len({fname}.{y}[0])),{fname}.{z},{fname}.{y},shading='nearest')\n"
                cmd += f"ax.set_ylabel('{z}')\ncolorbar(m, label='{y}')"
                fig.update_layout(title=title , yaxis_title=z)
            else:
                cmd = f"fig,ax = subplots()\n{fname}=loadh5(r'{file}')\nm=ax.pcolormesh({fname}.{x}[0],{fname}.{z},{fname}.{y},shading='nearest')\n"
                cmd += f"ax.set_xlabel('{x}')\nax.set_ylabel('{z}')\ncolorbar(m, label='{y}')"
                fig.update_layout(title=title , xaxis_title=x, yaxis_title=z)
            return fig, cmd


@app.callback(Output('table','columns'),
              Output('table','data'),
              Output('attrs','children'),
              Input('dropdown-dataset', 'value'),
              )
def update_table(file_dataset):
    if file_dataset is None or len(file_dataset)==0:
        return [], [], ''
    file_dataset = [json.loads(f) for f in file_dataset]
    data = []
    files = list(set([ f for f,ds in file_dataset]))
    for f1 in files:
        with h5file(f1,'r') as hf:
            for f2,ds in file_dataset:
                if f1==f2:
                    name = ds if len(files)==1 else '{0}:{1}'.format(os.path.splitext(os.path.basename(f1))[0],ds)
                    attrs = {'id':name}
                    attrs.update(hf[ds].attrs)
                    data.append(attrs)
    columns = ['id']
    other_columns = list(set([k for d in data for k in d.keys() if k!='id']))
    other_columns.sort()
    columns.extend( other_columns )
    columns = [ {'name':c,'id':c} for c in columns]
    return columns, data, 'Attributes'


if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(debug=False, host='0.0.0.0', port=5012)
