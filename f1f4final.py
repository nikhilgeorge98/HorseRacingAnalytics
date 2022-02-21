import dash
from dash.dependencies import Input, Output, State
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import functools
import datetime
from datetime import time
import utilities as utils

'''********************************************************'''

# df = pd.concat(pd.read_excel(r'data\F1F4.xlsx', sheet_name=None), ignore_index=True)
df = pd.read_excel(r'data\NEW F1F4.xlsb', engine='pyxlsb')
# a = [x for x in df['Race Time'].unique() if type(x) == str]
# anew = [datetime.time(int(x.split('.')[0]), int(x.split('.')[1])) for x in a]
# df['Race Time'].replace(a, anew, inplace=True)
# df['Race Time'].replace([np.nan], [datetime.time(0, 0)], inplace=True)
# df['Race Time'] = [x.strftime("%H:%M:%S") for x in df['Race Time']]

for i in ['Venue', 'Season', 'Season Code', 'Month', 'F#', 'Class', 'Distance', 'Age', 'Sh']:
    try:
        to_replace = [i for i in df[i].unique() if type(i) != str]
        df[i].replace(to_replace,['NA']*len(to_replace),inplace=True)
    except:
        pass

for i in ['Year', 'Date', 'NoR', 'R No.', 'RESULT', 'H. No', 'Dr']:
    try:
        to_replace = [i for i in df[i].unique() if type(i) != int and type(i) != np.int64]
        df[i].replace(to_replace,[0]*len(to_replace),inplace=True)
    except:
        pass

for i in ['Wt', 'LBW']:
    try:
        to_replace = [i for i in df[i].unique() if type(i) != float and type(i) != np.float64]
        df[i].replace(to_replace,[0.0]*len(to_replace),inplace=True)
    except:
        pass

df.rename(columns={'F#': 'F'}, inplace=True)
df['fav_num'] = df.apply(lambda row: int(row.F.strip('F')), axis=1)
df.dropna(subset=['RESULT'], inplace=True)
df['RESULT'] = df['RESULT'].astype('int64')
df['Truth'] = df['fav_num'] == df['RESULT']
df['f_r'] = list(zip(df.fav_num, df.RESULT))
try:
    df.Distance.replace(["1400 M", "1600 M", "1100 M", "1200 M", 1400, 1600, 1100, 1200, np.nan],
                        ["1400M", "1600M", "1100M", "1200M", "1400M", "1600M", "1100M", "1200M", "-"], inplace=True)
except:
    pass

combo = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3),
         (3, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
race_stats = df.groupby(["Venue", "Season Code", "Date","Month","Year", "R No.", "Distance", "Class", "f_r"],
                        as_index=True).size().unstack(fill_value=0)
mis_col = [i for i in combo if i not in race_stats.columns]
for col in mis_col:
    race_stats[col] = [0] * len(race_stats)
race_stats = race_stats[combo]

race_stats['F12'] = race_stats[(1, 1)] * race_stats[(2, 2)]
race_stats['F13'] = race_stats[(1, 1)] * race_stats[(3, 2)]
race_stats['F14'] = race_stats[(1, 1)] * race_stats[(4, 2)]

race_stats['F21'] = race_stats[(2, 1)] * race_stats[(1, 2)]
race_stats['F23'] = race_stats[(2, 1)] * race_stats[(3, 2)]
race_stats['F24'] = race_stats[(2, 1)] * race_stats[(4, 2)]

race_stats['F31'] = race_stats[(3, 1)] * race_stats[(1, 2)]
race_stats['F32'] = race_stats[(3, 1)] * race_stats[(2, 2)]
race_stats['F34'] = race_stats[(3, 1)] * race_stats[(4, 2)]

race_stats['F41'] = race_stats[(4, 1)] * race_stats[(1, 2)]
race_stats['F42'] = race_stats[(4, 1)] * race_stats[(2, 2)]
race_stats['F43'] = race_stats[(4, 1)] * race_stats[(3, 2)]

race_stats['F12|F21'] = (race_stats[(1, 1)] * race_stats[(2, 2)]) | (race_stats[(1, 2)] * race_stats[(2, 1)])
race_stats['F13|F31'] = (race_stats[(1, 1)] * race_stats[(3, 2)]) | (race_stats[(1, 2)] * race_stats[(3, 1)])
race_stats['F14|F41'] = (race_stats[(1, 1)] * race_stats[(4, 2)]) | (race_stats[(1, 2)] * race_stats[(4, 1)])

race_stats['F23|F32'] = (race_stats[(2, 1)] * race_stats[(3, 2)]) | (race_stats[(2, 2)] * race_stats[(3, 1)])
race_stats['F24|F42'] = (race_stats[(2, 1)] * race_stats[(4, 2)]) | (race_stats[(2, 2)] * race_stats[(4, 1)])

race_stats['F34|F43'] = (race_stats[(3, 1)] * race_stats[(4, 2)]) | (race_stats[(3, 2)] * race_stats[(4, 1)])

race_stats['F1|F2'] = race_stats[(1, 1)] | race_stats[(2, 1)]
race_stats['F1|F3'] = race_stats[(1, 1)] | race_stats[(3, 1)]
race_stats['F1|F4'] = race_stats[(1, 1)] | race_stats[(4, 1)]

race_stats['F2|F3'] = race_stats[(2, 1)] | race_stats[(3, 1)]
race_stats['F2|F4'] = race_stats[(2, 1)] | race_stats[(4, 1)]

race_stats['F3|F4'] = race_stats[(3, 1)] | race_stats[(4, 1)]

race_stats['F1|F2|F3'] = race_stats[(1, 1)] | race_stats[(2, 1)] | race_stats[(3, 1)]
race_stats['F1|F2|F4'] = race_stats[(1, 1)] | race_stats[(2, 1)] | race_stats[(4, 1)]
race_stats['F1|F3|F4'] = race_stats[(1, 1)] | race_stats[(3, 1)] | race_stats[(4, 1)]
race_stats['F2|F3|F4'] = race_stats[(2, 1)] | race_stats[(3, 1)] | race_stats[(4, 1)]

race_stats['F1|F2|F3|F4'] = race_stats[(1, 1)] | race_stats[(2, 1)] | race_stats[(3, 1)] | race_stats[(4, 1)]

race_stats['F123'] = race_stats[(1, 1)] * race_stats[(2, 2)] * race_stats[(3, 3)]
race_stats['F124'] = race_stats[(1, 1)] * race_stats[(2, 2)] * race_stats[(4, 3)]
race_stats['F132'] = race_stats[(1, 1)] * race_stats[(3, 2)] * race_stats[(2, 3)]
race_stats['F134'] = race_stats[(1, 1)] * race_stats[(3, 2)] * race_stats[(4, 3)]
race_stats['F142'] = race_stats[(1, 1)] * race_stats[(4, 2)] * race_stats[(2, 3)]
race_stats['F143'] = race_stats[(1, 1)] * race_stats[(4, 2)] * race_stats[(3, 3)]

race_stats['F213'] = race_stats[(2, 1)] * race_stats[(1, 2)] * race_stats[(3, 3)]
race_stats['F214'] = race_stats[(2, 1)] * race_stats[(1, 2)] * race_stats[(4, 3)]
race_stats['F231'] = race_stats[(2, 1)] * race_stats[(3, 2)] * race_stats[(1, 3)]
race_stats['F234'] = race_stats[(2, 1)] * race_stats[(3, 2)] * race_stats[(4, 3)]
race_stats['F241'] = race_stats[(2, 1)] * race_stats[(4, 2)] * race_stats[(1, 3)]
race_stats['F243'] = race_stats[(2, 1)] * race_stats[(4, 2)] * race_stats[(3, 3)]

race_stats['F312'] = race_stats[(3, 1)] * race_stats[(1, 2)] * race_stats[(2, 3)]
race_stats['F314'] = race_stats[(3, 1)] * race_stats[(1, 2)] * race_stats[(4, 3)]
race_stats['F321'] = race_stats[(3, 1)] * race_stats[(2, 2)] * race_stats[(1, 3)]
race_stats['F324'] = race_stats[(3, 1)] * race_stats[(2, 2)] * race_stats[(4, 3)]
race_stats['F341'] = race_stats[(3, 1)] * race_stats[(4, 2)] * race_stats[(1, 3)]
race_stats['F342'] = race_stats[(3, 1)] * race_stats[(4, 2)] * race_stats[(2, 3)]

race_stats['F412'] = race_stats[(4, 1)] * race_stats[(1, 2)] * race_stats[(2, 3)]
race_stats['F413'] = race_stats[(4, 1)] * race_stats[(1, 2)] * race_stats[(3, 3)]
race_stats['F421'] = race_stats[(4, 1)] * race_stats[(2, 2)] * race_stats[(1, 3)]
race_stats['F423'] = race_stats[(4, 1)] * race_stats[(2, 2)] * race_stats[(3, 3)]
race_stats['F431'] = race_stats[(4, 1)] * race_stats[(3, 2)] * race_stats[(1, 3)]
race_stats['F432'] = race_stats[(4, 1)] * race_stats[(3, 2)] * race_stats[(2, 3)]

race_stats['F1234'] = race_stats[(1, 1)] * race_stats[(2, 2)] * race_stats[(3, 3)] * race_stats[(4, 4)]
race_stats['F1243'] = race_stats[(1, 1)] * race_stats[(2, 2)] * race_stats[(4, 3)] * race_stats[(3, 4)]
race_stats['F1324'] = race_stats[(1, 1)] * race_stats[(3, 2)] * race_stats[(2, 3)] * race_stats[(4, 4)]
race_stats['F1342'] = race_stats[(1, 1)] * race_stats[(3, 2)] * race_stats[(4, 3)] * race_stats[(2, 4)]
race_stats['F1423'] = race_stats[(1, 1)] * race_stats[(4, 2)] * race_stats[(2, 3)] * race_stats[(3, 4)]
race_stats['F1432'] = race_stats[(1, 1)] * race_stats[(4, 2)] * race_stats[(3, 3)] * race_stats[(2, 4)]

race_stats['F2134'] = race_stats[(2, 1)] * race_stats[(1, 2)] * race_stats[(3, 3)] * race_stats[(4, 4)]
race_stats['F2143'] = race_stats[(2, 1)] * race_stats[(1, 2)] * race_stats[(4, 3)] * race_stats[(3, 4)]
race_stats['F2314'] = race_stats[(2, 1)] * race_stats[(3, 2)] * race_stats[(1, 3)] * race_stats[(4, 4)]
race_stats['F2341'] = race_stats[(2, 1)] * race_stats[(3, 2)] * race_stats[(4, 3)] * race_stats[(1, 4)]
race_stats['F2413'] = race_stats[(2, 1)] * race_stats[(4, 2)] * race_stats[(1, 3)] * race_stats[(3, 4)]
race_stats['F2431'] = race_stats[(2, 1)] * race_stats[(4, 2)] * race_stats[(3, 3)] * race_stats[(1, 4)]

race_stats['F3124'] = race_stats[(3, 1)] * race_stats[(1, 2)] * race_stats[(2, 3)] * race_stats[(4, 4)]
race_stats['F3142'] = race_stats[(3, 1)] * race_stats[(1, 2)] * race_stats[(4, 3)] * race_stats[(2, 4)]
race_stats['F3214'] = race_stats[(3, 1)] * race_stats[(2, 2)] * race_stats[(1, 3)] * race_stats[(4, 4)]
race_stats['F3241'] = race_stats[(3, 1)] * race_stats[(2, 2)] * race_stats[(4, 3)] * race_stats[(1, 4)]
race_stats['F3412'] = race_stats[(3, 1)] * race_stats[(4, 2)] * race_stats[(1, 3)] * race_stats[(2, 4)]
race_stats['F3421'] = race_stats[(3, 1)] * race_stats[(4, 2)] * race_stats[(2, 3)] * race_stats[(1, 4)]

race_stats['F4123'] = race_stats[(4, 1)] * race_stats[(1, 2)] * race_stats[(2, 3)] * race_stats[(3, 4)]
race_stats['F4132'] = race_stats[(4, 1)] * race_stats[(1, 2)] * race_stats[(3, 3)] * race_stats[(2, 4)]
race_stats['F4213'] = race_stats[(4, 1)] * race_stats[(2, 2)] * race_stats[(1, 3)] * race_stats[(3, 4)]
race_stats['F4231'] = race_stats[(4, 1)] * race_stats[(2, 2)] * race_stats[(3, 3)] * race_stats[(1, 4)]
race_stats['F4312'] = race_stats[(4, 1)] * race_stats[(3, 2)] * race_stats[(1, 3)] * race_stats[(2, 4)]
race_stats['F4321'] = race_stats[(4, 1)] * race_stats[(3, 2)] * race_stats[(2, 3)] * race_stats[(1, 4)]

race_stats.reset_index(inplace=True)
race_stats.sort_values(['Date', 'Season Code', 'R No.'], inplace=True)

forecast = ['F12', 'F13', 'F14', 'F21', 'F23', 'F24', 'F31', 'F32', 'F34', 'F41', 'F42', 'F43']
trinella = ['F123', 'F124', 'F132', 'F134', 'F142', 'F143', 'F213', 'F214', 'F231', 'F234', 'F241', 'F243', 'F312',
            'F314', 'F321', 'F324', 'F341', 'F342', 'F412', 'F413', 'F421', 'F423', 'F431', 'F432']
exacta = ['F1234', 'F1243', 'F1324', 'F1342', 'F1423', 'F1432', 'F2134', 'F2143', 'F2314', 'F2341', 'F2413', 'F2431',
          'F3124', 'F3142', 'F3214', 'F3241', 'F3412', 'F3421', 'F4123', 'F4132', 'F4213', 'F4231', 'F4312', 'F4321']
quinella = ['F12|F21', 'F13|F31', 'F14|F41', 'F23|F32', 'F24|F42', 'F34|F43']
# fc = ['F12']
# qn = ['F12|F21']
# qnp = ['F12|F21', 'F13|F31', 'F23|F32']
# tanala = ['F123']
# tanala_place = ['F123|F132']
# trio = ['F123', 'F132', 'F213', 'F231', 'F312', 'F321']
# exacta = ['F1234']
# first_4 = ['F1234', 'F1243', 'F1324', 'F1342', 'F1423', 'F1432', 'F2134', 'F2143', 'F2314', 'F2341', 'F2413', 'F2431',
#           'F3124', 'F3142', 'F3214', 'F3241', 'F3412', 'F3421', 'F4123', 'F4132', 'F4213', 'F4231', 'F4312', 'F4321']

'''**************************************************************************'''

app = dash.Dash(__name__)


filters = [
    html.P("Centre:", className="control_label"),
    html.Div(
        dcc.Dropdown(
            id='Centre',
            className='dcc_control',
            options=[{'label': 'All', 'value': 'All'}] + [{'label': i, 'value': i} for i in
                                                          sorted(df['Venue'].unique())],
            value='All',
            clearable=False
        ),
        className='dash-dropdown'
    ),
    html.P("Search by:", className="control_label"),
    html.Div(
        dcc.Dropdown(
            id='type',
            className='dcc_control',
            options=[
                {'label': 'ALL', 'value': 'All'},
                {'label': 'F1-F4', 'value': 'F1F4'},
                {'label': 'FORECAST', 'value': 'Forecast'},
                {'label': 'QUINELLA', 'value': 'Quinella'},
                {'label': 'TRINELLA', 'value': 'Trinella'},
                {'label': 'EXACTA', 'value': 'Exacta'}
            ],
            value='All',
            clearable=False
        ),
        className='dash-dropdown'
    ),
    html.P("Type:", className="control_label"),
    html.Div(
        dcc.Dropdown(
            id='howtype',
            className='dcc_control',
            options=[
                {'label': 'MULTI', 'value': 'multi'},
                {'label': 'SINGLE', 'value': 'single'}
            ],
            value='single',
            clearable=False
        ),
        className='dash-dropdown'
    ),
    html.P("Search:", className="control_label"),
    html.Div(
        dcc.Dropdown(
            id='combination',
            className='dcc_control',
            value=None,
            placeholder='Search',
            multi=False
        ),
        className='dash-dropdown'
    ),
    html.P("Filter", className="control_label"),
    html.Div(
        dcc.Dropdown(
            id='Season',
            className='dcc_control',
            value=None,
            placeholder='Season',
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Month',
            className='dcc_control',
            value=None,
            placeholder='Month',
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Year',
            className='dcc_control',
            value=None,
            placeholder='Year',
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Date',
            className='dcc_control',
            value=None,
            placeholder='Date',
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Race_no',
            className='dcc_control',
            value=None,
            placeholder='Race No.',
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Class',
            className='dcc_control',
            value=None,
            placeholder='Class',
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Distance',
            className='dcc_control',
            value=None,
            placeholder='Distance',
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='NoR',
            className='dcc_control',
            value=None,
            placeholder='NoR',
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Hno',
            className='dcc_control',
            value=None,
            placeholder='H. No',
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Age',
            className='dcc_control',
            value=None,
            placeholder='Age',
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Wt',
            className='dcc_control',
            value=None,
            placeholder='Wt',
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Sh',
            className='dcc_control',
            value=None,
            placeholder='Sh',
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='LBW',
            className='dcc_control',
            value=None,
            placeholder='LBW',
            multi=True
        ),
        className='dash-dropdown'
    )

]

row0 = dbc.Row(
    [
        dbc.Col(
            [
                html.Div(id='Overall_scoreboard', className='pretty_container', style={'display': 'initial'})
            ],
            width=12, className='pretty_container twelve columns', id='base_right-column'
        )
    ], id='fixedontop', className='flex-display'
)

row1 = dbc.Row(
    [
        dbc.Col(
            children=[html.H2("F1-F4 Analysis", id='title'),
                      html.Div(children=filters, id='cross-filter-options', className='pretty_container')],
            width=3, className='three columns', style={'display': 'flex', 'flex-direction': 'column'}
        ),
        dbc.Col(
            [
                html.Div(id='scoreboard', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboard2', className='pretty_container', style={'display': 'initial'},
                         children=[
                            html.Div(id='season_unfiltered', className='pretty_container', style={'display': 'block'}),
                            html.Div(id='season_filtered', className='pretty_container', style={'display': 'block'})]),
                html.Div(id='flexible_scoreboard', className='pretty_container', style={'display': 'initial'},
                         children=[
                            html.Div(
                                dcc.Dropdown(
                                    id='statsBy',
                                    className='dcc_control',
                                    options=[
                                        {'label': 'Distance', 'value': 'Distance'},
                                        {'label': 'Class', 'value': 'Class'},
                                        {'label': 'R No.', 'value': 'R No.'},
                                        {'label': 'Month', 'value': 'Month'},
                                        {'label': 'Year', 'value': 'Year'},
                                        {'label': 'NoR', 'value': 'NoR'},
                                        {'label': 'H. No', 'value': 'H. No'},
                                        {'label': 'Age', 'value': 'Age'},
                                        {'label': 'Wt', 'value': 'Wt'},
                                        {'label': 'Sh', 'value': 'Sh'}
                                    ],
                                    value='Distance',
                                    clearable=False
                                ),
                                className='dash-dropdown'
                            ),
                            html.Div(id='flexible_unfiltered', className='pretty_container', style={'display': 'block'}),
                            html.Div(id='flexible_filtered', className='pretty_container', style={'display': 'block'})])
            ],
            width=9, className='nine columns', id='right-column'
        )
    ], className='flex-display'
)

row2 = dbc.Row(
    [
        dbc.Col(html.Div(id='graphslot1', className='dash-graph'), width=12,
                className='pretty_container twelve columns', id='row2', style={'display': 'none'})
    ], className='flex-display'
)

row3 = dbc.Row(
    [
        dbc.Col(html.Div(id='graphslot2', className='dash-graph'), width=12,
                className='pretty_container twelve columns', id='row3', style={'display': 'none'})
    ], className='flex-display'
)

row4 = dbc.Row(
    [
        dbc.Col(html.Div(id='graphslot3', className='dash-graph'), width=12,
                className='pretty_container twelve columns', id='row4', style={'display': 'none'})
    ], className='flex-display'
)

app.layout = html.Div([
    row0,
    row1,
    row2,
    row3,
    row4,
    html.Div(id="store_original_df", children=[df.to_json(orient='split')], style={'display': 'none'}),
    html.Div(id="store_racewise_df", children=[race_stats.to_json(orient='split')], style={'display': 'none'}),
    html.Div(id="store_working_overall_df", style={'display': 'none'}),
    html.Div(id="store_working_filtered_df", style={'display': 'none'}),
    html.Div(id="store_season_unfiltered_stats", style={'display': 'none'}),
    html.Div(id="store_season_filtered_stats", style={'display': 'none'}),
    html.Div(id="store_unfiltered_stats", style={'display': 'none'}),
    html.Div(id="store_filtered_stats", style={'display': 'none'})],
    id='mainContainer', style={'display': 'flex', 'flex-direction': 'column'})


@app.callback(
    [Output(component_id='combination', component_property='options'),
     Output(component_id='Season', component_property='options'),
     Output(component_id='Race_no', component_property='options'),
     Output(component_id='Class', component_property='options'),
     Output(component_id='Distance', component_property='options'),
     Output(component_id='Season', component_property='value'),
     Output(component_id='Race_no', component_property='value'),
     Output(component_id='Class', component_property='value'),
     Output(component_id='Distance', component_property='value'),
     Output(component_id='NoR', component_property='options'),
     Output(component_id='NoR', component_property='value'),
     Output(component_id='Month', component_property='options'),
     Output(component_id='Month', component_property='value'),
     Output(component_id='Year', component_property='options'),
     Output(component_id='Year', component_property='value'),
     Output(component_id='Date', component_property='options'),
     Output(component_id='Date', component_property='value'),
     Output(component_id='Hno', component_property='options'),
     Output(component_id='Hno', component_property='value'),
     Output(component_id='Age', component_property='options'),
     Output(component_id='Age', component_property='value'),
     Output(component_id='Wt', component_property='options'),
     Output(component_id='Wt', component_property='value'),
     Output(component_id='Sh', component_property='options'),
     Output(component_id='Sh', component_property='value'),
     Output(component_id='LBW', component_property='options'),
     Output(component_id='LBW', component_property='value')],
    [Input(component_id='type', component_property='value'),
     Input(component_id='Centre', component_property='value'),
     Input(component_id='howtype', component_property='value')],
    prevent_initial_call=True
)
def update_search_dropdown(type, Centre, how):
    Forecast = ['F12', 'F13', 'F14', 'F21', 'F23', 'F24', 'F31', 'F32', 'F34', 'F41', 'F42', 'F43']
    Trinella = ['F123', 'F124', 'F132', 'F134', 'F142', 'F143', 'F213', 'F214', 'F231', 'F234', 'F241', 'F243', 'F312',
                'F314', 'F321', 'F324', 'F341', 'F342', 'F412', 'F413', 'F421', 'F423', 'F431', 'F432']
    Exacta = ['F1234', 'F1243', 'F1324', 'F1342', 'F1423', 'F1432', 'F2134', 'F2143', 'F2314', 'F2341', 'F2413',
              'F2431', 'F3124', 'F3142', 'F3214', 'F3241', 'F3412', 'F3421', 'F4123', 'F4132', 'F4213', 'F4231',
              'F4312', 'F4321']
    quinella = ['F12|F21', 'F13|F31', 'F14|F41', 'F23|F32', 'F24|F42', 'F34|F43']
    if type == 'Forecast':
        combolist = Forecast
    elif type == 'Quinella':
        combolist = quinella
    elif type == 'Trinella':
        combolist = Trinella
    elif type == 'Exacta':
        combolist = Exacta
    elif type == 'F1F4' or how == 'multi':
        combolist = ['F1', 'F2', 'F3', 'F4']
    elif type == 'All':
        combolist = ['F1', 'F2', 'F3', 'F4'] + Forecast + quinella + Trinella + Exacta
    options = [{'label': i, 'value': i} for i in combolist]
    season_options = [{"label": i, "value": i} for i in df['Season Code'].unique()]
    rno_options = [{"label": i, "value": i} for i in sorted(df['R No.'].unique())]
    distance_options = [{"label": i, "value": i} for i in sorted(df['Distance'].unique())]
    class_options = [{"label": i, "value": i} for i in sorted(df['Class'].unique())]
    NoR_options = [{"label": i, "value": i} for i in sorted(df['NoR'].unique())]
    month_options = [{"label": i, "value": i} for i in sorted(df['Month'].unique())]
    year_options = [{"label": i, "value": i} for i in sorted(df['Year'].unique())]
    date_options = [{"label": i, "value": i} for i in sorted(df['Date'].unique())]
    hno_options = [{"label": i, "value": i} for i in sorted(df['H. No'].unique())]
    age_options = [{"label": i, "value": i} for i in sorted(df['Age'].unique())]
    wt_options = [{"label": i, "value": i} for i in sorted(df['Wt'].unique())]
    sh_options = [{"label": i, "value": i} for i in sorted(df['Sh'].unique())]
    lbw_options = [{"label": i, "value": i} for i in sorted(df['LBW'].unique())]
    if Centre != 'All':
        season_options = [{"label": i, "value": i} for i in df.loc[df['Venue'] == Centre]['Season Code'].unique()]
        rno_options = [{"label": i, "value": i} for i in df.loc[df['Venue'] == Centre]['R No.'].unique()]
        distance_options = [{"label": i, "value": i} for i in df.loc[df['Venue'] == Centre]['Distance'].unique()]
        class_options = [{"label": i, "value": i} for i in df.loc[df['Venue'] == Centre]['Class'].unique()]
        NoR_options = [{"label": i, "value": i} for i in df.loc[df['Venue'] == Centre]['NoR'].unique()]
        month_options = [{"label": i, "value": i} for i in df.loc[df['Venue'] == Centre]['Month'].unique()]
        year_options = [{"label": i, "value": i} for i in df.loc[df['Venue'] == Centre]['Year'].unique()]
        date_options = [{"label": i, "value": i} for i in df.loc[df['Venue'] == Centre]['Date'].unique()]
        hno_options = [{"label": i, "value": i} for i in df.loc[df['Venue'] == Centre]['H. No'].unique()]
        age_options = [{"label": i, "value": i} for i in df.loc[df['Venue'] == Centre]['Age'].unique()]
        wt_options = [{"label": i, "value": i} for i in df.loc[df['Venue'] == Centre]['Wt'].unique()]
        sh_options = [{"label": i, "value": i} for i in df.loc[df['Venue'] == Centre]['Sh'].unique()]
        lbw_options = [{"label": i, "value": i} for i in df.loc[df['Venue'] == Centre]['LBW'].unique()]

    return [options, season_options, rno_options, class_options, distance_options, None, None, None, None, NoR_options,
            None,month_options,None,year_options,None,date_options,None,hno_options,None,age_options,None,wt_options,None,
            sh_options,None,lbw_options,None]


@app.callback(
    Output('combination', 'value'),
    [Input('type', 'value')]
)
def clear_combo(type):
    return None


@app.callback(
    [Output('type', 'value'),
     Output('combination', 'multi'),
     Output('type', 'disabled')],
    [Input('howtype', 'value')]
)
def reset_type(how):
    if how == 'multi':
        return ['F1F4', True, True]
    return ['All', False, False]


@app.callback(
    [Output(component_id='Overall_scoreboard', component_property='children'),
     Output(component_id='Overall_scoreboard', component_property='style'),
     Output(component_id='scoreboard', component_property='children'),
     # Output(component_id='scoreboard2', component_property='children'),
     Output(component_id='scoreboard2', component_property='style'),
     Output(component_id='flexible_scoreboard', component_property='style'),
     Output(component_id='graphslot1', component_property='children'),
     Output(component_id='row2', component_property='style'),
     Output(component_id='graphslot2', component_property='children'),
     Output(component_id='row3', component_property='style'),
     Output(component_id='store_working_overall_df', component_property='children'),
     Output(component_id='store_working_filtered_df', component_property='children'),
     Output("store_season_unfiltered_stats", component_property='children'),
     Output("store_season_filtered_stats", component_property='children'),
     Output("store_unfiltered_stats", component_property='children'),
     Output("store_filtered_stats", component_property='children'),
     Output("flexible_unfiltered", component_property='children'),
     Output("flexible_unfiltered", component_property='style'),
     Output("flexible_filtered", component_property='children'),
     Output("flexible_filtered", component_property='style'),
     Output("season_unfiltered", component_property='children'),
     Output("season_unfiltered", component_property='style'),
     Output("season_filtered", component_property='children'),
     Output("season_filtered", component_property='style')
     ],
    [Input('howtype', 'value'),
     Input(component_id='type', component_property='value'),
     Input(component_id='combination', component_property='value'),
     Input(component_id='Centre', component_property='value'),
     Input(component_id='statsBy', component_property='value'),
     Input(component_id='Season', component_property='value'),
     Input(component_id='Race_no', component_property='value'),
     Input(component_id='Distance', component_property='value'),
     Input(component_id='Class', component_property='value'),
     Input(component_id='Month', component_property='value'),
     Input(component_id='Year', component_property='value'),
     Input(component_id='Date', component_property='value'),
     Input(component_id='NoR', component_property='value'),
     Input(component_id='Hno', component_property='value'),
     Input(component_id='Age', component_property='value'),
     Input(component_id='Wt', component_property='value'),
     Input(component_id='Sh', component_property='value'),
     Input(component_id='LBW', component_property='value')
     ],
    prevent_initial_call=True
)
def update_main(how, type, combination, Centre, statsby, *searchparam):
    if Centre is None:
        print("Hello")
        raise PreventUpdate
    # print("Type:", type)
    # print("combo:", combination)
    # print("centre:", Centre)
    # print("extra:", searchparam)
    if Centre != 'All':
        centre_race_stats = race_stats[race_stats['Venue'] == Centre]
        centre_df = df[df['Venue'] == Centre]
    else:
        centre_race_stats = race_stats
        centre_df = df

    all_f1f4_stats = utils.generate_f1f4_stats(centre_df)
    all_f1f4_season_stats = utils.generate_f1f4_season_stats(centre_df)
    all_f1f4_generic_stats = utils.generate_f1f4_generic_stats(centre_df, statsby)

    filtered_race_stats = centre_race_stats
    filtered_df = centre_df
    cond = []
    cond_df = []
    cols = ['Season Code', 'R No.', 'Distance', 'Class','Month', 'Year', 'Date']
    cols_df = ['Season Code', 'R No.', 'Distance', 'Class','Month', 'Year', 'Date', 'NoR',  'H. No', 'Age', 'Wt', 'Sh', 'LBW']
    condition = []
    condition_df = []

    for (key, value) in zip(cols, searchparam[:-6]):
        # print(key,'=',value)
        if value != [] and value is not None and value != "":
            for v in value:
                # print('V--->',v)
                if (v != "All" and v is not None and v != ""):
                    cond.append(filtered_race_stats[key] == v)
            condition.append(utils.conjunction(1, *cond))
            cond = []
    if condition:
        filtered_race_stats = filtered_race_stats[utils.conjunction(0, *condition)]

    for (key, value) in zip(cols_df, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "":
            for v in value:
                # print('V--->',v)
                if (v != "All" and v is not None and v != ""):
                    cond_df.append(filtered_df[key] == v)
            condition_df.append(utils.conjunction(1, *cond_df))
            cond_df = []
    if condition_df:
        filtered_df = filtered_df[utils.conjunction(0, *condition_df)]

    f1f4_stats = utils.generate_f1f4_stats(filtered_df)
    f1f4_season_stats = utils.generate_f1f4_season_stats(filtered_df)
    f1f4_generic_stats = utils.generate_f1f4_generic_stats(filtered_df, statsby)

    if len(filtered_race_stats) == 0 or len(filtered_df) == 0:
        score_return = "No Races to Show"
        # score2_return = []
        score2_style = {'display': 'none'}
        flex_score_style = {'display': 'none'}
        ov_score = []
        ov_style = {'display': 'none'}
        graph1 = []
        row2_style = {'display': 'none'}
        graph2 = []
        row3_style = {'display': 'none'}

        flexible_unfiltered_return = []
        flexible_unfiltered_style = {'display': 'none'}
        flexible_filtered_return = []
        flexible_filtered_style = {'display': 'none'}

        season_unfiltered_return = []
        season_unfiltered_style = {'display': 'none'}
        season_filtered_return = []
        season_filtered_style = {'display': 'none'}
    elif combination == [] or combination == "" or combination is None:
        score_return = utils.generate_favourite_scoreboard(f1f4_stats)
        # score2_return = utils.generate_favourite_season_scoreboard(f1f4_season_stats.loc['F1'], 'F1')
        score2_style = {'display': 'initial'}
        flex_score_style = {'display': 'initial'}
        ov_score = []
        ov_style = {'display': 'none'}
        graph1 = utils.gen_overall_centre_table(filtered_race_stats)
        row2_style = {'display': 'initial'}
        graph2 = [html.Div(),
                  dcc.Graph(figure=utils.generatebaroverall(filtered_race_stats), style={'height': '100%', 'width': '100%'})]
        row3_style = {'display': 'initial'}

        flexible_unfiltered_return = utils.generate_favourite_generic_scoreboard(all_f1f4_generic_stats.loc['F1'], 'F1', 'tableflex_unf', statsby, 'Unfiltered')
        flexible_unfiltered_style = {'display': 'block'}
        flexible_filtered_return = utils.generate_favourite_generic_scoreboard(f1f4_generic_stats.loc['F1'], 'F1', 'tableflex_fil', statsby, 'Filtered')
        flexible_filtered_style = {'display': 'block'}

        season_unfiltered_return = utils.generate_favourite_season_scoreboard(all_f1f4_season_stats.loc['F1'], 'F1', 'Unfiltered')
        season_unfiltered_style = {'display': 'block'}
        season_filtered_return = utils.generate_favourite_season_scoreboard(f1f4_season_stats.loc['F1'], 'F1', 'Filtered')
        season_filtered_style = {'display': 'block'}
    elif combination not in ['F1', 'F2', 'F3', 'F4'] and how != 'multi':
        print("comnination is in those 4 NOT!!!!!!!!!!")
        score_return = [
            html.Div(id="form", children=[
                html.Div('Form Guide(W):', className='form_guide_header'),
                html.Ul(id='form_guide_ul',
                        className='form_guide_ul_win',
                        children=[utils.generate_li_adv(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(filtered_race_stats[combination][::-1],
                                                filtered_race_stats[combination][::-1]))]
                        )]),
            html.Div(
                [html.Div("Streaks:", className='streak_header'), "{}".format(utils.streak(filtered_race_stats['F12'])[0])],
                id='streak_display')
        ]
        # score2_return = []
        score2_style = {'display': 'none'}
        flex_score_style = {'display': 'none'}
        ov_score = utils.generate_overall_scoreboard_for_combo(centre_race_stats, filtered_race_stats, combination)
        ov_style = {'display': 'initial'}
        graph1 = []
        row2_style = {'display': 'none'}
        graph2 = []
        row3_style = {'display': 'none'}

        flexible_unfiltered_return = []
        flexible_unfiltered_style = {'display': 'none'}
        flexible_filtered_return = []
        flexible_filtered_style = {'display': 'none'}

        season_unfiltered_return = []
        season_unfiltered_style = {'display': 'none'}
        season_filtered_return = []
        season_filtered_style = {'display': 'none'}
    elif combination in ['F1', 'F2', 'F3', 'F4']:
        print("comnination is in those 4")
        filtered_df = filtered_df[filtered_df['F'] == combination]
        centre_df = centre_df[centre_df['F'] == combination]
        score_return = [
            html.Div(id="form", children=[
                html.Div('Form Guide:', className='form_guide_header'),
                html.Ul(id='form_guide_ul',
                        className='form_guide_ul_first',
                        children=[utils.generate_li_adv_f1f4(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(filtered_df['RESULT'][::-1],
                                                filtered_df['RESULT'][::-1]))]
                        )]),
            html.Div(
                [html.Div("Streaks:", className='streak_header'), "{}".format(utils.streakf1f4(filtered_df)[0])],
                id='streak_display')
        ]
        # score2_return = []
        score2_style = {'display': 'none'}
        flex_score_style = {'display': 'none'}
        ov_score = utils.generate_overall_scoreboard(centre_df, filtered_df)
        ov_style = {'display': 'initial'}
        graph1 = []
        row2_style = {'display': 'none'}
        graph2 = []
        row3_style = {'display': 'none'}

        flexible_unfiltered_return = []
        flexible_unfiltered_style = {'display': 'none'}
        flexible_filtered_return = []
        flexible_filtered_style = {'display': 'none'}

        season_unfiltered_return = []
        season_unfiltered_style = {'display': 'none'}
        season_filtered_return = []
        season_filtered_style = {'display': 'none'}
    elif how == 'multi':  #################do here
        try:
            combo = ""
            # print("combolist")
            if set(combination) == set(["F1"]):
                combo
            elif set(combination) == set(['F1', 'F2']):
                combo = "F1|F2"
            elif set(combination) == set(['F1', 'F3']):
                combo = "F1|F3"
            elif set(combination) == set(['F1', 'F4']):
                combo = "F1|F4"
            elif set(combination) == set(['F2', 'F3']):
                combo = "F2|F3"
            elif set(combination) == set(['F2', 'F4']):
                combo = "F2|F4"
            elif set(combination) == set(['F3', 'F4']):
                combo = "F3|F4"
            elif set(combination) == set(['F1', 'F2', 'F3']):
                combo = "F1|F2|F3"
            elif set(combination) == set(['F1', 'F2', 'F4']):
                combo = "F1|F2|F4"
            elif set(combination) == set(['F1', 'F3', 'F4']):
                combo = "F1|F3|F4"
            elif set(combination) == set(['F2', 'F3', 'F4']):
                combo = "F2|F3|F4"
            elif set(combination) == set(['F1', 'F2', 'F3', 'F4']):
                combo = "F1|F2|F3|F4"
            # print("combo is {}".format(combo))
            score_return = [
                html.Div(id="form", children=[
                    html.Div('Form Guide(W):', className='form_guide_header'),
                    html.Ul(id='form_guide_ul',
                            className='form_guide_ul_win',
                            children=[utils.generate_li_adv(1, i, j, n) for n, (i, j) in
                                      enumerate(zip(filtered_race_stats[combo][::-1],
                                                    filtered_race_stats[combo][::-1]))]
                            )]),
                html.Div(
                    [html.Div("Streaks:", className='streak_header'),
                     "{}".format(utils.streak(filtered_race_stats['F12'])[0])],
                    id='streak_display')
            ]
            # score2_return = []
            score2_style = {'display': 'none'}
            flex_score_style = {'display': 'none'}
            ov_score = utils.generate_overall_scoreboard_for_combo(centre_race_stats, filtered_race_stats, combo)
            ov_style = {'display': 'initial'}
            graph1 = []
            row2_style = {'display': 'none'}
            graph2 = []
            row3_style = {'display': 'none'}

            flexible_unfiltered_return = []
            flexible_unfiltered_style = {'display': 'none'}
            flexible_filtered_return = []
            flexible_filtered_style = {'display': 'none'}

            season_unfiltered_return = []
            season_unfiltered_style = {'display': 'none'}
            season_filtered_return = []
            season_filtered_style = {'display': 'none'}
        except:
            raise PreventUpdate
    # print(f1f4_season_stats.index.names)
    return [ov_score, ov_style, score_return, score2_style, flex_score_style, graph1, row2_style, graph2, row3_style,
            [centre_df.to_json(orient='split')], [filtered_df.to_json(orient='split')],
            [all_f1f4_season_stats.index.names[0], all_f1f4_season_stats.index.names[1], all_f1f4_season_stats.reset_index().to_json(orient='split')],
            [f1f4_season_stats.index.names[0], f1f4_season_stats.index.names[1], f1f4_season_stats.reset_index().to_json(orient='split')],
            [all_f1f4_generic_stats.index.names[0], all_f1f4_generic_stats.index.names[1], all_f1f4_generic_stats.reset_index().to_json(orient='split')],
            [f1f4_generic_stats.index.names[0], f1f4_generic_stats.index.names[1], f1f4_generic_stats.reset_index().to_json(orient='split')],
            flexible_unfiltered_return, flexible_unfiltered_style,
            flexible_filtered_return, flexible_filtered_style,
            season_unfiltered_return, season_unfiltered_style,
            season_filtered_return, season_filtered_style]


@app.callback(
    [Output('form_guide_ul', 'className'),
     Output('loss_button', 'n_clicks')],
    [Input('win_button', 'n_clicks')]
)
def update_formguide_win(winclick):
    if winclick == 0:
        raise PreventUpdate
    return ['form_guide_ul_win', 0]


@app.callback(
    [Output('form_guide_ul', 'className'),
     Output('win_button', 'n_clicks')],
    [Input('loss_button', 'n_clicks')]
)
def update_formguide_loss(lossclick):
    if lossclick == 0:
        raise PreventUpdate
    return ['form_guide_ul_loss', 0]


@app.callback(
    [Output(component_id='longest_winning_streak_header', component_property='children'),
     Output(component_id='longest_losing_streak_header', component_property='children'),
     Output(component_id='overall_longest_winning_streak_data', component_property='children'),
     Output(component_id='overall_longest_losing_streak_data', component_property='children'),
     Output(component_id='longest_winning_streak_data', component_property='children'),
     Output(component_id='longest_losing_streak_data', component_property='children'),
     Output('form_guide_ul', 'className'),
     Output('second_button', 'n_clicks'),
     Output('third_button', 'n_clicks'),
     Output('place_button', 'n_clicks')],
    [Input('first_button', 'n_clicks')],
    [State(component_id='store_working_overall_df', component_property='children'),
     State(component_id='store_working_filtered_df', component_property='children')]
)
def update_formguide_first(click, overall, filtered):
    if click == 0:
        raise PreventUpdate
    overall_dff = pd.read_json(overall[0], orient='split')
    filtered_dff = pd.read_json(filtered[0], orient='split')
    return ["LWS(WIN)",
            "LLS(WIN)",
            "{}".format(utils.maxminstreaks(1, overall_dff)[0]),
            "{}".format(utils.maxminstreaks(1, overall_dff)[1]),
            "{}".format(utils.maxminstreaks(1, filtered_dff)[0]),
            "{}".format(utils.maxminstreaks(1, filtered_dff)[1]),
            'form_guide_ul_first', 0, 0, 0]


@app.callback(
    [Output(component_id='longest_winning_streak_header', component_property='children'),
     Output(component_id='longest_losing_streak_header', component_property='children'),
     Output(component_id='overall_longest_winning_streak_data', component_property='children'),
     Output(component_id='overall_longest_losing_streak_data', component_property='children'),
     Output(component_id='longest_winning_streak_data', component_property='children'),
     Output(component_id='longest_losing_streak_data', component_property='children'),
     Output('form_guide_ul', 'className'),
     Output('first_button', 'n_clicks'),
     Output('third_button', 'n_clicks'),
     Output('place_button', 'n_clicks')],
    [Input('second_button', 'n_clicks')],
    [State(component_id='store_working_overall_df', component_property='children'),
     State(component_id='store_working_filtered_df', component_property='children')]
)
def update_formguide_second(click, overall, filtered):
    if click == 0:
        raise PreventUpdate
    overall_dff = pd.read_json(overall[0], orient='split')
    filtered_dff = pd.read_json(filtered[0], orient='split')
    return ["LWS(SHP)",
            "LLS(SHP)",
            "{}".format(utils.maxminstreaks(2, overall_dff)[0]),
            "{}".format(utils.maxminstreaks(2, overall_dff)[1]),
            "{}".format(utils.maxminstreaks(2, filtered_dff)[0]),
            "{}".format(utils.maxminstreaks(2, filtered_dff)[1]),
            'form_guide_ul_second', 0, 0, 0]


@app.callback(
    [Output(component_id='longest_winning_streak_header', component_property='children'),
     Output(component_id='longest_losing_streak_header', component_property='children'),
     Output(component_id='overall_longest_winning_streak_data', component_property='children'),
     Output(component_id='overall_longest_losing_streak_data', component_property='children'),
     Output(component_id='longest_winning_streak_data', component_property='children'),
     Output(component_id='longest_losing_streak_data', component_property='children'),
     Output('form_guide_ul', 'className'),
     Output('second_button', 'n_clicks'),
     Output('first_button', 'n_clicks'),
     Output('place_button', 'n_clicks')],
    [Input('third_button', 'n_clicks')],
    [State(component_id='store_working_overall_df', component_property='children'),
     State(component_id='store_working_filtered_df', component_property='children')]
)
def update_formguide_third(click, overall, filtered):
    if click == 0:
        raise PreventUpdate
    overall_dff = pd.read_json(overall[0], orient='split')
    filtered_dff = pd.read_json(filtered[0], orient='split')
    return ["LWS(THP)",
            "LLS(THP)",
            "{}".format(utils.maxminstreaks(3, overall_dff)[0]),
            "{}".format(utils.maxminstreaks(3, overall_dff)[1]),
            "{}".format(utils.maxminstreaks(3, filtered_dff)[0]),
            "{}".format(utils.maxminstreaks(3, filtered_dff)[1]),
            'form_guide_ul_third', 0, 0, 0]


@app.callback(
    [Output(component_id='longest_winning_streak_header', component_property='children'),
     Output(component_id='longest_losing_streak_header', component_property='children'),
     Output(component_id='overall_longest_winning_streak_data', component_property='children'),
     Output(component_id='overall_longest_losing_streak_data', component_property='children'),
     Output(component_id='longest_winning_streak_data', component_property='children'),
     Output(component_id='longest_losing_streak_data', component_property='children'),
     Output('form_guide_ul', 'className'),
     Output('second_button', 'n_clicks'),
     Output('third_button', 'n_clicks'),
     Output('first_button', 'n_clicks')],
    [Input('place_button', 'n_clicks')],
    [State(component_id='store_working_overall_df', component_property='children'),
     State(component_id='store_working_filtered_df', component_property='children')]
)
def update_formguide_place(click, overall, filtered):
    if click == 0:
        raise PreventUpdate
    overall_dff = pd.read_json(overall[0], orient='split')
    filtered_dff = pd.read_json(filtered[0], orient='split')
    return ["LWS(Plc)",
            "LLS(Plc)",
            "{}".format(utils.streakf1f4(overall_dff)[1]),
            "{}".format(utils.streakf1f4(overall_dff)[2]),
            "{}".format(utils.streakf1f4(filtered_dff)[1]),
            "{}".format(utils.streakf1f4(filtered_dff)[2]),
            'form_guide_ul_place', 0, 0, 0]


@app.callback(
    [Output('season_unfiltered', 'children'),
     Output('season_filtered', 'children'),
     Output('flexible_unfiltered', 'children'),
     Output('flexible_filtered', 'children')],
    [Input('table', 'active_cell')],
    [State('table', 'derived_viewport_data'),
     State('store_season_unfiltered_stats', 'children'),
     State('store_season_filtered_stats', 'children'),
     State('store_unfiltered_stats', 'children'),
     State('store_filtered_stats', 'children'),
     State('statsBy', 'value')]
)
def updatebyseason(active_cell, data, unf_season_df, fil_season_df, unfiltered_df, filtered_df, statsby):
    if active_cell is None:
        raise PreventUpdate
    print(active_cell)
    all_f1f4_season_stats = pd.read_json(unf_season_df[2], orient='split').set_index([unf_season_df[0], unf_season_df[1]])
    f1f4_season_stats = pd.read_json(fil_season_df[2], orient='split').set_index([fil_season_df[0], fil_season_df[1]])
    all_f1f4_generic_stats = pd.read_json(unfiltered_df[2], orient='split').set_index([unfiltered_df[0], unfiltered_df[1]])
    f1f4_generic_stats = pd.read_json(filtered_df[2], orient='split').set_index([filtered_df[0], filtered_df[1]])

    row = active_cell['row']
    column_id = active_cell['column_id']
    type = data[row]['F']
    # f1f4_stats = pd.read_json(season_df[0],orient='split')
    # print(season_df[0])
    # return []
    return [utils.generate_favourite_season_scoreboard(all_f1f4_season_stats.loc[type], type, 'Unfiltered'),
            utils.generate_favourite_season_scoreboard(f1f4_season_stats.loc[type], type, 'Filtered'),
            utils.generate_favourite_generic_scoreboard(all_f1f4_generic_stats.loc[type], type, 'tableflex_unf', statsby, 'Unfiltered'),
            utils.generate_favourite_generic_scoreboard(f1f4_generic_stats.loc[type], type, 'tableflex_fil', statsby, 'Filtered')]


if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)