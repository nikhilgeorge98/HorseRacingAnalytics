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
'''**************************************************************************'''

app = dash.Dash(__name__)


def generatebaroverall(dfff):
    c = list(
        ["#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
         "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
         "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
         "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
         "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
         "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
         "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF"])
    F = list(forecast + quinella + trinella + exacta)
    print(len(F))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=F,
                         y=[100 * (dfff[i].value_counts().get(1, 0) / len(dfff)) for i in F],
                         marker_color=c,
                         text=[round(100 * (dfff[i].value_counts().get(1, 0) / len(dfff)), 1) for i in F],
                         textposition='outside',
                         hoverinfo='none'
                         ))
    fig.update_xaxes(title_text="Combination",
                     ticktext=F,
                     showticklabels=True)
    fig.update_yaxes(title_text="W%",
                     showticklabels=True,
                     showgrid=False,
                     tickvals=np.arange(1, 100))
    fig.update_layout(title_text='Win percentages for all combinations',
                      showlegend=True,
                      autosize=True,
                      height=700,
                      xaxis={'showgrid': True, 'zeroline': True, 'gridcolor': '#082255', 'zerolinecolor': '#082255'},
                      yaxis={'showgrid': True, 'zeroline': True, 'gridcolor': '#082255', 'zerolinecolor': '#082255'},
                      xaxis_tickangle=-45,
                      plot_bgcolor="#061e44",
                      paper_bgcolor="#082255",
                      font_color="white"
                      )
    return fig


def generatelinebinary(dfff, combination):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, len(dfff)), y=dfff[combination],
                             mode='markers',
                             name='markers'))
    fig.update_xaxes(title_text="Races",
                     tickvals=dfff['Date'],
                     showticklabels=False)
    fig.update_yaxes(title_text="Win/Loss",
                     showticklabels=True,
                     showgrid=False,
                     tickvals=["L", "W"])
    fig.update_layout(title_text='Win distribution',
                      showlegend=True,
                      autosize=True,
                      height=700,
                      xaxis={'showgrid': True, 'zeroline': True, 'gridcolor': '#082255', 'zerolinecolor': '#082255'},
                      yaxis={'showgrid': True, 'zeroline': True, 'gridcolor': '#082255', 'zerolinecolor': '#082255'},
                      xaxis_tickangle=-45,
                      plot_bgcolor="#061e44",
                      paper_bgcolor="#082255",
                      font_color="white"
                      )
    return fig


def gen_overall_centre_table(dff):
    a = [
        html.Div("Total Races: {}".format(len(dff)), className='overall_table_header_f1f4'),
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Forecast", id='forecast_header', className='overallf1f4boxcolheader', colSpan=3,
                            style={'width': '25%'}),
                    html.Th("Quinella", id='quinella_header', className='overallf1f4boxcolheader', colSpan=3,
                            style={'width': '25%'}),
                    html.Th("Trinella", id='trinella_header', className='overallf1f4boxcolheader', colSpan=3,
                            style={'width': '25%'}),
                    html.Th("Exacta", id='exacta_header', className='overallf1f4boxcolheader', colSpan=3,
                            style={'width': '25%'})
                ])
            ]),
            html.Tr([
                html.Th("Combination", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W%", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("Combination", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W%", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("Combination", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W%", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("Combination", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W%", id='OO_header', className='overallf1f4boxrowheader'),
            ]),
            html.Tr([
                html.Th("{}".format(forecast[0]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[forecast[0]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[forecast[0]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(quinella[0]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[quinella[0]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[quinella[0]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[0]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[0]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[0]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[0]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[0]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[0]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("{}".format(forecast[1]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[forecast[1]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[forecast[1]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(quinella[1]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[quinella[1]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[quinella[1]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[1]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[1]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[1]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[1]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[1]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[1]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("{}".format(forecast[2]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[forecast[2]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[forecast[2]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(quinella[2]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[quinella[2]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[quinella[2]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[2]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[2]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[2]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[2]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[2]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[2]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("{}".format(forecast[3]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[forecast[3]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[forecast[3]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(quinella[3]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[quinella[3]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[quinella[3]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[3]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[3]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[3]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[3]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[3]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[3]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("{}".format(forecast[4]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[forecast[4]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[forecast[4]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(quinella[4]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[quinella[4]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[quinella[4]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[4]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[4]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[4]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[4]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[4]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[4]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("{}".format(forecast[5]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[forecast[5]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[forecast[5]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(quinella[5]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[quinella[5]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[quinella[5]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[5]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[5]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[5]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[5]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[5]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[5]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("{}".format(forecast[6]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[forecast[6]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[forecast[6]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[6]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[6]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[6]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[6]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[6]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[6]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("{}".format(forecast[7]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[forecast[7]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[forecast[7]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[7]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[7]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[7]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[7]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[7]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[7]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("{}".format(forecast[8]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[forecast[8]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[forecast[8]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[8]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[8]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[8]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[8]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[8]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[8]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("{}".format(forecast[9]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[forecast[9]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[forecast[9]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[9]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[9]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[9]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[9]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[9]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[9]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("{}".format(forecast[10]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[forecast[10]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[forecast[10]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[10]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[10]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[10]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[10]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[10]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[10]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("{}".format(forecast[11]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[forecast[11]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[forecast[11]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[11]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[11]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[11]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[11]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[11]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[11]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[12]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[12]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[12]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[12]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[12]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[12]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[13]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[13]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[13]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[13]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[13]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[13]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[14]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[14]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[14]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[14]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[14]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[14]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[15]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[15]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[15]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[15]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[15]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[15]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[16]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[16]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[16]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[16]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[16]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[16]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[17]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[17]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[17]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[17]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[17]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[17]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[18]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[18]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[18]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[18]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[18]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[18]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[19]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[19]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[19]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[19]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[19]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[19]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[20]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[20]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[20]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[20]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[20]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[20]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[21]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[21]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[21]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[21]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[21]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[21]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[22]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[22]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[22]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[22]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[22]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[22]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trinella[23]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[trinella[23]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trinella[23]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[23]), className='overallf1f4boxcell'),
                html.Td("{}".format(dff[exacta[23]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[23]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ])

        ], className='overf1f4table')]
    return a


def conjunction(type, *conditions):
    if type:
        return functools.reduce(np.logical_or, conditions)
    else:
        return functools.reduce(np.logical_and, conditions)


def maxminstreaks(n, dfff):
    lstreak = 0
    wstreak = 0
    maxlstreak = 0
    maxwstreak = 0
    sumLstreak = 0
    sumWstreak = 0
    numLstreak = 0
    numWstreak = 0
    for i in dfff['RESULT']:
        if i != n:
            if wstreak > 0:
                numWstreak += 1
                sumWstreak += wstreak
            if wstreak > maxwstreak:
                maxwstreak = wstreak
            wstreak = 0
            lstreak += 1
        if i == n:
            if lstreak > 0:
                numLstreak += 1
                sumLstreak += lstreak
            if lstreak > maxlstreak:
                maxlstreak = lstreak
            lstreak = 0
            wstreak += 1
    if lstreak > 0:
        numLstreak += 1
        sumLstreak += lstreak
    if lstreak > maxlstreak:
        maxlstreak = lstreak
    if wstreak > 0:
        numWstreak += 1
        sumWstreak += wstreak
    if wstreak > maxwstreak:
        maxwstreak = wstreak
    avgWstreak = 0
    avgLstreak = 0
    if numWstreak > 0:
        avgWstreak = sumWstreak / numWstreak
    if numLstreak:
        avgLstreak = sumLstreak / numLstreak

    return [maxwstreak, maxlstreak, avgWstreak, avgLstreak]


def maxwinstreaks(dfff, n=1):
    wstreak = 0
    maxwstreak = 0
    for i in dfff:
        if i != n:
            if wstreak > maxwstreak:
                maxwstreak = wstreak
            wstreak = 0
        if i == n:
            wstreak += 1
    if wstreak > maxwstreak:
        maxwstreak = wstreak

    return maxwstreak


def maxlosestreaks(dfff, n=1):
    lstreak = 0
    maxlstreak = 0
    for i in dfff:
        if i != n:
            lstreak += 1
        if i == n:
            if lstreak > maxlstreak:
                maxlstreak = lstreak
            lstreak = 0
    if lstreak > maxlstreak:
        maxlstreak = lstreak

    return maxlstreak


def streak(col):
    # find streaks and maxstreaklengths
    streak = ""
    ws = ""
    ls = ""
    lstreak = 0
    wstreak = 0
    maxlstreak = 0
    maxwstreak = 0
    for i in col[::-1]:
        if i == 0:
            if wstreak != 0:
                streak += " " + ws
            if wstreak > maxwstreak:
                maxwstreak = wstreak
            wstreak = 0
            lstreak += 1
            ls = ls.replace(ls, "{}L".format(lstreak))
        if i == 1:
            if lstreak != 0:
                streak += " " + ls
            if lstreak > maxlstreak:
                maxlstreak = lstreak
            lstreak = 0
            wstreak += 1
            ws = ws.replace(ws, "{}W".format(wstreak))
    if lstreak != 0:
        streak += " " + ls
    if lstreak > maxlstreak:
        maxlstreak = lstreak
    if wstreak != 0:
        streak += " " + ws
    if wstreak > maxwstreak:
        maxwstreak = wstreak
    return [streak, maxwstreak, maxlstreak]


def streakf1f4(dfff):
    # find streaks and maxstreaklengths
    streak = ""
    ws = ""
    shps = ""
    thps = ""
    ls = ""
    Placestreak = 0
    Noplacestreak = 0
    lstreak = 0
    wstreak = 0
    thpstreak = 0
    shpstreak = 0
    maxPS = 0
    maxNPS = 0
    sumPS = 0
    sumNPS = 0
    numPS = 0
    numNPS = 0
    for i in dfff['RESULT'][::-1]:
        if i >= 4 or i == 0:
            if wstreak != 0:
                streak += " " + ws
            elif shpstreak != 0:
                streak += " " + shps
            elif thpstreak != 0:
                streak += " " + thps
            if Placestreak > 0:
                numPS += 1
                sumPS += Placestreak
            if Placestreak > maxPS:
                maxPS = Placestreak
            wstreak = 0
            shpstreak = 0
            thpstreak = 0
            Placestreak = 0
            Noplacestreak += 1
            lstreak += 1
            ls = ls.replace(ls, "{}L".format(lstreak))
        if i < 4 and i > 0:
            if lstreak != 0:
                streak += " " + ls
            if Noplacestreak > 0:
                numNPS += 1
                sumNPS += Noplacestreak
            if Noplacestreak > maxNPS:
                maxNPS = Noplacestreak
            lstreak = 0
            Noplacestreak = 0
            Placestreak += 1
            if i == 1:
                if shpstreak != 0:
                    streak += " " + shps
                elif thpstreak != 0:
                    streak += " " + thps
                shpstreak = 0
                thpstreak = 0
                wstreak += 1
                ws = ws.replace(ws, "{}W".format(wstreak))
            elif i == 3:
                if wstreak != 0:
                    streak += " " + ws
                elif shpstreak != 0:
                    streak += " " + shps
                wstreak = 0
                shpstreak = 0
                thpstreak += 1
                thps = thps.replace(thps, "{}THP".format(thpstreak))
            elif i == 2:
                if wstreak != 0:
                    streak += " " + ws
                elif thpstreak != 0:
                    streak += " " + thps
                wstreak = 0
                thpstreak = 0
                shpstreak += 1
                shps = shps.replace(shps, "{}SHP".format(shpstreak))
    if lstreak != 0:
        streak += " " + ls
    if Noplacestreak > 0:
        numNPS += 1
        sumNPS += Noplacestreak
    if Noplacestreak > maxNPS:
        maxNPS = Noplacestreak
    if wstreak != 0:
        streak += " " + ws
    if Placestreak > 0:
        numPS += 1
        sumPS += Placestreak
    if Placestreak > maxPS:
        maxPS = Placestreak
    if shpstreak != 0:
        streak += " " + shps
    if thpstreak != 0:
        streak += " " + thps
    avgPS = 0
    avgNPS = 0
    if numPS > 0:
        avgPS = sumPS / numPS
    if numNPS > 0:
        avgNPS = sumNPS / numNPS
    # print(streak)
    return [streak, maxPS, maxNPS, avgPS, avgNPS]


def generate_li_adv(n, pl, odds, box_number):
    if box_number % 125 >= 25:
        if box_number % 125 == 0 and box_number != 0:
            if pl == 1:
                return html.Li("W",
                               className='form_guide__outcome form_guide__outcome--win',
                               style={"margin-top": "15px"})
            elif pl == 0:
                return html.Li("L",
                               className='form_guide__outcome form_guide__outcome--loss',
                               style={"margin-top": "15px"})

        elif box_number % 25 == 0 and box_number != 0:
            if pl == 1:
                return html.Li("W",
                               className='form_guide__outcome form_guide__outcome--win last')
            elif pl == 0:
                return html.Li("L",
                               className='form_guide__outcome form_guide__outcome--loss last')
        elif box_number % 5 == 0 and box_number != 0:
            if pl == 1:
                return html.Li("W",
                               className='form_guide__outcome form_guide__outcome--win',
                               style={"margin-left": "20px"})
            elif pl == 0:
                return html.Li("L",
                               className='form_guide__outcome form_guide__outcome--loss',
                               style={"margin-left": "20px"})
        else:
            if pl == 1:
                return html.Li("W",
                               className='form_guide__outcome form_guide__outcome--win')
            elif pl == 0:
                return html.Li("L",
                               className='form_guide__outcome form_guide__outcome--loss')

    else:
        if box_number % 125 == 0 and box_number != 0:
            if pl == 1:
                return html.Li("W",
                               className='form_guide__outcome form_guide__outcome--win',
                               style={"margin-top": "15px"})
            elif pl == 0:
                return html.Li("L",
                               className='form_guide__outcome form_guide__outcome--loss',
                               style={"margin-top": "15px"})

        elif box_number % 25 == 0 and box_number != 0:
            if pl == 1:
                return html.Li("W",
                               className='form_guide__outcome form_guide__outcome--win last',
                               style={"margin-top": "15px"})
            elif pl == 0:
                return html.Li("L",
                               className='form_guide__outcome form_guide__outcome--loss last',
                               style={"margin-top": "15px"})
        elif box_number % 5 == 0 and box_number != 0:
            if pl == 1:
                return html.Li("W",
                               className='form_guide__outcome form_guide__outcome--win',
                               style={"margin-left": "20px", "margin-top": "15px"})
            elif pl == 0:
                return html.Li("L",
                               className='form_guide__outcome form_guide__outcome--loss',
                               style={"margin-left": "20px", "margin-top": "15px"})
        else:
            if pl == 1:
                return html.Li("W",
                               className='form_guide__outcome form_guide__outcome--win',
                               style={"margin-top": "15px"})
            elif pl == 0:
                return html.Li("L",
                               className='form_guide__outcome form_guide__outcome--loss',
                               style={"margin-top": "15px"})


def generate_li_adv_f1f4(n, pl, odds, box_number):
    codes = ["W", "SHP", "THP", "0"]
    if box_number % 125 >= 25:
        if box_number % 125 == 0 and box_number != 0:
            if pl == 1:
                return html.Li("W",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--win',
                               style={"margin-top": "15px"})
            elif pl == 2:
                return html.Li("SHP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--shp',
                               style={"margin-top": "15px"})
            elif pl == 3:
                return html.Li("THP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--thp',
                               style={"margin-top": "15px"})
            else:
                return html.Li("{}".format(pl),
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--bo',
                               style={"margin-top": "15px"})
        elif box_number % 25 == 0 and box_number != 0:
            if pl == 1:
                return html.Li("W",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--win')
            elif pl == 2:
                return html.Li("SHP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--shp')
            elif pl == 3:
                return html.Li("THP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--thp')
            else:
                return html.Li("{}".format(pl),
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--bo')
        elif box_number % 5 == 0 and box_number != 0:
            if pl == 1:
                return html.Li("W",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--win',
                               style={"margin-left": "20px"})
            elif pl == 2:
                return html.Li("SHP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--shp',
                               style={"margin-left": "20px"})
            elif pl == 3:
                return html.Li("THP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--thp',
                               style={"margin-left": "20px"})
            else:
                return html.Li("{}".format(pl),
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--bo',
                               style={"margin-left": "20px"})
        else:
            if pl == 1:
                return html.Li("W",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--win')
            elif pl == 2:
                return html.Li("SHP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--shp')
            elif pl == 3:
                return html.Li("THP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--thp')
            else:
                return html.Li("{}".format(pl),
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--bo')
    else:
        if box_number % 125 == 0 and box_number != 0:
            if pl == 1:
                return html.Li("W",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--win',
                               style={"margin-top": "15px"})
            elif pl == 2:
                return html.Li("SHP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--shp',
                               style={"margin-top": "15px"})
            elif pl == 3:
                return html.Li("THP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--thp',
                               style={"margin-top": "15px"})
            else:
                return html.Li("{}".format(pl),
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--bo',
                               style={"margin-top": "15px"})
        elif box_number % 25 == 0 and box_number != 0:
            if pl == 1:
                return html.Li("W",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--win',
                               style={"margin-top": "15px"})
            elif pl == 2:
                return html.Li("SHP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--shp',
                               style={"margin-top": "15px"})
            elif pl == 3:
                return html.Li("THP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--thp',
                               style={"margin-top": "15px"})
            else:
                return html.Li("{}".format(pl),
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--bo',
                               style={"margin-top": "15px"})
        elif box_number % 5 == 0 and box_number != 0:
            if pl == 1:
                return html.Li("W",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--win',
                               style={"margin-top": "15px", "margin-left": "20px"})
            elif pl == 2:
                return html.Li("SHP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--shp',
                               style={"margin-top": "15px", "margin-left": "20px"})
            elif pl == 3:
                return html.Li("THP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--thp',
                               style={"margin-top": "15px", "margin-left": "20px"})
            else:
                return html.Li("{}".format(pl),
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--bo',
                               style={"margin-top": "15px", "margin-left": "20px"})
        else:
            if pl == 1:
                return html.Li("W",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--win',
                               style={"margin-top": "15px"})
            elif pl == 2:
                return html.Li("SHP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--shp',
                               style={"margin-top": "15px"})
            elif pl == 3:
                return html.Li("THP",
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--thp',
                               style={"margin-top": "15px"})
            else:
                return html.Li("{}".format(pl),
                               className='form_guide_f1f4__outcome form_guide_f1f4__outcome--bo',
                               style={"margin-top": "15px"})


def table_counts(df, odds, f):
    return df[odds].value_counts().get(f, 0)


def unique_non_null(s):
    return s.dropna().unique()


def generate_overall_scoreboard(overall_dff, dff):
    Overall_SB = [html.Div(id='selected_filters'),
                  html.Hr(className='othr'),
                  html.Table([
                      html.Thead([
                          html.Tr([
                              html.Th("", id='overall_header'),
                              html.Th("TR", id='overall_TR_header'),
                              html.Th("TR%", id='overall_TRpct_header'),
                              html.Th("WIN", id='overall_first_header'),
                              html.Th('W%', id='overall_firstpct_header'),
                              html.Th("SHP", id='overall_second_header'),
                              html.Th('2%', id='overall_secondpct_header'),
                              html.Th("THP", id='overall_third_header'),
                              html.Th('3%', id='overall_thirdpct_header'),
                              html.Th('Plc', id='overall_place_header'),
                              html.Th('P%', id='overall_placepct_header'),
                              html.Th('BO', id='overall_loss_header'),
                              html.Th("LWS(WIN)", id='overall_longest_winning_streak_header'),
                              html.Th("LLS(WIN)", id='overall_longest_losing_streak_header')
                          ])
                      ]),
                      html.Tr([
                          html.Td("Overall", className='type_header'),
                          html.Td("{}".format(len(overall_dff))),
                          html.Td("100%"),
                          html.Td(html.Button("{}".format(len(overall_dff.loc[overall_dff['RESULT'] == 1])),
                                              id='overall_first_button',
                                              n_clicks=0),
                                  id='overall_first_pos'),
                          html.Td("{0:.2f}".format(
                              (len(overall_dff.loc[overall_dff['RESULT'] == 1]) / len(overall_dff)) * 100),
                                  id='overall_first_pct'),
                          html.Td(html.Button("{}".format(len(overall_dff.loc[overall_dff['RESULT'] == 2])),
                                              id='overall_second_button',
                                              n_clicks=0),
                                  id='overall_second_pos'),
                          html.Td("{0:.2f}".format(
                              (len(overall_dff.loc[overall_dff['RESULT'] == 2]) / len(overall_dff)) * 100),
                                  id='overall_second_pct'),
                          html.Td(html.Button("{}".format(len(overall_dff.loc[overall_dff['RESULT'] == 3])),
                                              id='overall_third_button',
                                              n_clicks=0),
                                  id='overall_third_pos'),
                          html.Td("{0:.2f}".format(
                              (len(overall_dff.loc[overall_dff['RESULT'] == 3]) / len(overall_dff)) * 100),
                                  id='overall_third_pct'),
                          html.Td(html.Button("{}".format((len(overall_dff.loc[overall_dff['RESULT'] == 1]) + len(
                              overall_dff.loc[overall_dff['RESULT'] == 2]) + len(
                              overall_dff.loc[overall_dff['RESULT'] == 3]))),
                                              id='overall_place_button', n_clicks=0),
                                  id='overall_place_pos'),
                          html.Td("{0:.2f}".format(
                              ((len(overall_dff.loc[overall_dff['RESULT'] == 1]) + len(
                                  overall_dff.loc[overall_dff['RESULT'] == 2]) + len(
                                  overall_dff.loc[overall_dff['RESULT'] == 3])) / len(overall_dff)) * 100),
                              id='overall_place_pct'),
                          html.Td(html.Button("{}".format(len(overall_dff) - (
                                  len(overall_dff.loc[overall_dff['RESULT'] == 1]) + len(
                              overall_dff.loc[overall_dff['RESULT'] == 2]) + len(
                              overall_dff.loc[overall_dff['RESULT'] == 3]))), id='overall_loss_button', n_clicks=0),
                                  id='loss'),
                          html.Td("{}".format(maxminstreaks(1, overall_dff)[0]),
                                  id='overall_longest_winning_streak_data'),
                          html.Td("{}".format(maxminstreaks(1, overall_dff)[1]),
                                  id='overall_longest_losing_streak_data')
                      ]),
                      html.Tr([
                          html.Td("Filtered", className='type_header'),
                          html.Td("{}".format(len(dff))),
                          html.Td("{}%".format(round(((len(dff) / len(overall_dff)) * 100), 2))),
                          html.Td(
                              html.Button("{}".format(len(dff.loc[dff['RESULT'] == 1])), id='first_button', n_clicks=0),
                              id='first_pos'),
                          html.Td("{0:.2f}".format((len(dff.loc[dff['RESULT'] == 1]) / len(dff)) * 100),
                                  id='first_pct'),
                          html.Td(html.Button("{}".format(len(dff.loc[dff['RESULT'] == 2])), id='second_button',
                                              n_clicks=0),
                                  id='second_pos'),
                          html.Td("{0:.2f}".format((len(dff.loc[dff['RESULT'] == 2]) / len(dff)) * 100),
                                  id='second_pct'),
                          html.Td(
                              html.Button("{}".format(len(dff.loc[dff['RESULT'] == 3])), id='third_button', n_clicks=0),
                              id='third_pos'),
                          html.Td("{0:.2f}".format((len(dff.loc[dff['RESULT'] == 3]) / len(dff)) * 100),
                                  id='third_pct'),
                          html.Td(html.Button("{}".format((len(dff.loc[dff['RESULT'] == 1]) + len(
                              dff.loc[dff['RESULT'] == 2]) + len(dff.loc[dff['RESULT'] == 3]))), id='place_button',
                                              n_clicks=0),
                                  id='place_pos'),
                          html.Td("{0:.2f}".format(
                              ((len(dff.loc[dff['RESULT'] == 1]) + len(dff.loc[dff['RESULT'] == 2]) + len(
                                  dff.loc[dff['RESULT'] == 3])) / len(dff)) * 100), id='place_pct'),
                          html.Td(html.Button("{}".format(len(dff) - (
                                  len(dff.loc[dff['RESULT'] == 1]) + len(dff.loc[dff['RESULT'] == 2]) + len(
                              dff.loc[dff['RESULT'] == 3]))), id='bo_button', n_clicks=0),
                                  id='loss'),
                          html.Td("{}".format(maxminstreaks(1, dff)[0]), id='longest_winning_streak_data'),
                          html.Td("{}".format(maxminstreaks(1, dff)[1]), id='longest_losing_streak_data')
                      ])
                  ], id='overall_sb_table')]
    return Overall_SB


def generate_overall_scoreboard_for_combo(overall_df, df, combination):
    Overall_SB = [html.Div(id='selected_filters'),
                  html.Hr(className='othr'),
                  html.Table([
                      html.Thead([
                          html.Tr([
                              html.Th("", id='overall_header'),
                              html.Th("TR", id='overall_TR_header'),
                              html.Th("TR%", id='overall_TRpct_header'),
                              html.Th("WIN", id='overall_first_header'),
                              html.Th('W%', id='overall_firstpct_header'),
                              html.Th("LOSS", id='overall_loss_header'),
                              html.Th('L%', id='overall_loss_header'),
                              html.Th("LWS", id='overall_longest_winning_streak_header'),
                              html.Th("LLS", id='overall_longest_losing_streak_header')
                          ])
                      ]),
                      html.Tr([
                          html.Td("Overall", className='type_header'),
                          html.Td("{}".format(len(overall_df))),
                          html.Td("100%"),
                          html.Td(html.Button("{}".format(len(overall_df.loc[overall_df[combination] == 1])),
                                              id='overall_first_button',
                                              n_clicks=0),
                                  id='overall_first_pos'),
                          html.Td("{0:.2f}".format(
                              (len(overall_df.loc[overall_df[combination] == 1]) / len(overall_df)) * 100),
                                  id='overall_first_pct'),
                          html.Td(html.Button("{}".format(len(overall_df.loc[overall_df[combination] == 0])),
                                              id='overall_loss_button',
                                              n_clicks=0),
                                  id='overall_los_pos'),
                          html.Td("{0:.2f}".format(
                              (len(overall_df.loc[overall_df[combination] == 0]) / len(overall_df)) * 100),
                                  id='overall_loss_pct'),
                          html.Td("{}".format(streak(overall_df[combination])[1]),
                                  id='overall_longest_winning_streak_data'),
                          html.Td("{}".format(streak(overall_df[combination])[2]),
                                  id='overall_longest_losing_streak_data')
                      ]),
                      html.Tr([
                          html.Td("Filtered", className='type_header'),
                          html.Td("{}".format(len(df))),
                          html.Td("{}%".format(round(((len(df) / len(overall_df)) * 100), 2))),
                          html.Td(
                              html.Button("{}".format(len(df.loc[df[combination] == 1])), id='win_button', n_clicks=0),
                              id='first_pos'),
                          html.Td("{0:.2f}".format((len(df.loc[df[combination] == 1]) / len(df)) * 100), id='win_pct'),
                          html.Td(
                              html.Button("{}".format(len(df.loc[df[combination] == 0])), id='loss_button', n_clicks=0),
                              id='loss_pos'),
                          html.Td("{0:.2f}".format((len(df.loc[df[combination] == 0]) / len(df)) * 100), id='loss_pct'),
                          html.Td("{}".format(streak(df[combination])[1]), id='longest_winning_streak_data'),
                          html.Td("{}".format(streak(df[combination])[2]), id='longest_losing_streak_data')
                      ])
                  ], id='overall_sb_table')]
    return Overall_SB


def generate_f1f4_stats(dff):
    foo = ['F1', 'F2', 'F3', 'F4']
    pl = [1, 2, 3, 4, 0]
    n = dff.groupby(["F", "RESULT"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = dff.groupby(["F"], as_index=True).size().to_frame()
    k = dff.groupby(["F"], as_index=True)['RESULT'].agg([maxwinstreaks, maxlosestreaks])
    k.rename(columns={'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'}, inplace=True)
    fodf = pd.concat([n, m, k], axis=1)
    fodf.columns = [1, 2, 3, 4, 0, 'TR', 'LWS', 'LLS']
    fodf.rename(columns={1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'}, inplace=True)
    fodf["W%"] = ((fodf["W"] / fodf['TR']) * 100).round(2)
    fodf["SHP%"] = ((fodf["SHP"] / fodf['TR']) * 100).round(2)
    fodf["THP%"] = ((fodf["THP"] / fodf['TR']) * 100).round(2)
    fodf["Plc%"] = (((fodf["W"] + fodf["SHP"] + fodf["THP"]) / fodf['TR']) * 100).round(2)
    fodf["Plc"] = fodf["W"] + fodf["SHP"] + fodf["THP"]
    fodf["BO"] = fodf["TR"] - fodf["Plc"]
    # fodf["TR%"] = ((fodf["TR"] / len(df)) * 100 *4).round(2)
    f = [i for i in foo if i in fodf.index]
    fodf = fodf.reindex(f)
    fodf = fodf[['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']]
    return fodf


def generate_f1f4_season_stats(dff):
    foo = ['F1', 'F2', 'F3', 'F4']
    pl = [1, 2, 3, 4, 0]
    n = dff.groupby(["F", "Season Code", "RESULT"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = dff.groupby(["F", "Season Code"], as_index=True).size().to_frame()
    k = dff.groupby(["F", "Season Code"], as_index=True)['RESULT'].agg([maxwinstreaks, maxlosestreaks])
    k.rename(columns={'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'}, inplace=True)
    fseasondf = pd.concat([n, m, k], axis=1)
    fseasondf.columns = [1, 2, 3, 4, 0, 'TR', 'LWS', 'LLS']
    fseasondf.rename(columns={1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'},
                     inplace=True)
    fseasondf["W%"] = ((fseasondf["W"] / fseasondf['TR']) * 100).round(2)
    fseasondf["SHP%"] = ((fseasondf["SHP"] / fseasondf['TR']) * 100).round(2)
    fseasondf["THP%"] = ((fseasondf["THP"] / fseasondf['TR']) * 100).round(2)
    fseasondf["Plc%"] = (((fseasondf["W"] + fseasondf["SHP"] + fseasondf["THP"]) / fseasondf['TR']) * 100).round(2)
    fseasondf["Plc"] = fseasondf["W"] + fseasondf["SHP"] + fseasondf["THP"]
    fseasondf["BO"] = fseasondf["TR"] - fseasondf["Plc"]
    fseasondf = fseasondf[['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']]
    return fseasondf


def generate_f1f4_distance_stats(dff):
    foo = ['F1', 'F2', 'F3', 'F4']
    pl = [1, 2, 3, 4, 0]
    n = dff.groupby(["F", "Distance", "RESULT"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = dff.groupby(["F", "Distance"], as_index=True).size().to_frame()
    k = dff.groupby(["F", "Distance"], as_index=True)['RESULT'].agg([maxwinstreaks, maxlosestreaks])
    k.rename(columns={'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'}, inplace=True)
    fdistancedf = pd.concat([n, m, k], axis=1)

    fdistancedf.columns = [1, 2, 3, 4, 0, 'TR', 'LWS', 'LLS']
    fdistancedf.rename(columns={1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'},
                       inplace=True)
    fdistancedf["W%"] = ((fdistancedf["W"] / fdistancedf['TR']) * 100).round(2)
    fdistancedf["SHP%"] = ((fdistancedf["SHP"] / fdistancedf['TR']) * 100).round(2)
    fdistancedf["THP%"] = ((fdistancedf["THP"] / fdistancedf['TR']) * 100).round(2)
    fdistancedf["Plc%"] = (
            ((fdistancedf["W"] + fdistancedf["SHP"] + fdistancedf["THP"]) / fdistancedf['TR']) * 100).round(2)
    fdistancedf["Plc"] = fdistancedf["W"] + fdistancedf["SHP"] + fdistancedf["THP"]
    fdistancedf["BO"] = fdistancedf["TR"] - fdistancedf["Plc"]
    fdistancedf = fdistancedf[['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']]
    return fdistancedf


def generate_f1f4_rno_stats(dff):
    foo = ['F1', 'F2', 'F3', 'F4']
    pl = [1, 2, 3, 4, 0]
    n = dff.groupby(["F", "R No.", "RESULT"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = dff.groupby(["F", "R No."], as_index=True).size().to_frame()
    k = dff.groupby(["F", "R No."], as_index=True)['RESULT'].agg([maxwinstreaks, maxlosestreaks])
    k.rename(columns={'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'}, inplace=True)
    fclassdf = pd.concat([n, m, k], axis=1)

    fclassdf.columns = [1, 2, 3, 4, 0, 'TR', 'LWS', 'LLS']
    fclassdf.rename(columns={1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'}, inplace=True)
    fclassdf["W%"] = ((fclassdf["W"] / fclassdf['TR']) * 100).round(2)
    fclassdf["SHP%"] = ((fclassdf["SHP"] / fclassdf['TR']) * 100).round(2)
    fclassdf["THP%"] = ((fclassdf["THP"] / fclassdf['TR']) * 100).round(2)
    fclassdf["Plc%"] = (((fclassdf["W"] + fclassdf["SHP"] + fclassdf["THP"]) / fclassdf['TR']) * 100).round(2)
    fclassdf["Plc"] = fclassdf["W"] + fclassdf["SHP"] + fclassdf["THP"]
    fclassdf["BO"] = fclassdf["TR"] - fclassdf["Plc"]
    fclassdf = fclassdf[['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']]
    return fclassdf


def generate_f1f4_class_stats(dff):
    foo = ['F1', 'F2', 'F3', 'F4']
    pl = [1, 2, 3, 4, 0]
    n = dff.groupby(["F", "Class", "RESULT"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = dff.groupby(["F", "Class"], as_index=True).size().to_frame()
    k = dff.groupby(["F", "Class"], as_index=True)['RESULT'].agg([maxwinstreaks, maxlosestreaks])
    k.rename(columns={'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'}, inplace=True)
    fclassdf = pd.concat([n, m, k], axis=1)

    fclassdf.columns = [1, 2, 3, 4, 0, 'TR', 'LWS', 'LLS']
    fclassdf.rename(columns={1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'}, inplace=True)
    fclassdf["W%"] = ((fclassdf["W"] / fclassdf['TR']) * 100).round(2)
    fclassdf["SHP%"] = ((fclassdf["SHP"] / fclassdf['TR']) * 100).round(2)
    fclassdf["THP%"] = ((fclassdf["THP"] / fclassdf['TR']) * 100).round(2)
    fclassdf["Plc%"] = (((fclassdf["W"] + fclassdf["SHP"] + fclassdf["THP"]) / fclassdf['TR']) * 100).round(2)
    fclassdf["Plc"] = fclassdf["W"] + fclassdf["SHP"] + fclassdf["THP"]
    fclassdf["BO"] = fclassdf["TR"] - fclassdf["Plc"]
    fclassdf = fclassdf[['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']]
    return fclassdf


def generate_f1f4_generic_stats(dff, column):
    foo = ['F1', 'F2', 'F3', 'F4']
    pl = [1, 2, 3, 4, 0]
    n = dff.groupby(["F", column, "RESULT"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = dff.groupby(["F", column], as_index=True).size().to_frame()
    k = dff.groupby(["F", column], as_index=True)['RESULT'].agg([maxwinstreaks, maxlosestreaks])
    k.rename(columns={'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'}, inplace=True)
    fdistancedf = pd.concat([n, m, k], axis=1)

    fdistancedf.columns = [1, 2, 3, 4, 0, 'TR', 'LWS', 'LLS']
    fdistancedf.rename(columns={1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'},
                       inplace=True)
    fdistancedf["W%"] = ((fdistancedf["W"] / fdistancedf['TR']) * 100).round(2)
    fdistancedf["SHP%"] = ((fdistancedf["SHP"] / fdistancedf['TR']) * 100).round(2)
    fdistancedf["THP%"] = ((fdistancedf["THP"] / fdistancedf['TR']) * 100).round(2)
    fdistancedf["Plc%"] = (
            ((fdistancedf["W"] + fdistancedf["SHP"] + fdistancedf["THP"]) / fdistancedf['TR']) * 100).round(2)
    fdistancedf["Plc"] = fdistancedf["W"] + fdistancedf["SHP"] + fdistancedf["THP"]
    fdistancedf["BO"] = fdistancedf["TR"] - fdistancedf["Plc"]
    fdistancedf = fdistancedf[['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']]
    return fdistancedf



def generate_favourite_scoreboard(xdf):
    header = html.H3("F1-F4 stats",
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    fav_sb = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in xdf[
            ['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']].reset_index().columns],
        data=xdf[
            ['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']].reset_index().to_dict(
            'records'),
        editable=False,
        export_columns='all',
        export_format='xlsx',
        # filterable=False,
        # filter_action="none",
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable=None,
        row_selectable=None,
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
        style_header={
            'backgroundColor': '#082255',
            'color': 'white',
            'fontWeight': 'bold',
            'font-size': 16,
            'textAlign': 'center'
        },
        style_cell={
            'backgroundColor': '#082255',
            'color': 'white',
            'border': '1px solid #bebebe',
            'font-size': 14,
            'textAlign': 'center',
            'minWidth': 40
        },
        style_cell_conditional=[
            {'if': {'column_id': 'Jockey'},
             'width': '150px'},
            {'if': {'column_id': 'Trainer'},
             'width': '150px'},
            {
                'if': {'column_id': 'W'},
                # 'backgroundColor': '#2ECC71',
                'color': '#2ECC71'
            },
            {
                'if': {'column_id': 'W%'},
                # 'backgroundColor': '#2ECC71',
                'color': '#2ECC71'
            },
            {
                'if': {'column_id': 'SHP'},
                # 'backgroundColor': '#2ECC71',
                'color': '#e8e40e'
            },
            {
                'if': {'column_id': 'SHP%'},
                # 'backgroundColor': '#2ECC71',
                'color': '#e8e40e'
            },
            {
                'if': {'column_id': 'THP'},
                # 'backgroundColor': '#2ECC71',
                'color': '#3498DB'
            },
            {
                'if': {'column_id': 'THP%'},
                # 'backgroundColor': '#2ECC71',
                'color': '#3498DB'
            },
            {
                'if': {'column_id': 'BO'},
                # 'backgroundColor': '#2ECC71',
                'color': '#ff2200'
            }
        ],
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{{W}} = {}'.format(xdf['W'].max()),
                    'column_id': 'W'
                },
                # 'border':'2px solid 2ECC71',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{W%}} = {}'.format(xdf['W%'].max()),
                    'column_id': 'W%'
                },
                # 'border':'2px solid 2ECC71',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{SHP}} = {}'.format(xdf['SHP'].max()),
                    'column_id': 'SHP'
                },
                # 'border': '2px solid e8e40e',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{SHP%}} = {}'.format(xdf['SHP%'].max()),
                    'column_id': 'SHP%'
                },
                # 'border': '2px solid e8e40e',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{THP}} = {}'.format(xdf['THP'].max()),
                    'column_id': 'THP'
                },
                # 'border': '2px solid 3498DB',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{THP%}} = {}'.format(xdf['THP%'].max()),
                    'column_id': 'THP%'
                },
                # 'border': '2px solid 3498DB',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{Plc}} = {}'.format(xdf['Plc'].max()),
                    'column_id': 'Plc'
                },
                # 'border': '2px solid white',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{Plc%}} = {}'.format(xdf['Plc%'].max()),
                    'column_id': 'Plc%'
                },
                # 'border': '2px solid white',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{LWS}} = {}'.format(xdf['LWS'].max()),
                    'column_id': 'LWS'
                },
                # 'border': '2px solid 2ECC71',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{LLS}} = {}'.format(xdf['LLS'].max()),
                    'column_id': 'LLS'
                },
                # 'border': '2px solid ff2200',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{BO}} = {}'.format(xdf['BO'].max()),
                    'column_id': 'BO'
                },
                # 'border': '2px solid ff2200',
                'font-weight': 'bold',
                'font-size': 18
            }
        ],
        style_filter={
            'textAlign': 'center'
        },
        style_as_list_view=True
    )

    return header, fav_sb


def generate_favourite_season_scoreboard(xdf, type):
    header = html.H3("{} stats by season".format(type),
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    fav_sb = dash_table.DataTable(
        id='table2',
        columns=[{"name": i, "id": i} for i in xdf[
            ['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']].reset_index().columns],
        data=xdf[
            ['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']].reset_index().to_dict(
            'records'),
        editable=False,
        export_columns='all',
        export_format='xlsx',
        # filterable=False,
        # filter_action="none",
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable=None,
        row_selectable=None,
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
        style_header={
            'backgroundColor': '#082255',
            'color': 'white',
            'fontWeight': 'bold',
            'font-size': 16,
            'textAlign': 'center'
        },
        style_cell={
            'backgroundColor': '#082255',
            'color': 'white',
            'border': '1px solid #bebebe',
            'font-size': 14,
            'textAlign': 'center',
            'minWidth': 40
        },
        style_cell_conditional=[
            {'if': {'column_id': 'Jockey'},
             'width': '150px'},
            {'if': {'column_id': 'Trainer'},
             'width': '150px'},
            {
                'if': {'column_id': 'W'},
                # 'backgroundColor': '#2ECC71',
                'color': '#2ECC71'
            },
            {
                'if': {'column_id': 'W%'},
                # 'backgroundColor': '#2ECC71',
                'color': '#2ECC71'
            },
            {
                'if': {'column_id': 'SHP'},
                # 'backgroundColor': '#2ECC71',
                'color': '#e8e40e'
            },
            {
                'if': {'column_id': 'SHP%'},
                # 'backgroundColor': '#2ECC71',
                'color': '#e8e40e'
            },
            {
                'if': {'column_id': 'THP'},
                # 'backgroundColor': '#2ECC71',
                'color': '#3498DB'
            },
            {
                'if': {'column_id': 'THP%'},
                # 'backgroundColor': '#2ECC71',
                'color': '#3498DB'
            },
            {
                'if': {'column_id': 'BO'},
                # 'backgroundColor': '#2ECC71',
                'color': '#ff2200'
            }
        ],
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{{W}} = {}'.format(xdf['W'].max()),
                    'column_id': 'W'
                },
                # 'border':'2px solid 2ECC71',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{W%}} = {}'.format(xdf['W%'].max()),
                    'column_id': 'W%'
                },
                # 'border':'2px solid 2ECC71',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{SHP}} = {}'.format(xdf['SHP'].max()),
                    'column_id': 'SHP'
                },
                # 'border': '2px solid e8e40e',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{SHP%}} = {}'.format(xdf['SHP%'].max()),
                    'column_id': 'SHP%'
                },
                # 'border': '2px solid e8e40e',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{THP}} = {}'.format(xdf['THP'].max()),
                    'column_id': 'THP'
                },
                # 'border': '2px solid 3498DB',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{THP%}} = {}'.format(xdf['THP%'].max()),
                    'column_id': 'THP%'
                },
                # 'border': '2px solid 3498DB',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{Plc}} = {}'.format(xdf['Plc'].max()),
                    'column_id': 'Plc'
                },
                # 'border': '2px solid white',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{Plc%}} = {}'.format(xdf['Plc%'].max()),
                    'column_id': 'Plc%'
                },
                # 'border': '2px solid white',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{LWS}} = {}'.format(xdf['LWS'].max()),
                    'column_id': 'LWS'
                },
                # 'border': '2px solid 2ECC71',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{LLS}} = {}'.format(xdf['LLS'].max()),
                    'column_id': 'LLS'
                },
                # 'border': '2px solid ff2200',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{BO}} = {}'.format(xdf['BO'].max()),
                    'column_id': 'BO'
                },
                # 'border': '2px solid ff2200',
                'font-weight': 'bold',
                'font-size': 18
            }
        ],
        style_filter={
            'textAlign': 'center'
        },
        style_as_list_view=True
    )

    return header, fav_sb


def generate_favourite_generic_scoreboard(xdf, type, idd, what):
    header = html.H3("{} stats by {}".format(type, what),
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    fav_sb = dash_table.DataTable(
        id=idd,
        columns=[{"name": i, "id": i} for i in xdf[
            ['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']].reset_index().columns],
        data=xdf[
            ['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']].reset_index().to_dict(
            'records'),
        editable=False,
        export_columns='all',
        export_format='xlsx',
        # filterable=False,
        # filter_action="none",
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable=None,
        row_selectable=None,
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
        style_header={
            'backgroundColor': '#082255',
            'color': 'white',
            'fontWeight': 'bold',
            'font-size': 16,
            'textAlign': 'center'
        },
        style_cell={
            'backgroundColor': '#082255',
            'color': 'white',
            'border': '1px solid #bebebe',
            'font-size': 14,
            'textAlign': 'center',
            'minWidth': 40
        },
        style_cell_conditional=[
            {'if': {'column_id': 'Jockey'},
             'width': '150px'},
            {'if': {'column_id': 'Trainer'},
             'width': '150px'},
            {
                'if': {'column_id': 'W'},
                # 'backgroundColor': '#2ECC71',
                'color': '#2ECC71'
            },
            {
                'if': {'column_id': 'W%'},
                # 'backgroundColor': '#2ECC71',
                'color': '#2ECC71'
            },
            {
                'if': {'column_id': 'SHP'},
                # 'backgroundColor': '#2ECC71',
                'color': '#e8e40e'
            },
            {
                'if': {'column_id': 'SHP%'},
                # 'backgroundColor': '#2ECC71',
                'color': '#e8e40e'
            },
            {
                'if': {'column_id': 'THP'},
                # 'backgroundColor': '#2ECC71',
                'color': '#3498DB'
            },
            {
                'if': {'column_id': 'THP%'},
                # 'backgroundColor': '#2ECC71',
                'color': '#3498DB'
            },
            {
                'if': {'column_id': 'BO'},
                # 'backgroundColor': '#2ECC71',
                'color': '#ff2200'
            }
        ],
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{{W}} = {}'.format(xdf['W'].max()),
                    'column_id': 'W'
                },
                # 'border':'2px solid 2ECC71',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{W%}} = {}'.format(xdf['W%'].max()),
                    'column_id': 'W%'
                },
                # 'border':'2px solid 2ECC71',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{SHP}} = {}'.format(xdf['SHP'].max()),
                    'column_id': 'SHP'
                },
                # 'border': '2px solid e8e40e',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{SHP%}} = {}'.format(xdf['SHP%'].max()),
                    'column_id': 'SHP%'
                },
                # 'border': '2px solid e8e40e',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{THP}} = {}'.format(xdf['THP'].max()),
                    'column_id': 'THP'
                },
                # 'border': '2px solid 3498DB',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{THP%}} = {}'.format(xdf['THP%'].max()),
                    'column_id': 'THP%'
                },
                # 'border': '2px solid 3498DB',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{Plc}} = {}'.format(xdf['Plc'].max()),
                    'column_id': 'Plc'
                },
                # 'border': '2px solid white',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{Plc%}} = {}'.format(xdf['Plc%'].max()),
                    'column_id': 'Plc%'
                },
                # 'border': '2px solid white',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{LWS}} = {}'.format(xdf['LWS'].max()),
                    'column_id': 'LWS'
                },
                # 'border': '2px solid 2ECC71',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{LLS}} = {}'.format(xdf['LLS'].max()),
                    'column_id': 'LLS'
                },
                # 'border': '2px solid ff2200',
                'font-weight': 'bold',
                'font-size': 18
            },
            {
                'if': {
                    'filter_query': '{{BO}} = {}'.format(xdf['BO'].max()),
                    'column_id': 'BO'
                },
                # 'border': '2px solid ff2200',
                'font-weight': 'bold',
                'font-size': 18
            }
        ],
        style_filter={
            'textAlign': 'center'
        },
        style_as_list_view=True
    )

    return header, fav_sb


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
                html.Div(id='scoreboard2', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboard3', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboard4', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboard5', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboardMonth', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboardYear', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboardNor', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboardHno', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboardAge', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboardWt', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboardSh', className='pretty_container', style={'display': 'initial'})
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
    html.Div(id="store_season_stats", style={'display': 'none'}),
    html.Div(id="store_class_stats", style={'display': 'none'}),
    html.Div(id="store_distance_stats", style={'display': 'none'}),
    html.Div(id="store_rno_stats", style={'display': 'none'}),
    html.Div(id="store_month_stats", style={'display': 'none'}),
    html.Div(id="store_year_stats", style={'display': 'none'}),
    html.Div(id="store_nor_stats", style={'display': 'none'}),
    html.Div(id="store_hno_stats", style={'display': 'none'}),
    html.Div(id="store_age_stats", style={'display': 'none'}),
    html.Div(id="store_wt_stats", style={'display': 'none'}),
    html.Div(id="store_sh_stats", style={'display': 'none'})],
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
     Output(component_id='scoreboard2', component_property='children'),
     Output(component_id='scoreboard2', component_property='style'),
     Output(component_id='graphslot1', component_property='children'),
     Output(component_id='row2', component_property='style'),
     Output(component_id='graphslot2', component_property='children'),
     Output(component_id='row3', component_property='style'),
     Output(component_id='store_working_overall_df', component_property='children'),
     Output(component_id='store_working_filtered_df', component_property='children'),
     Output("store_season_stats", component_property='children'),
     Output(component_id='scoreboard3', component_property='children'),
     Output(component_id='scoreboard3', component_property='style'),
     Output(component_id='scoreboard4', component_property='children'),
     Output(component_id='scoreboard4', component_property='style'),
     Output("store_distance_stats", component_property='children'),
     Output("store_class_stats", component_property='children'),
     Output(component_id='scoreboard5', component_property='children'),
     Output(component_id='scoreboard5', component_property='style'),
     Output("store_rno_stats", component_property='children'),
     Output(component_id='scoreboardMonth', component_property='children'),
     Output(component_id='scoreboardMonth', component_property='style'),
     Output(component_id='scoreboardYear', component_property='children'),
     Output(component_id='scoreboardYear', component_property='style'),
     Output(component_id='scoreboardNor', component_property='children'),
     Output(component_id='scoreboardNor', component_property='style'),
     Output(component_id='scoreboardHno', component_property='children'),
     Output(component_id='scoreboardHno', component_property='style'),
     Output(component_id='scoreboardAge', component_property='children'),
     Output(component_id='scoreboardAge', component_property='style'),
     Output(component_id='scoreboardWt', component_property='children'),
     Output(component_id='scoreboardWt', component_property='style'),
     Output(component_id='scoreboardSh', component_property='children'),
     Output(component_id='scoreboardSh', component_property='style'),
     Output("store_month_stats", component_property='children'),
     Output("store_year_stats", component_property='children'),
     Output("store_nor_stats", component_property='children'),
     Output("store_hno_stats", component_property='children'),
     Output("store_age_stats", component_property='children'),
     Output("store_wt_stats", component_property='children'),
     Output("store_sh_stats", component_property='children')
     ],
    [Input('howtype', 'value'),
     Input(component_id='type', component_property='value'),
     Input(component_id='combination', component_property='value'),
     Input(component_id='Centre', component_property='value'),
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
def update_main(how, type, combination, Centre, *searchparam):
    if Centre is None:
        print("Hello")
        raise PreventUpdate
    print("Type:", type)
    print("combo:", combination)
    print("centre:", Centre)
    print("extra:", searchparam)
    if Centre != 'All':
        centre_race_stats = race_stats[race_stats['Venue'] == Centre]
        centre_df = df[df['Venue'] == Centre]
    else:
        centre_race_stats = race_stats
        centre_df = df
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
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        filtered_race_stats = filtered_race_stats[conjunction(0, *condition)]
    for (key, value) in zip(cols_df, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "":
            for v in value:
                # print('V--->',v)
                if (v != "All" and v is not None and v != ""):
                    cond_df.append(filtered_df[key] == v)
            condition_df.append(conjunction(1, *cond_df))
            cond_df = []
    if condition_df:
        filtered_df = filtered_df[conjunction(0, *condition_df)]
    f1f4_stats = generate_f1f4_stats(filtered_df)
    f1f4_season_stats = generate_f1f4_season_stats(filtered_df)
    f1f4_distance_stats = generate_f1f4_distance_stats(filtered_df)
    f1f4_class_stats = generate_f1f4_class_stats(filtered_df)
    f1f4_rno_stats = generate_f1f4_rno_stats(filtered_df)
    f1f4_month_stats = generate_f1f4_generic_stats(filtered_df, "Month")
    f1f4_year_stats = generate_f1f4_generic_stats(filtered_df, "Year")
    f1f4_nor_stats = generate_f1f4_generic_stats(filtered_df, "NoR")
    f1f4_hno_stats = generate_f1f4_generic_stats(filtered_df, "H. No")
    f1f4_age_stats = generate_f1f4_generic_stats(filtered_df, "Age")
    f1f4_wt_stats = generate_f1f4_generic_stats(filtered_df, "Wt")
    f1f4_sh_stats = generate_f1f4_generic_stats(filtered_df, "Sh")
    if len(filtered_race_stats) == 0 or len(filtered_df) == 0:
        score_return = "No Races to Show"
        score2_return = []
        score2_style = {'display': 'none'}
        ov_score = []
        ov_style = {'display': 'none'}
        graph1 = []
        row2_style = {'display': 'none'}
        graph2 = []
        row3_style = {'display': 'none'}
        score3_return = []
        score3_style = {'display': 'none'}
        score4_return = []
        score4_style = {'display': 'none'}
        score5_return = []
        score5_style = {'display': 'none'}
        scoremonth_return = []
        scoremonth_style = {'display': 'none'}
        scoreyear_return = []
        scoreyear_style = {'display': 'none'}
        scorenor_return = []
        scorenor_style = {'display': 'none'}
        scorehno_return = []
        scorehno_style = {'display': 'none'}
        scoreage_return = []
        scoreage_style = {'display': 'none'}
        scorewt_return = []
        scorewt_style = {'display': 'none'}
        scoresh_return = []
        scoresh_style = {'display': 'none'}
    elif combination == [] or combination == "" or combination is None:
        score_return = generate_favourite_scoreboard(f1f4_stats)
        score2_return = generate_favourite_season_scoreboard(f1f4_season_stats.loc['F1'], 'F1')
        score2_style = {'display': 'initial'}
        ov_score = []
        ov_style = {'display': 'none'}
        graph1 = gen_overall_centre_table(filtered_race_stats)
        row2_style = {'display': 'initial'}
        graph2 = [html.Div(),
                  dcc.Graph(figure=generatebaroverall(filtered_race_stats), style={'height': '100%', 'width': '100%'})]
        row3_style = {'display': 'initial'}
        score3_return = generate_favourite_generic_scoreboard(f1f4_class_stats.loc['F1'], 'F1', 'table3', 'class')
        score3_style = {'display': 'initial'}
        score4_return = generate_favourite_generic_scoreboard(f1f4_distance_stats.loc['F1'], 'F1', 'table4', 'distance')
        score4_style = {'display': 'initial'}
        score5_return = generate_favourite_generic_scoreboard(f1f4_rno_stats.loc['F1'], 'F1', 'table5', 'Race No.')
        score5_style = {'display': 'initial'}

        scoremonth_return = generate_favourite_generic_scoreboard(f1f4_month_stats.loc['F1'], 'F1', 'table6', 'Month')
        scoremonth_style = {'display': 'initial'}
        scoreyear_return = generate_favourite_generic_scoreboard(f1f4_year_stats.loc['F1'], 'F1', 'table7', 'Year')
        scoreyear_style = {'display': 'initial'}
        scorenor_return = generate_favourite_generic_scoreboard(f1f4_nor_stats.loc['F1'], 'F1', 'table8', 'NoR')
        scorenor_style = {'display': 'initial'}
        scorehno_return = generate_favourite_generic_scoreboard(f1f4_hno_stats.loc['F1'], 'F1', 'table9', 'H. No')
        scorehno_style = {'display': 'initial'}
        scoreage_return = generate_favourite_generic_scoreboard(f1f4_age_stats.loc['F1'], 'F1', 'table10', 'Age')
        scoreage_style = {'display': 'initial'}
        scorewt_return = generate_favourite_generic_scoreboard(f1f4_wt_stats.loc['F1'], 'F1', 'table11', 'Wt')
        scorewt_style = {'display': 'initial'}
        scoresh_return = generate_favourite_generic_scoreboard(f1f4_sh_stats.loc['F1'], 'F1', 'table12', 'Sh')
        scoresh_style = {'display': 'initial'}
    elif combination not in ['F1', 'F2', 'F3', 'F4'] and how != 'multi':
        print("comnination is in those 4 NOT!!!!!!!!!!")
        score_return = [
            html.Div(id="form", children=[
                html.Div('Form Guide(W):', className='form_guide_header'),
                html.Ul(id='form_guide_ul',
                        className='form_guide_ul_win',
                        children=[generate_li_adv(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(filtered_race_stats[combination][::-1],
                                                filtered_race_stats[combination][::-1]))]
                        )]),
            html.Div(
                [html.Div("Streaks:", className='streak_header'), "{}".format(streak(filtered_race_stats['F12'])[0])],
                id='streak_display')
        ]
        score2_return = []
        score2_style = {'display': 'none'}
        ov_score = generate_overall_scoreboard_for_combo(centre_race_stats, filtered_race_stats, combination)
        ov_style = {'display': 'initial'}
        graph1 = []
        row2_style = {'display': 'none'}
        graph2 = []
        row3_style = {'display': 'none'}
        score3_return = []
        score3_style = {'display': 'none'}
        score4_return = []
        score4_style = {'display': 'none'}
        score5_return = []
        score5_style = {'display': 'none'}
        scoremonth_return = []
        scoremonth_style = {'display': 'none'}
        scoreyear_return = []
        scoreyear_style = {'display': 'none'}
        scorenor_return = []
        scorenor_style = {'display': 'none'}
        scorehno_return = []
        scorehno_style = {'display': 'none'}
        scoreage_return = []
        scoreage_style = {'display': 'none'}
        scorewt_return = []
        scorewt_style = {'display': 'none'}
        scoresh_return = []
        scoresh_style = {'display': 'none'}
    elif combination in ['F1', 'F2', 'F3', 'F4']:
        print("comnination is in those 4")
        filtered_df = filtered_df[filtered_df['F'] == combination]
        centre_df = centre_df[centre_df['F'] == combination]
        score_return = [
            html.Div(id="form", children=[
                html.Div('Form Guide:', className='form_guide_header'),
                html.Ul(id='form_guide_ul',
                        className='form_guide_ul_first',
                        children=[generate_li_adv_f1f4(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(filtered_df['RESULT'][::-1],
                                                filtered_df['RESULT'][::-1]))]
                        )]),
            html.Div(
                [html.Div("Streaks:", className='streak_header'), "{}".format(streakf1f4(filtered_df)[0])],
                id='streak_display')
        ]
        score2_return = []
        score2_style = {'display': 'none'}
        ov_score = generate_overall_scoreboard(centre_df, filtered_df)
        ov_style = {'display': 'initial'}
        graph1 = []
        row2_style = {'display': 'none'}
        graph2 = []
        row3_style = {'display': 'none'}
        score3_return = []
        score3_style = {'display': 'none'}
        score4_return = []
        score4_style = {'display': 'none'}
        score5_return = []
        score5_style = {'display': 'none'}
        scoremonth_return = []
        scoremonth_style = {'display': 'none'}
        scoreyear_return = []
        scoreyear_style = {'display': 'none'}
        scorenor_return = []
        scorenor_style = {'display': 'none'}
        scorehno_return = []
        scorehno_style = {'display': 'none'}
        scoreage_return = []
        scoreage_style = {'display': 'none'}
        scorewt_return = []
        scorewt_style = {'display': 'none'}
        scoresh_return = []
        scoresh_style = {'display': 'none'}
    elif how == 'multi':  #################do here
        try:
            combo = ""
            print("combolist")
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
            print("combo is {}".format(combo))
            score_return = [
                html.Div(id="form", children=[
                    html.Div('Form Guide(W):', className='form_guide_header'),
                    html.Ul(id='form_guide_ul',
                            className='form_guide_ul_win',
                            children=[generate_li_adv(1, i, j, n) for n, (i, j) in
                                      enumerate(zip(filtered_race_stats[combo][::-1],
                                                    filtered_race_stats[combo][::-1]))]
                            )]),
                html.Div(
                    [html.Div("Streaks:", className='streak_header'),
                     "{}".format(streak(filtered_race_stats['F12'])[0])],
                    id='streak_display')
            ]
            score2_return = []
            score2_style = {'display': 'none'}
            ov_score = generate_overall_scoreboard_for_combo(centre_race_stats, filtered_race_stats, combo)
            ov_style = {'display': 'initial'}
            graph1 = []
            row2_style = {'display': 'none'}
            graph2 = []
            row3_style = {'display': 'none'}
            score3_return = []
            score3_style = {'display': 'none'}
            score4_return = []
            score4_style = {'display': 'none'}
            score5_return = []
            score5_style = {'display': 'none'}
            scoremonth_return = []
            scoremonth_style = {'display': 'none'}
            scoreyear_return = []
            scoreyear_style = {'display': 'none'}
            scorenor_return = []
            scorenor_style = {'display': 'none'}
            scorehno_return = []
            scorehno_style = {'display': 'none'}
            scoreage_return = []
            scoreage_style = {'display': 'none'}
            scorewt_return = []
            scorewt_style = {'display': 'none'}
            scoresh_return = []
            scoresh_style = {'display': 'none'}
        except:
            raise PreventUpdate

    return [ov_score, ov_style, score_return, score2_return, score2_style, graph1, row2_style, graph2, row3_style,
            [centre_df.to_json(orient='split')], [filtered_df.to_json(orient='split')],
            [f1f4_season_stats.index.names, f1f4_season_stats.reset_index().to_json(orient='split')],
            score3_return, score3_style, score4_return, score4_style,
            [f1f4_distance_stats.index.names, f1f4_distance_stats.reset_index().to_json(orient='split')],
            [f1f4_class_stats.index.names, f1f4_class_stats.reset_index().to_json(orient='split')],
            score5_return, score5_style,
            [f1f4_rno_stats.index.names, f1f4_rno_stats.reset_index().to_json(orient='split')],
            scoremonth_return, scoremonth_style,
            scoreyear_return, scoreyear_style,
            scorenor_return, scorenor_style,
            scorehno_return, scorehno_style,
            scoreage_return, scoreage_style,
            scorewt_return, scorewt_style,
            scoresh_return, scoresh_style,
            [f1f4_month_stats.index.names, f1f4_month_stats.reset_index().to_json(orient='split')],
            [f1f4_year_stats.index.names, f1f4_year_stats.reset_index().to_json(orient='split')],
            [f1f4_nor_stats.index.names, f1f4_nor_stats.reset_index().to_json(orient='split')],
            [f1f4_hno_stats.index.names, f1f4_hno_stats.reset_index().to_json(orient='split')],
            [f1f4_age_stats.index.names, f1f4_age_stats.reset_index().to_json(orient='split')],
            [f1f4_wt_stats.index.names, f1f4_wt_stats.reset_index().to_json(orient='split')],
            [f1f4_sh_stats.index.names, f1f4_sh_stats.reset_index().to_json(orient='split')]]


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
            "{}".format(maxminstreaks(1, overall_dff)[0]),
            "{}".format(maxminstreaks(1, overall_dff)[1]),
            "{}".format(maxminstreaks(1, filtered_dff)[0]),
            "{}".format(maxminstreaks(1, filtered_dff)[1]),
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
            "{}".format(maxminstreaks(2, overall_dff)[0]),
            "{}".format(maxminstreaks(2, overall_dff)[1]),
            "{}".format(maxminstreaks(2, filtered_dff)[0]),
            "{}".format(maxminstreaks(2, filtered_dff)[1]),
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
            "{}".format(maxminstreaks(3, overall_dff)[0]),
            "{}".format(maxminstreaks(3, overall_dff)[1]),
            "{}".format(maxminstreaks(3, filtered_dff)[0]),
            "{}".format(maxminstreaks(3, filtered_dff)[1]),
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
            "{}".format(streakf1f4(overall_dff)[1]),
            "{}".format(streakf1f4(overall_dff)[2]),
            "{}".format(streakf1f4(filtered_dff)[1]),
            "{}".format(streakf1f4(filtered_dff)[2]),
            'form_guide_ul_place', 0, 0, 0]


@app.callback(
    [Output('scoreboard2', 'children'),
     Output('scoreboard3', 'children'),
     Output('scoreboard4', 'children'),
     Output('scoreboard5', 'children')],
    [Input('table', 'active_cell')],
    [State('table', 'derived_viewport_data'),
     State('store_season_stats', 'children'),
     State('store_class_stats', 'children'),
     State('store_distance_stats', 'children'),
     State('store_rno_stats', 'children'),
     State('store_month_stats', 'children'),
     State('store_year_stats', 'children'),
     State('store_nor_stats', 'children'),
     State('store_hno_stats', 'children'),
     State('store_age_stats', 'children'),
     State('store_wt_stats', 'children'),
     State('store_sh_stats', 'children')]
)
def updatebyseason(active_cell, data, season_df, class_df, distance_df, rno_df, month_df, year_df, nor_df, hno_df, age_df, wt_df, sh_df):
    if active_cell is None:
        raise PreventUpdate
    print(active_cell)
    f1f4_season_stats = pd.read_json(season_df[1], orient='split').set_index(season_df[0])
    f1f4_distance_stats = pd.read_json(distance_df[1], orient='split').set_index(distance_df[0])
    f1f4_class_stats = pd.read_json(class_df[1], orient='split').set_index(class_df[0])
    f1f4_rno_stats = pd.read_json(rno_df[1], orient='split').set_index(rno_df[0])
    f1f4_month_stats = pd.read_json(month_df[1], orient='split').set_index(month_df[0])
    f1f4_year_stats = pd.read_json(year_df[1], orient='split').set_index(year_df[0])
    f1f4_nor_stats = pd.read_json(nor_df[1], orient='split').set_index(nor_df[0])
    f1f4_hno_stats = pd.read_json(hno_df[1], orient='split').set_index(hno_df[0])
    f1f4_age_stats = pd.read_json(age_df[1], orient='split').set_index(age_df[0])
    f1f4_wt_stats = pd.read_json(wt_df[1], orient='split').set_index(wt_df[0])
    f1f4_sh_stats = pd.read_json(sh_df[1], orient='split').set_index(sh_df[0])

    row = active_cell['row']
    column_id = active_cell['column_id']
    type = data[row]['F']
    # f1f4_stats = pd.read_json(season_df[0],orient='split')
    # print(season_df[0])
    # return []
    return [generate_favourite_season_scoreboard(f1f4_season_stats.loc[type], type),
            generate_favourite_generic_scoreboard(f1f4_class_stats.loc[type], type, 'table3', 'class'),
            generate_favourite_generic_scoreboard(f1f4_distance_stats.loc[type], type, 'table4', 'distance'),
            generate_favourite_generic_scoreboard(f1f4_rno_stats.loc[type], type, 'table5', 'Race No.'),
            generate_favourite_generic_scoreboard(f1f4_month_stats.loc[type], type, 'table6', 'Month'),
            generate_favourite_generic_scoreboard(f1f4_year_stats.loc[type], type, 'table7', 'Year'),
            generate_favourite_generic_scoreboard(f1f4_nor_stats.loc[type], type, 'table8', 'NoR'),
            generate_favourite_generic_scoreboard(f1f4_hno_stats.loc[type], type, 'table9', 'H. No'),
            generate_favourite_generic_scoreboard(f1f4_age_stats.loc[type], type, 'table10', 'Age'),
            generate_favourite_generic_scoreboard(f1f4_wt_stats.loc[type], type, 'table11', 'Wt'),
            generate_favourite_generic_scoreboard(f1f4_sh_stats.loc[type], type, 'table12', 'Sh')]


if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)