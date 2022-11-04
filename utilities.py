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

forecast = ['F12', 'F13', 'F14', 'F21', 'F23', 'F24', 'F31', 'F32', 'F34', 'F41', 'F42', 'F43']
trinella = ['F123', 'F124', 'F132', 'F134', 'F142', 'F143', 'F213', 'F214', 'F231', 'F234', 'F241', 'F243', 'F312',
            'F314', 'F321', 'F324', 'F341', 'F342', 'F412', 'F413', 'F421', 'F423', 'F431', 'F432']
exacta = ['F1234', 'F1243', 'F1324', 'F1342', 'F1423', 'F1432', 'F2134', 'F2143', 'F2314', 'F2341', 'F2413', 'F2431',
          'F3124', 'F3142', 'F3214', 'F3241', 'F3412', 'F3421', 'F4123', 'F4132', 'F4213', 'F4231', 'F4312', 'F4321']
quinella = ['F12|F21', 'F13|F31', 'F14|F41', 'F23|F32', 'F24|F42', 'F34|F43']
fc = ['F12']
qn = ['F12|F21']
qnp = ['F12|F21', 'F13|F31', 'F23|F32']
tanala = ['F123']
tanala_place = ['F123|F132']
trio = ['F123', 'F132', 'F213', 'F231', 'F312', 'F321']
exacta = ['F1234']
first_4 = ['F1234', 'F1243', 'F1324', 'F1342', 'F1423', 'F1432', 'F2134', 'F2143', 'F2314', 'F2341', 'F2413', 'F2431',
          'F3124', 'F3142', 'F3214', 'F3241', 'F3412', 'F3421', 'F4123', 'F4132', 'F4213', 'F4231', 'F4312', 'F4321']


def safe_division(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0


def generatebaroverall(dfff):
    # c = list(
    #     ["#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
    #      "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
    #      "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
    #      "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
    #      "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
    #      "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
    #      "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF"])
    c = list(
        ["#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
         "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
         "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA", "#D2B4DE", "#DFFF00",
         "#7FB3D5", "#58D68D", "#F5B041", "#E59866", "#E5E7E9", "#AEB6BF", "#F5B7B1", "#E6B0AA"])
    # F = list(forecast + quinella + trinella + exacta)
    F = list(set(fc + qn + qnp + tanala + tanala_place + trio + exacta + first_4))
    # print(len(F))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=F,
                         y=[100 * (safe_division(dfff[i].value_counts().get(1, 0), len(dfff))) for i in F],
                         marker_color=c[0:len(F)],
                         text=[round(100 * (safe_division(dfff[i].value_counts().get(1, 0), len(dfff))), 1) for i in F],
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


def gen_overall_centre_table_new(dff):
    a = [
        html.Div("Total Races: {}".format(len(dff)), className='overall_table_header_f1f4'),
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th("FC", id='FC_header', className='overallf1f4boxcolheader', colSpan=3,
                            style={'width': '12.5%'}),
                    html.Th("QN", id='QN_header', className='overallf1f4boxcolheader', colSpan=3,
                            style={'width': '12.5%'}),
                    html.Th("QNP", id='QNP_header', className='overallf1f4boxcolheader', colSpan=3,
                            style={'width': '12.5%'}),
                    html.Th("TANALA", id='TANALA_header', className='overallf1f4boxcolheader', colSpan=3,
                            style={'width': '12.5%'}),
                    html.Th("TANALA Place", id='TANALA_Place_header', className='overallf1f4boxcolheader', colSpan=3,
                            style={'width': '12.5%'}),
                    html.Th("TRIO", id='TRIO_header', className='overallf1f4boxcolheader', colSpan=3,
                            style={'width': '12.5%'}),
                    html.Th("EXACTA", id='EXACTA_header', className='overallf1f4boxcolheader', colSpan=3,
                            style={'width': '12.5%'}),
                    html.Th("FIRST 4", id='FIRST_4_header', className='overallf1f4boxcolheader', colSpan=3,
                            style={'width': '12.5%'})
                ])
            ]),
            html.Tr([
                html.Th("Combo", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W%", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("Combo", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W%", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("Combo", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W%", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("Combo", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W%", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("Combo", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W%", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("Combo", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W%", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("Combo", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W%", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("Combo", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W", id='OO_header', className='overallf1f4boxrowheader'),
                html.Th("W%", id='OO_header', className='overallf1f4boxrowheader')
            ]),
            html.Tr([
                html.Th("{}".format(fc[0]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[fc[0]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[fc[0]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(qn[0]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[qn[0]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[qn[0]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(qnp[0]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[qnp[0]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[qnp[0]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(tanala[0]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[tanala[0]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[tanala[0]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(tanala_place[0]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[tanala_place[0]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[tanala_place[0]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(trio[0]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[trio[0]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trio[0]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(exacta[0]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[exacta[0]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[exacta[0]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[0]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[0]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[0]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("",className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(qnp[1]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[qnp[1]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[qnp[1]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trio[1]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[trio[1]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trio[1]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[1]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[1]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[1]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(qnp[2]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[qnp[2]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[qnp[2]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trio[2]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[trio[2]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trio[2]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[2]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[2]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[2]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trio[3]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[trio[3]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trio[3]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[3]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[3]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[3]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trio[4]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[trio[4]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trio[4]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[4]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[4]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[4]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(trio[5]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[trio[5]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[trio[5]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[5]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[5]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[5]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[6]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[6]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[6]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[7]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[7]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[7]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[8]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[8]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[8]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[9]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[9]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[9]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[10]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[10]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[10]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[11]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[11]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[11]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[12]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[12]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[12]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[13]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[13]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[13]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[14]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[14]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[14]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[15]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[15]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[15]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[16]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[16]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[16]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[17]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[17]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[17]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[18]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[18]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[18]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[19]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[19]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[19]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[20]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[20]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[20]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[21]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[21]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[21]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[22]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[22]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[22]].value_counts(normalize=True).get(1, 0) * 100, 2)),
                        className='overallf1f4boxcell')
            ]),
            html.Tr([
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Td("", className='overallf1f4boxcell'),
                html.Th("{}".format(first_4[23]), className='overallf1f4boxcell', style={'color':'yellow'}),
                html.Td("{}".format(dff[first_4[23]].value_counts().get(1, 0)), className='overallf1f4boxcell'),
                html.Td("{}".format(round(dff[first_4[23]].value_counts(normalize=True).get(1, 0) * 100, 2)),
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
        avgWstreak = safe_division(sumWstreak, numWstreak)
    if numLstreak:
        avgLstreak = safe_division(sumLstreak, numLstreak)

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
        avgPS = safe_division(sumPS, numPS)
    if numNPS > 0:
        avgNPS = safe_division(sumNPS, numNPS)
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


def generate_overall_scoreboard(overall_dff, dff, allf1f4, f1f4):
    Overall_SB = [html.Hr(className='othr'),
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
                              html.Th("LLS(WIN)", id='overall_longest_losing_streak_header'),
                              html.Th("Edge-W", id='overall_TR_header'),
                              html.Th("Edge-2", id='overall_TR_header'),
                              html.Th("Edge-3", id='overall_TR_header'),
                              html.Th("Edge-P", id='overall_TR_header')
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
                              (safe_division(len(overall_dff.loc[overall_dff['RESULT'] == 1]), len(overall_dff))) * 100),
                                  id='overall_first_pct'),
                          html.Td(html.Button("{}".format(len(overall_dff.loc[overall_dff['RESULT'] == 2])),
                                              id='overall_second_button',
                                              n_clicks=0),
                                  id='overall_second_pos'),
                          html.Td("{0:.2f}".format(
                              (safe_division(len(overall_dff.loc[overall_dff['RESULT'] == 2]), len(overall_dff))) * 100),
                                  id='overall_second_pct'),
                          html.Td(html.Button("{}".format(len(overall_dff.loc[overall_dff['RESULT'] == 3])),
                                              id='overall_third_button',
                                              n_clicks=0),
                                  id='overall_third_pos'),
                          html.Td("{0:.2f}".format(
                              (safe_division(len(overall_dff.loc[overall_dff['RESULT'] == 3]), len(overall_dff))) * 100),
                                  id='overall_third_pct'),
                          html.Td(html.Button("{}".format((len(overall_dff.loc[overall_dff['RESULT'] == 1]) + len(
                              overall_dff.loc[overall_dff['RESULT'] == 2]) + len(
                              overall_dff.loc[overall_dff['RESULT'] == 3]))),
                                              id='overall_place_button', n_clicks=0),
                                  id='overall_place_pos'),
                          html.Td("{0:.2f}".format(
                              (safe_division((len(overall_dff.loc[overall_dff['RESULT'] == 1]) + len(
                                  overall_dff.loc[overall_dff['RESULT'] == 2]) + len(
                                  overall_dff.loc[overall_dff['RESULT'] == 3])), len(overall_dff))) * 100),
                              id='overall_place_pct'),
                          html.Td(html.Button("{}".format(len(overall_dff) - (
                                  len(overall_dff.loc[overall_dff['RESULT'] == 1]) + len(
                              overall_dff.loc[overall_dff['RESULT'] == 2]) + len(
                              overall_dff.loc[overall_dff['RESULT'] == 3]))), id='overall_loss_button', n_clicks=0),
                                  id='loss'),
                          html.Td("{}".format(maxminstreaks(1, overall_dff)[0]),
                                  id='overall_longest_winning_streak_data'),
                          html.Td("{}".format(maxminstreaks(1, overall_dff)[1]),
                                  id='overall_longest_losing_streak_data'),
                          html.Td("{}".format(allf1f4['edgeW'])),
                          html.Td("{}".format(allf1f4['edgeSHP'])),
                          html.Td("{}".format(allf1f4['edgeTHP'])),
                          html.Td("{}".format(allf1f4['edgePlc']))
                      ]),
                      html.Tr([
                          html.Td("Filtered", className='type_header'),
                          html.Td("{}".format(len(dff))),
                          html.Td("{}%".format(round(((safe_division(len(dff), len(overall_dff))) * 100), 2))),
                          html.Td(
                              html.Button("{}".format(len(dff.loc[dff['RESULT'] == 1])), id='first_button', n_clicks=0),
                              id='first_pos'),
                          html.Td("{0:.2f}".format((safe_division(len(dff.loc[dff['RESULT'] == 1]), len(dff))) * 100),
                                  id='first_pct'),
                          html.Td(html.Button("{}".format(len(dff.loc[dff['RESULT'] == 2])), id='second_button',
                                              n_clicks=0),
                                  id='second_pos'),
                          html.Td("{0:.2f}".format((safe_division(len(dff.loc[dff['RESULT'] == 1]), len(dff))) * 100),
                                  id='second_pct'),
                          html.Td(
                              html.Button("{}".format(len(dff.loc[dff['RESULT'] == 3])), id='third_button', n_clicks=0),
                              id='third_pos'),
                          html.Td("{0:.2f}".format((safe_division(len(dff.loc[dff['RESULT'] == 3]), len(dff))) * 100),
                                  id='third_pct'),
                          html.Td(html.Button("{}".format((len(dff.loc[dff['RESULT'] == 1]) + len(
                              dff.loc[dff['RESULT'] == 2]) + len(dff.loc[dff['RESULT'] == 3]))), id='place_button',
                                              n_clicks=0),
                                  id='place_pos'),
                          html.Td("{0:.2f}".format(
                              (safe_division((len(dff.loc[dff['RESULT'] == 1]) + len(dff.loc[dff['RESULT'] == 2]) + len(
                                  dff.loc[dff['RESULT'] == 3])), len(dff))) * 100), id='place_pct'),
                          html.Td(html.Button("{}".format(len(dff) - (
                                  len(dff.loc[dff['RESULT'] == 1]) + len(dff.loc[dff['RESULT'] == 2]) + len(
                              dff.loc[dff['RESULT'] == 3]))), id='bo_button', n_clicks=0),
                                  id='loss'),
                          html.Td("{}".format(maxminstreaks(1, dff)[0]), id='longest_winning_streak_data'),
                          html.Td("{}".format(maxminstreaks(1, dff)[1]), id='longest_losing_streak_data'),
                          html.Td("{}".format(f1f4['edgeW'])),
                          html.Td("{}".format(f1f4['edgeSHP'])),
                          html.Td("{}".format(f1f4['edgeTHP'])),
                          html.Td("{}".format(f1f4['edgePlc']))
                      ])
                  ], id='overall_sb_table')]
    return Overall_SB


def generate_overall_scoreboard_for_combo(overall_df, df, combination):
    Overall_SB = [html.Hr(className='othr'),
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
                              (safe_division(len(overall_df.loc[overall_df[combination] == 1]), len(overall_df))) * 100),
                                  id='overall_first_pct'),
                          html.Td(html.Button("{}".format(len(overall_df.loc[overall_df[combination] == 0])),
                                              id='overall_loss_button',
                                              n_clicks=0),
                                  id='overall_los_pos'),
                          html.Td("{0:.2f}".format(
                              (safe_division(len(overall_df.loc[overall_df[combination] == 0]), len(overall_df))) * 100),
                                  id='overall_loss_pct'),
                          html.Td("{}".format(streak(overall_df[combination])[1]),
                                  id='overall_longest_winning_streak_data'),
                          html.Td("{}".format(streak(overall_df[combination])[2]),
                                  id='overall_longest_losing_streak_data')
                      ]),
                      html.Tr([
                          html.Td("Filtered", className='type_header'),
                          html.Td("{}".format(len(df))),
                          html.Td("{}%".format(round(((safe_division(len(df), len(overall_df))) * 100), 2))),
                          html.Td(
                              html.Button("{}".format(len(df.loc[df[combination] == 1])), id='win_button', n_clicks=0),
                              id='first_pos'),
                          html.Td("{0:.2f}".format((safe_division(len(df.loc[df[combination] == 1]), len(df))) * 100), id='win_pct'),
                          html.Td(
                              html.Button("{}".format(len(df.loc[df[combination] == 0])), id='loss_button', n_clicks=0),
                              id='loss_pos'),
                          html.Td("{0:.2f}".format((safe_division(len(df.loc[df[combination] == 0]), len(df))) * 100), id='loss_pct'),
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
    fodf["W%"] = ((safe_division(fodf["W"], fodf['TR'])) * 100).round(2)
    fodf["SHP%"] = ((safe_division(fodf["SHP"], fodf['TR'])) * 100).round(2)
    fodf["THP%"] = ((safe_division(fodf["THP"], fodf['TR'])) * 100).round(2)
    fodf["Plc%"] = ((safe_division((fodf["W"] + fodf["SHP"] + fodf["THP"]), fodf['TR'])) * 100).round(2)
    fodf["Plc"] = fodf["W"] + fodf["SHP"] + fodf["THP"]
    fodf["BO"] = fodf["TR"] - fodf["Plc"]
    # fodf["TR%"] = ((fodf["TR"] / len(df)) * 100 *4).round(2)
    f = [i for i in foo if i in fodf.index]
    fodf = fodf.reindex(f)
    fodf = fodf[['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']]
    return fodf


def generate_f1f4_stats_new(dff):
    foo = ['F1', 'F2', 'F3', 'F4']
    pl = [1, 2, 3, 4, 0]
    n = dff.groupby(["F", "RESULT"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = dff.groupby(["F"], as_index=True).size().to_frame()
    k = dff.groupby(["F"], as_index=True)['RESULT'].agg([maxwinstreaks, maxlosestreaks])
    j = dff.groupby(["F"], as_index=True)['W'].sum()
    i = dff.groupby(["F"], as_index=True)['SHP'].sum()
    h = dff.groupby(["F"], as_index=True)['THP'].sum()
    g = dff.groupby(["F"], as_index=True)['Plc'].sum()
    k.rename(columns={'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'}, inplace=True)
    fodf = pd.concat([n, m, k, j, i, h, g], axis=1)
    fodf.columns = [1, 2, 3, 4, 0, 'TR', 'LWS', 'LLS', 'Wodds', 'SHPodds', 'THPodds', 'Plcodds']
    fodf.rename(columns={1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'}, inplace=True)
    fodf["W%"] = ((safe_division(fodf["W"], fodf['TR'])) * 100).round(2)
    fodf["SHP%"] = ((safe_division(fodf["SHP"], fodf['TR'])) * 100).round(2)
    fodf["THP%"] = ((safe_division(fodf["THP"], fodf['TR'])) * 100).round(2)
    fodf["Plc%"] = ((safe_division((fodf["W"] + fodf["SHP"] + fodf["THP"]), fodf['TR'])) * 100).round(2)
    fodf["Plc"] = fodf["W"] + fodf["SHP"] + fodf["THP"]
    fodf["BO"] = fodf["TR"] - fodf["Plc"]
#     for i in fodf.index:
#         fodf.loc[i, "countW"] = len(newdf_f.loc[(newdf_f['F'] == i) & (newdf_f['RESULT'] == 1) & (newdf_f['W'] != 0.0)])
#         fodf.loc[i, "count2"] = len(newdf_f.loc[(newdf_f['F'] == i) & (newdf_f['RESULT'] == 2) & (newdf_f['SHP'] != 0.0)])
#         fodf.loc[i, "count3"] = len(newdf_f.loc[(newdf_f['F'] == i) & (newdf_f['RESULT'] == 3) & (newdf_f['THP'] != 0.0)])
#         fodf.loc[i, "countPlc"] = len(newdf_f.loc[(newdf_f['F'] == i) & (newdf_f['RESULT'].isin([1,2,3])) & (newdf_f['Plc'] != 0.0)])
    # fodf["TR%"] = ((fodf["TR"] / len(df)) * 100 *4).round(2)
    fodf['Wavg'] = safe_division(fodf['Wodds'], fodf['W'])
    fodf["edgeW"] = round(fodf['W%']/100 * fodf['Wavg'] - (100 - fodf['W%'])/100, 2)
    fodf['SHPavg'] = safe_division(fodf['SHPodds'], fodf['SHP'])
    fodf["edgeSHP"] = round(fodf['SHP%']/100 * fodf['SHPavg'] - (100 - fodf['SHP%'])/100, 2)
    fodf['THPavg'] = safe_division(fodf['THPodds'], fodf['THP'])
    fodf["edgeTHP"] = round(fodf['THP%']/100 * fodf['THPavg'] - (100 - fodf['THP%'])/100, 2)
    fodf['Plcavg'] = safe_division(fodf['Plcodds'], fodf['Plc'])
    fodf["edgePlc"] = round(fodf['Plc%']/100 * fodf['Plcavg'] - (100 - fodf['Plc%'])/100, 2)
    f = [i for i in foo if i in fodf.index]
    fodf = fodf.reindex(f)
    fodf = fodf[['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS',
                 'edgeW', 'edgeSHP', 'edgeTHP', 'edgePlc']]
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
    fseasondf["W%"] = ((safe_division(fseasondf["W"], fseasondf['TR'])) * 100).round(2)
    fseasondf["SHP%"] = ((safe_division(fseasondf["SHP"], fseasondf['TR'])) * 100).round(2)
    fseasondf["THP%"] = ((safe_division(fseasondf["THP"], fseasondf['TR'])) * 100).round(2)
    fseasondf["Plc%"] = ((safe_division((fseasondf["W"] + fseasondf["SHP"] + fseasondf["THP"]), fseasondf['TR'])) * 100).round(2)
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
    fdistancedf["W%"] = ((safe_division(fdistancedf["W"], fdistancedf['TR'])) * 100).round(2)
    fdistancedf["SHP%"] = ((safe_division(fdistancedf["SHP"], fdistancedf['TR'])) * 100).round(2)
    fdistancedf["THP%"] = ((safe_division(fdistancedf["THP"], fdistancedf['TR'])) * 100).round(2)
    fdistancedf["Plc%"] = (
            (safe_division((fdistancedf["W"] + fdistancedf["SHP"] + fdistancedf["THP"]), fdistancedf['TR'])) * 100).round(2)
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
    fclassdf["W%"] = ((safe_division(fclassdf["W"], fclassdf['TR'])) * 100).round(2)
    fclassdf["SHP%"] = ((safe_division(fclassdf["SHP"], fclassdf['TR'])) * 100).round(2)
    fclassdf["THP%"] = ((safe_division(fclassdf["THP"], fclassdf['TR'])) * 100).round(2)
    fclassdf["Plc%"] = ((safe_division((fclassdf["W"] + fclassdf["SHP"] + fclassdf["THP"]), fclassdf['TR'])) * 100).round(2)
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
    fclassdf["W%"] = ((safe_division(fclassdf["W"], fclassdf['TR'])) * 100).round(2)
    fclassdf["SHP%"] = ((safe_division(fclassdf["SHP"], fclassdf['TR'])) * 100).round(2)
    fclassdf["THP%"] = ((safe_division(fclassdf["THP"], fclassdf['TR'])) * 100).round(2)
    fclassdf["Plc%"] = ((safe_division((fclassdf["W"] + fclassdf["SHP"] + fclassdf["THP"]), fclassdf['TR'])) * 100).round(2)
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
    fdistancedf["W%"] = ((safe_division(fdistancedf["W"], fdistancedf['TR'])) * 100).round(2)
    fdistancedf["SHP%"] = ((safe_division(fdistancedf["SHP"], fdistancedf['TR'])) * 100).round(2)
    fdistancedf["THP%"] = ((safe_division(fdistancedf["THP"], fdistancedf['TR'])) * 100).round(2)
    fdistancedf["Plc%"] = (
            (safe_division((fdistancedf["W"] + fdistancedf["SHP"] + fdistancedf["THP"]), fdistancedf['TR'])) * 100).round(2)
    fdistancedf["Plc"] = fdistancedf["W"] + fdistancedf["SHP"] + fdistancedf["THP"]
    fdistancedf["BO"] = fdistancedf["TR"] - fdistancedf["Plc"]
    fdistancedf = fdistancedf[['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']]
    return fdistancedf


def generate_f1f4_generic_stats_new(dff, column):
    foo = ['F1', 'F2', 'F3', 'F4']
    pl = [1, 2, 3, 4, 0]
    n = dff.groupby(["F", column, "RESULT"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = dff.groupby(["F", column], as_index=True).size().to_frame()
    k = dff.groupby(["F", column], as_index=True)['RESULT'].agg([maxwinstreaks, maxlosestreaks])
    j = dff.groupby(["F", column], as_index=True)['W'].sum()
    i = dff.groupby(["F", column], as_index=True)['SHP'].sum()
    h = dff.groupby(["F", column], as_index=True)['THP'].sum()
    g = dff.groupby(["F", column], as_index=True)['Plc'].sum()
    k.rename(columns={'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'}, inplace=True)
    fdistancedf = pd.concat([n, m, k, j, i, h, g], axis=1)

    fdistancedf.columns = [1, 2, 3, 4, 0, 'TR', 'LWS', 'LLS', 'Wodds', 'SHPodds', 'THPodds', 'Plcodds']
    fdistancedf.rename(columns={1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'},
                       inplace=True)
    fdistancedf["W%"] = ((safe_division(fdistancedf["W"], fdistancedf['TR'])) * 100).round(2)
    fdistancedf["SHP%"] = ((safe_division(fdistancedf["SHP"], fdistancedf['TR'])) * 100).round(2)
    fdistancedf["THP%"] = ((safe_division(fdistancedf["THP"], fdistancedf['TR'])) * 100).round(2)
    fdistancedf["Plc%"] = (
            (safe_division(fdistancedf["THP"], fdistancedf['TR'])) * 100).round(2)
    fdistancedf["Plc"] = fdistancedf["W"] + fdistancedf["SHP"] + fdistancedf["THP"]
    fdistancedf["BO"] = fdistancedf["TR"] - fdistancedf["Plc"]
#     for i in fdistancedf.index:
#         fdistancedf.loc[i, "countW"] = len(newdf_f.loc[(newdf_f['F'] == i) & (newdf_f['RESULT'] == 1) & (newdf_f['W'] != 0.0)])
#         fdistancedf.loc[i, "count2"] = len(newdf_f.loc[(newdf_f['F'] == i) & (newdf_f['RESULT'] == 2) & (newdf_f['SHP'] != 0.0)])
#         fdistancedf.loc[i, "count3"] = len(newdf_f.loc[(newdf_f['F'] == i) & (newdf_f['RESULT'] == 3) & (newdf_f['THP'] != 0.0)])
#         fdistancedf.loc[i, "countPlc"] = len(newdf_f.loc[(newdf_f['F'] == i) & (newdf_f['RESULT'].isin([1,2,3])) & (newdf_f['Plc'] != 0.0)])
#     fdistancedf["TR%"] = ((fdistancedf["TR"] / len(df)) * 100 *4).round(2)
    fdistancedf['Wavg'] = safe_division(fdistancedf['Wodds'], fdistancedf['W'])
    fdistancedf["edgeW"] = round(fdistancedf['W%']/100 * fdistancedf['Wavg'] - (100 - fdistancedf['W%'])/100, 2)
    fdistancedf['SHPavg'] = safe_division(fdistancedf['SHPodds'], fdistancedf['SHP'])
    fdistancedf["edgeSHP"] = round(fdistancedf['SHP%']/100 * fdistancedf['SHPavg'] - (100 - fdistancedf['SHP%'])/100, 2)
    fdistancedf['THPavg'] = safe_division(fdistancedf['THPodds'], fdistancedf['THP'])
    fdistancedf["edgeTHP"] = round(fdistancedf['THP%']/100 * fdistancedf['THPavg'] - (100 - fdistancedf['THP%'])/100, 2)
    fdistancedf['Plcavg'] = safe_division(fdistancedf['Plcodds'], fdistancedf['Plc'])
    fdistancedf["edgePlc"] = round(fdistancedf['Plc%']/100 * fdistancedf['Plcavg'] - (100 - fdistancedf['Plc%'])/100, 2)
    fdistancedf = fdistancedf[['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS',
                 'edgeW', 'edgeSHP', 'edgeTHP', 'edgePlc']]
    return fdistancedf



def generate_favourite_scoreboard(xdf, filt):
    header = html.H3("F1-F4 stats - {}".format(filt),
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    fav_sb = dash_table.DataTable(
        id='table_{}'.format(filt),
        columns=[{"name": i, "id": i, "hideable":True} for i in xdf[
            ['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS', 'edgeW', 'edgeSHP', 'edgeTHP', 'edgePlc']].reset_index().columns],
        data=xdf[
            ['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS', 'edgeW', 'edgeSHP', 'edgeTHP', 'edgePlc']].reset_index().to_dict(
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


def generate_favourite_season_scoreboard(xdf, type, filt):
    header = html.H3("{} stats by season - {}".format(type, filt),
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    fav_sb = dash_table.DataTable(
        id='table2_{}'.format(filt),
        columns=[{"name": i, "id": i, "hideable":True} for i in xdf[
            ['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS', 'edgeW', 'edgeSHP', 'edgeTHP', 'edgePlc']].reset_index().columns],
        data=xdf[
            ['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS', 'edgeW', 'edgeSHP', 'edgeTHP', 'edgePlc']].reset_index().to_dict(
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


def generate_favourite_generic_scoreboard(xdf, type, idd, what, filt):
    header = html.H3("{} stats by {} - {}".format(type, what, filt),
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    fav_sb = dash_table.DataTable(
        id=idd,
        columns=[{"name": i, "id": i, "hideable":True} for i in xdf[
            ['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS', 'edgeW', 'edgeSHP', 'edgeTHP', 'edgePlc']].reset_index().columns],
        data=xdf[
            ['TR', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS', 'edgeW', 'edgeSHP', 'edgeTHP', 'edgePlc']].reset_index().to_dict(
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