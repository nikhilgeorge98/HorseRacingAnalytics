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
from pandas.api.types import is_string_dtype, is_numeric_dtype
import functools
import json
import itertools
from collections import defaultdict
import math

all_df = pd.concat(pd.read_excel(r'data\ALL CENTRES-DATA 210122.xlsx', sheet_name=None), ignore_index=True)
all_df.drop(columns=['Season Race Day', 'CUP', 'NoH', 'Pedigree', 'Desc', 'Al',  'Sh', 'Won By',
                 'Dist Win', 'Rtg', 'Odds', 'Time', 'BLR Tote Win', 'BLR Tote Win-Plc', 'BLR Tote SHP',
                 'BLR Tote SHP-Plc', 'BLR Tote THP', 'BLR Tote THP-Plc', 'HYD Tote Win', 'HYD Tote Win-Plc',
                 'HYD Tote SHP', 'HYD Tote SHP-Plc', 'HYD Tote THP', 'HYD Tote THP-Plc', 'KOL Tote Win',
                 'KOL  Tote Win-Plc', 'KOL  Tote SHP', 'KOL  Tote SHP-Plc', 'KOL Tote THP', 'KOL Tote THP-Plc',
                 'MYS Tote Win', 'MYS Tote Win-Plc', 'MYS Tote SHP', 'MYS Tote SHP-Plc', 'MYS Tote THP',
                 'MYS Tote THP-Plc', 'MUM Tote Win', 'MUM Tote Win-Plc', 'MUM Tote SHP', 'MUM Tote SHP-Plc',
                 'MUM Tote THP', 'MUM Tote THP-Plc', 'PUN Tote Win', 'PUN Tote Win-Plc', 'PUN Tote SHP',
                 'PUN Tote SHP-Plc', 'PUN Tote THP', 'PUN Tote THP-Plc', 'C Tote Win', 'C Tote Win-Plc',
                 'C Tote SHP', 'C Tote SHP-Plc', 'C Tote THP', 'C Tote THP-Plc', 'Unnamed: 96', 'Unnamed: 97',
                 'Unnamed: 98', 'Unnamed: 99', 'Unnamed: 100'], inplace=True)
all_df['Age'] = [x.strip('y') for x in all_df['Age']]
all_df['Age'] = pd.to_numeric(all_df['Age'])
all_df['Age'].unique()
all_df.rename(columns={'CLASSIFICATION': 'Classification', 'DISTANCE': 'Distance', 'Opening Odds': 'OPEN', 'Latest Odds': 'LATEST',
                     'Middle Odds': 'MIDDLE', 'Final Odds': 'FINAL', 'OOF': 'OO', 'LOF': 'LO', 'MOF': 'MO', 'FOF': 'FO', 'Result': 'Pl'
                    }, inplace = True)
all_df['Horse']=[i.strip() for i in all_df['Horse']]
all_df.Distance.replace(1500, 1600, inplace=True)
all_df.Season.replace(190222, 'BW1819', inplace=True)
all_df.Dr.replace([np.nan,'-','3?', '2?', '6?', '1?', '4?', '5?', '10?', '8?', '9?',
       '7?', '14?', '12?', '11?', '13?', '5 h'], [0,0,3,2,6,1,4,5,10,8,9,7,14,12,11,13,5], inplace=True)
all_df.replace([' F1', 'F1 ', ' F2', 'F2 ', ' F3', 'F3 ', ' F4', 'F4 ', ' F5', 'F5 ', ' F6', 'F6 ', ' F7', 'F7 ', ' F8', 'F8 ', ' F9', 'F9 ', ' F10', 'F10 ', ' F11', 'F11 ', ' F12', 'F12 ', ' F13', 'F13 ', ' F14', 'F14 ', ' F15', 'F15 '], ['F1', 'F1', 'F2', 'F2', 'F3', 'F3', 'F4', 'F4', 'F5', 'F5', 'F6', 'F6', 'F7', 'F7', 'F8', 'F8', 'F9', 'F9', 'F10', 'F10', 'F11', 'F11', 'F12', 'F12', 'F13', 'F13', 'F14', 'F14', 'F15', 'F15'], inplace=True)
all_df.Pl.replace(['w', 'W', 'NC'], [0, 0, 0], inplace=True)
all_df.OPEN.replace(['NO',' '],[np.nan, np.nan],inplace=True)
all_df.OO.replace(['NO',np.nan],['F0', 'F0'],inplace=True)
all_df.LATEST.replace(['NO',' ','N0'],[np.nan, np.nan, np.nan],inplace=True)
all_df.LO.replace(['NO',np.nan],['F0', 'F0'],inplace=True)
all_df.MIDDLE.replace(['NO',' ','N0'],[np.nan, np.nan, np.nan],inplace=True)
all_df.MO.replace(['NO',np.nan,'F66'],['F0','F0', 'F0'],inplace=True)
all_df.FINAL.replace(['NO',' ','N0'],[np.nan, np.nan, np.nan],inplace=True)
all_df.FO.replace(['NO',np.nan,'F100'],['F0','F0', 'F0'],inplace=True)
all_df['TurfBee'].replace([' ','.','9-','f',1,' F',2,3,' T'], [np.nan,np.nan,np.nan,'F','F','F','S','T','T'], inplace=True)
all_df['BOL'].replace([' ','.','9-','f',1,' F',2,3,' T'], [np.nan,np.nan,np.nan,'F','F','F','S','T','T'], inplace=True)
all_df['Dina Thanthi'].replace([0,' ','.','9-','f',1,' F',2,3,' T'], [np.nan,np.nan,np.nan,np.nan,'F','F','F','S','T','T'], inplace=True)
all_df['Experts'].replace([0,' ','.','9-','f',1,' F',2,3,' T'], [np.nan,np.nan,np.nan,np.nan,'F','F','F','S','T','T'], inplace=True)
all_df['Media Man'].replace(['A','   ','  ',0,' ','.','9-','f',1,' F',2,3,' T'], [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'F','F','F','S','T','T'], inplace=True)
all_df['S. Today'].replace([4,'A','   ','  ',0,' ','.','9-','f',1,' F',2,' S',3,' T'], [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'F','F','F','S','S','T','T'], inplace=True)
all_df['TRS'].replace([4,'A','   ','  ',0,' ','.','9-','f',1,' F',2,' S',3,' T'], [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'F','F','F','S','S','T','T'], inplace=True)
all_df['Telegraph'].replace([5,4,'A','   ','  ',0,' ','.','9-','f',1,' F',2,' S',3,' T'], [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'F','F','F','S','S','T','T'], inplace=True)
all_df['BNG'].replace([5,4,'A','   ','  ',0,' ','.','9-','f',1,' F',2,' S',3,' T'], [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'F','F','F','S','S','T','T'], inplace=True)
all_df['Telangana Today'].replace([5,4,'A','   ','  ',0,' ','.','9-','f',1,' F',2,' S',3,' T'], [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'F','F','F','S','S','T','T'], inplace=True)
all_df['Telangana Times'].replace([6, 7, 8, 9, 10, 11, 12,5,4,'A','   ','  ',0,' ','.','9-','f',1,' F',2,' S',3,' T'], [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'F','F','F','S','S','T','T'], inplace=True)
all_df['Andhra Jyothi'].replace([6, 7, 8, 9, 10, 11, 12,5,4,'A','   ','  ',0,' ','.','9-','f',1,' F',2,' S',3,' T'], [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'F','F','F','S','S','T','T'], inplace=True)
all_df['Asian Age'].replace([6, 7, 8, 9, 10, 11, 12,5,4,'A','   ','  ',0,' ','.','9-','f',1,' F',2,' S',3,' T'], [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'F','F','F','S','S','T','T'], inplace=True)
all_df['Times of India'].replace([6, 7, 8, 9, 10, 11, 12,5,4,'A','   ','  ',0,' ','.','9-','f',1,' F',2,' S',3,' T'], [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'F','F','F','S','S','T','T'], inplace=True)
all_df['The Hindu'].replace([6, 7, 8, 9, 10, 11, 12,5,4,'A','   ','  ',0,' ','.','9-','f',1,' F',2,' S',3,' T'], [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'F','F','F','S','S','T','T'], inplace=True)
all_df['Mumbai Mirror'].replace([6, 7, 8, 9, 10, 11, 12,5,4,'A','   ','  ',0,' ','.','9-','f',1,' F',2,' S',3,' T'], [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'F','F','F','S','S','T','T'], inplace=True)
all_df['Deccan Herald'].replace([6, 7, 8, 9, 10, 11, 12,5,4,'A','   ','  ',0,' ','.','9-','f',1,' F',2,' S',3,' T'], [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'F','F','F','S','S','T','T'], inplace=True)
all_df['newPl']= all_df['Pl']==1
all_df['newPl'].replace([False, True],[0,1], inplace=True)
all_df['Win_OPEN'] = all_df['newPl']*all_df['OPEN']
all_df['Win_LATEST'] = all_df['newPl']*all_df['LATEST']
all_df['Win_MIDDLE'] = all_df['newPl']*all_df['MIDDLE']
all_df['Win_FINAL'] = all_df['newPl']*all_df['FINAL']
# all_df['Win_OPEN'].replace(['', np.nan],[0,0], inplace=True)
# all_df['Win_LATEST'].replace(['', np.nan],[0,0], inplace=True)
# all_df['Win_MIDDLE'].replace(['', np.nan],[0,0], inplace=True)


media_list = ['TurfBee', 'BOL', 'Dina Thanthi', 'Experts', 'Media Man',
       'S. Today', 'TRS', 'Telegraph', 'BNG', 'Telangana Today',
       'Telangana Times', 'Andhra Jyothi', 'Asian Age', 'Times of India',
       'The Hindu', 'Mumbai Mirror', 'Deccan Herald']
# dicts = defaultdict(dict)
# for m in media_list:
#     for i in ['F', 'S', 'T']:
#         temp = all_df[all_df[m]==i]
#         if len(temp) == 0:
#             # print(m)
#             continue
#         dicts[m][i] = round(100*len(temp.loc[temp["Pl"]==1])/len(temp), 2)
#
# media_df = pd.DataFrame.from_dict(dicts, 'index')



app =dash.Dash(__name__)


def generatejockeystack(test, searchtype):
    if searchtype=='Jockey':
        st='Trainer'
    else:
        st='Jockey'
    c = list(
        ['#2ECC71', '#e8e40e', '#3498DB', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C',
         '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C'])
    j = test[st].unique()
    fig = go.Figure(data=[
        go.Bar(name='1', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 1]) for i in j],
               marker_color=c[0],
               text='1st', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='2', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 2]) for i in j],
               marker_color=c[1],
               text='2nd', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='3', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 3]) for i in j],
               marker_color=c[2],
               text='3rd', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='4', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 4]) for i in j],
               marker_color=c[3],
               text='4th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='5', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 5]) for i in j],
               marker_color=c[4],
               text='5th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='6', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 6]) for i in j],
               marker_color=c[5],
               text='6th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='7', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 7]) for i in j],
               marker_color=c[6],
               text='7th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='8', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 8]) for i in j],
               marker_color=c[7],
               text='8th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='9', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 9]) for i in j],
               marker_color=c[8],
               text='9th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='10', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 10]) for i in j],
               marker_color=c[9],
               text='10th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='11', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 11]) for i in j],
               marker_color=c[10],
               text='11th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='12', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 12]) for i in j],
               marker_color=c[11],
               text='12th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='13', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 13]) for i in j],
               marker_color=c[12],
               text='13th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='14', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 14]) for i in j],
               marker_color=c[13],
               text='14th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='15', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 15]) for i in j],
               marker_color=c[14],
               text='15th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='16', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 16]) for i in j],
               marker_color=c[15],
               text='16th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='17', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 17]) for i in j],
               marker_color=c[16],
               text='17th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='18', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 18]) for i in j],
               marker_color=c[17],
               text='18th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='19', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 19]) for i in j],
               marker_color=c[18],
               text='19th', textposition='inside',
               hovertemplate ='%{y} time(s)'),
        go.Bar(name='20', x=j,
               y=[len(test.loc[test[st] == i].loc[test.loc[test[st] == i]['Pl'] == 20]) for i in j],
               marker_color=c[19],
               text='20th', textposition='inside',
               hovertemplate ='%{y} time(s)')
    ])
    fig.update_xaxes(title_text="{}".format(st))
    fig.update_yaxes(title_text="Total races divided into placements")
    fig.update_layout(title_text='Results with each {}'.format(st),
                      height=700,
                      barmode='stack',
                      plot_bgcolor="#061e44",
                      paper_bgcolor="#082255",
                      font_color="white",
                      showlegend=True,
                      autosize=True,
                      xaxis={'showgrid': True, 'zeroline': True, 'categoryorder': 'total descending',
                             'gridcolor': '#082255', 'zerolinecolor': '#082255'},
                      yaxis={'showgrid': True, 'zeroline': True, 'gridcolor': '#082255', 'zerolinecolor': '#082255', 'tickformat':',d'},
                      )
    return fig


def generatescatter(dfff, searchtype):
    c = list(['#2ECC71', '#e8e40e', '#3498DB', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C',
         '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C'])
    fig = go.Figure()
    if len(dfff[searchtype].unique())>1:
        n=50
    else:
        n=100
    for i in dfff[searchtype].unique():
        fig.add_trace(go.Scatter(x=np.arange(1, n+1, 1),
                                         y=dfff.loc[dfff[searchtype]==i]['Pl'].tail(n),
                                         mode='lines+markers+text',
                                         marker_color=[c[j-1] for j in dfff.loc[dfff[searchtype]==i]['Pl'].tail(n)],
                                         marker_size=5,
                                         text=dfff.loc[dfff[searchtype]==i]['Pl'].tail(n),
                                         textposition='top center',
                                         hoverinfo='none',
                                         name=i
                                 ))
    fig.update_xaxes(title_text="Races")
    fig.update_yaxes(title_text="Position", range=[20, 0],
                     showticklabels=False)
    fig.update_layout(title_text='Recent results',
                      showlegend=True,
                      autosize=True,
                      xaxis={'showgrid': True, 'zeroline': True, 'categoryorder': 'total ascending', 'gridcolor': '#082255', 'zerolinecolor': '#082255'},
                      yaxis={'showgrid': True, 'zeroline': True, 'gridcolor': '#082255', 'zerolinecolor': '#082255'},
                      plot_bgcolor="#061e44",
                      paper_bgcolor="#082255",
                      font_color="white"
                      )
    return fig


def generatebar(dfff, searchtype):
    c = list(['#2ECC71', '#e8e40e', '#3498DB', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C',
         '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C'])
    fig = go.Figure()
    for i in dfff[searchtype].unique():
        fig.add_trace(go.Bar(x=np.arange(1, dfff.loc[dfff[searchtype]==i]['Pl'].max() + 1, 1),
                                 y=[len(dfff.loc[dfff[searchtype]==i].loc[dfff.loc[dfff[searchtype]==i]['Pl'] == n]) for n in range(1, dfff.loc[dfff[searchtype]==i]['Pl'].max() + 1)],
                                 marker_color = c,
                                 text=[len(dfff.loc[dfff[searchtype]==i].loc[dfff.loc[dfff[searchtype]==i]['Pl'] == n]) for n in range(1, dfff.loc[dfff[searchtype]==i]['Pl'].max() + 1)],
                                 textposition='outside',
                                 hoverinfo='none',
                                 name=i
                             ))
    fig.update_xaxes(title_text="Position", range=[0.5, 20],
                     tickvals=np.arange(1, dfff['Pl'].max() + 1, 1))
    fig.update_yaxes(title_text="No. of finishes",
                     showticklabels=False,
                     showgrid=False,
                     tickvals=np.arange(1, max([len(dfff.loc[dfff['Pl'] == i]) for i in range(1, dfff['Pl'].max() + 1)]) + 1, 1))
    fig.update_layout(title_text='Position-wise results',
                      showlegend=True,
                      autosize=True,
                      xaxis={'showgrid': True, 'zeroline': True, 'categoryorder': 'total ascending', 'gridcolor': '#082255', 'zerolinecolor': '#082255'},
                      yaxis={'showgrid': True, 'zeroline': True, 'gridcolor': '#082255', 'zerolinecolor': '#082255'},
                      plot_bgcolor="#061e44",
                      paper_bgcolor="#082255",
                      font_color="white"
                      )
    return fig


def generatebarfavorite(dfff, searchtype):
    F = list(['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F0'])
    fig = go.Figure()
    for i in dfff[searchtype].unique():
        fig.add_trace(go.Bar(x=F,
                                 y=[len(dfff.loc[dfff[searchtype]==i].loc[dfff.loc[dfff[searchtype]==i]['FO'] == n]) for n in F],
                                 # marker_color = c,
                                 text=[len(dfff.loc[dfff[searchtype]==i].loc[dfff.loc[dfff[searchtype]==i]['FO'] == n]) for n in F],
                                 textposition='outside',
                                 hoverinfo='none',
                                 name=i
                             ))
    fig.update_xaxes(title_text="Favorite",
                     tickvals=F)
    fig.update_yaxes(title_text="No. of times",
                     showticklabels=False,
                     showgrid=False,
                     tickvals=np.arange(1, max([len(dfff.loc[dfff['FO'] == i]) for i in F])))
    fig.update_layout(title_text='Final Odds',
                      showlegend=True,
                      autosize=True,
                      xaxis={'showgrid': True, 'zeroline': True, 'gridcolor': '#082255', 'zerolinecolor': '#082255'},
                      yaxis={'showgrid': True, 'zeroline': True, 'gridcolor': '#082255', 'zerolinecolor': '#082255'},
                      plot_bgcolor="#061e44",
                      paper_bgcolor="#082255",
                      font_color="white"
                      )
    return fig


def conjunction(type, *conditions):
    if type:
        return functools.reduce(np.logical_or, conditions)
    else:
        return functools.reduce(np.logical_and, conditions)


def streak(dfff):
    #find streaks and maxstreaklengths
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
    for i in dfff['Pl'][::-1]:
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
        avgPS = sumPS/numPS
    if numNPS > 0:
        avgNPS = sumNPS/numNPS
    # print(streak)
    return [streak, maxPS, maxNPS, avgPS, avgNPS]

def streak_media(type,res):
    codes=[]
    if type == 'win':
        codes = ['F/W', 'S/W', 'T/W']
    elif type == 'SHP':
        codes = ['F/2', 'S/2', 'T/2']
    elif type =='THP':
        codes = ['F/3', 'S/3', 'T/3']
    elif type =='plc':
        codes = ['W','W','W']
    #find streaks and maxstreaklengths
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
    for i in res[::-1]:
        if i == 'L':
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
        if i in codes:
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
            if i == codes[0]:
                if shpstreak != 0:
                    streak += " " + shps
                elif thpstreak != 0:
                    streak += " " + thps
                shpstreak = 0
                thpstreak = 0
                wstreak += 1
                ws = ws.replace(ws, "{}{}".format(wstreak,codes[0]))
            elif i == codes[2]:
                if wstreak != 0:
                    streak += " " + ws
                elif shpstreak != 0:
                    streak += " " + shps
                wstreak = 0
                shpstreak = 0
                thpstreak += 1
                thps = thps.replace(thps, "{}{}".format(thpstreak,codes[2]))
            elif i == codes[1]:
                if wstreak != 0:
                    streak += " " + ws
                elif thpstreak != 0:
                    streak += " " + thps
                wstreak = 0
                thpstreak = 0
                shpstreak += 1
                shps = shps.replace(shps, "{}{}".format(shpstreak,codes[1]))
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
    # print(streak)
    return [streak, maxPS, maxNPS]


def maxminstreaks(n, dfff):
    lstreak = 0
    wstreak = 0
    maxlstreak = 0
    maxwstreak = 0
    sumLstreak = 0
    sumWstreak = 0
    numLstreak = 0
    numWstreak = 0
    for i in dfff['Pl']:
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
        avgWstreak = sumWstreak/numWstreak
    if numLstreak:
        avgLstreak = sumLstreak/numLstreak

    return [maxwstreak, maxlstreak, avgWstreak, avgLstreak]


def maxminstreaks_media(n, res):
    lstreak = 0
    wstreak = 0
    maxlstreak = 0
    maxwstreak = 0
    sumLstreak = 0
    sumWstreak = 0
    numLstreak = 0
    numWstreak = 0
    for i in res:
        if i in n:
            if wstreak > 0:
                numWstreak += 1
                sumWstreak += wstreak
            if wstreak > maxwstreak:
                maxwstreak = wstreak
            wstreak = 0
            lstreak += 1
        if i not in n:
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

    return [maxwstreak, maxlstreak]


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


def generate_li_adv(n, pl, odds, box_number):
    if odds != np.nan:
        odds=round(odds,1)
    codes = ["W", "SHP", "THP", "0"]
    if box_number % 125 >= 25:
        if box_number % 125 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--win', style={"margin-top":"15px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--loss', style={"margin-top":"15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--loss', style={"margin-top":"15px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--win', style={"margin-top":"15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--loss', style={"margin-top":"15px"})
        elif box_number % 25 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--win last')
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--loss last')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--loss last')

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--win last')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--loss last')
        elif box_number%5 == 0 and box_number !=0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--win', style={"margin-left":"20px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--loss', style={"margin-left":"20px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--loss', style={"margin-left":"20px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--win', style={"margin-left":"20px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--loss', style={"margin-left":"20px"})
        else:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--win')
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--loss')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--loss')

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--win')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='standings-table__outcome standings-table__outcome--loss')
    else:
        if box_number % 125 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--win',
                                   style={"margin-top": "15px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--loss',
                                   style={"margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--loss',
                                   style={"margin-top": "15px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--win',
                                   style={"margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--loss',
                                   style={"margin-top": "15px"})
        elif box_number % 25 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--win last',
                                   style={"margin-top": "15px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--loss last',
                                   style={"margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--loss last',
                                   style={"margin-top": "15px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--win last',
                                   style={"margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--loss last',
                                   style={"margin-top": "15px"})
        elif box_number % 5 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--win',
                                   style={"margin-left": "20px", "margin-top": "15px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--loss',
                                   style={"margin-left": "20px", "margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--loss',
                                   style={"margin-left": "20px", "margin-top": "15px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--win',
                                   style={"margin-left": "20px", "margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--loss',
                                   style={"margin-left": "20px", "margin-top": "15px"})
        else:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--win',
                                   style={"margin-top": "15px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--loss',
                                   style={"margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--loss',
                                   style={"margin-top": "15px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--win',
                                   style={"margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='standings-table__outcome standings-table__outcome--loss',
                                   style={"margin-top": "15px"})


def generate_li_for_fav_sb(n, pl, odds, box_number):
    if odds != np.nan:
        odds=round(odds,1)
    codes = ["W", "SHP", "THP", "0"]
    if box_number % 175 >= 35:
        if box_number % 175 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--win', style={"margin-top":"5px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--loss', style={"margin-top":"5px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--loss', style={"margin-top":"5px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--win', style={"margin-top":"5px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--loss', style={"margin-top":"5px"})
        elif box_number % 35 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--win last')
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--loss last')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--loss last')

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--win last')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--loss last')
        elif box_number%5 == 0 and box_number !=0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--win', style={"margin-left":"15px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--loss', style={"margin-left":"15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--loss', style={"margin-left":"15px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--win', style={"margin-left":"15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--loss', style={"margin-left":"15px"})
        else:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--win')
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--loss')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--loss')

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--win')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='fav_standings-table__outcome fav_standings-table__outcome--loss')
    else:
        if box_number % 175 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--win',
                                   style={"margin-top": "15px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--loss',
                                   style={"margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--loss',
                                   style={"margin-top": "15px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--win',
                                   style={"margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--loss',
                                   style={"margin-top": "15px"})
        elif box_number % 35 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--win last',
                                   style={"margin-top": "15px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--loss last',
                                   style={"margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--loss last',
                                   style={"margin-top": "15px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--win last',
                                   style={"margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--loss last',
                                   style={"margin-top": "15px"})
        elif box_number % 5 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--win',
                                   style={"margin-left": "15px", "margin-top": "15px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--loss',
                                   style={"margin-left": "15px", "margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--loss',
                                   style={"margin-left": "15px", "margin-top": "15px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--win',
                                   style={"margin-left": "15px", "margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--loss',
                                   style={"margin-left": "15px", "margin-top": "15px"})
        else:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--win',
                                   style={"margin-top": "15px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--loss',
                                   style={"margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--loss',
                                   style={"margin-top": "15px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--win',
                                   style={"margin-top": "15px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='fav_standings-table__outcome fav_standings-table__outcome--loss',
                                   style={"margin-top": "15px"})


def generate_li_for_fav_split(n, pl, odds, box_number):
    if odds != np.nan:
        odds=round(odds,1)
    codes = ["W", "SHP", "THP", "0"]
    if box_number % 175 >= 35:
        if box_number % 175 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win')
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss')

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss')
        elif box_number % 35 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win last')
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss last')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss last')

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win last')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss last')
        elif box_number%5 == 0 and box_number !=0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win', style={"margin-left":"10px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss', style={"margin-left":"10px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss', style={"margin-left":"10px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl-1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win', style={"margin-left":"10px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss', style={"margin-left":"10px"})
        else:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win')
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss')

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win')
                else:
                    return html.Li(["{}".format(pl), html.Br(), html.Div("{}".format(odds),style={'background-color':'#082255'})], className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss')
    else:
        if box_number % 175 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win')
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss')
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss')

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win')
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss')
        elif box_number % 35 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win last')
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss last')
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss last')

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win last')
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss last')
        elif box_number % 5 == 0 and box_number != 0:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win',
                                   style={"margin-left": "10px"})
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss',
                                   style={"margin-left": "10px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss',
                                   style={"margin-left": "10px"})

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win',
                                   style={"margin-left": "10px"})
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss',
                                   style={"margin-left": "10px"})
        else:
            if n < 4:
                if pl == n:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win')
                elif pl < 4:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss')
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss')

            else:
                if pl == 1 or pl == 2 or pl == 3:
                    return html.Li(["{}".format(codes[pl - 1]), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--win')
                else:
                    return html.Li(["{}".format(pl), html.Br(),
                                    html.Div("{}".format(odds), style={'background-color': '#082255'})],
                                   className='splitfav_standings-table__outcome splitfav_standings-table__outcome--loss')


def table_counts(dff, odds, f):
    return dff[odds].value_counts().get(f, 0)


def unique_non_null(s):
    return s.dropna().unique()


def generate_overall_stats(Trainer_stats, Jockey_stats, Trainer_jockey_stats, Centre, media_df):
    centre_list={'BLR':'Bangalore', 'HYD':'Hyderabad', 'KOL':'Kolkata', 'MAA':'MAA', 'MYS':'Mysore', 'MUM':'Mumbai', 'PUN':'Pune'}
    overall_stats = [
        html.H3("{} Centre Stats".format(centre_list[Centre]),
                style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                       'font-family': 'Arial Black'}),
        html.Div([
            html.Div("Trainer", className='mini_container',
                     style={'textAlign': 'center', 'font-family': 'Arial Black', 'width': '55px', 'padding-top': '60px',
                            'color': '#fbff00'}),
            html.Div([html.H6("Most wins", className='statboxheader'),
                      html.Div("{}".format(Trainer_stats['W'].idxmax()), className='statboxcontent'),
                      html.Div("{}".format(Trainer_stats['W'].max()), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest win%", className='statboxheader'),
                      html.Div("{}".format(Trainer_stats['W%'].idxmax()), className='statboxcontent'),
                      html.Div("{}%".format(Trainer_stats['W%'].max()), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest win streak", className='statboxheader'),
                      html.Div("{}".format(Trainer_stats['LWS'].idxmax()), className='statboxcontent'),
                      html.Div("{}".format(Trainer_stats['LWS'].max()), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest losing streak", className='statboxheader'),
                      html.Div("{}".format(Trainer_stats['LLS'].idxmax()), className='statboxcontent'),
                      html.Div("{}".format(Trainer_stats['LLS'].max()), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest F1%", className='statboxheader'),
                      html.Div("{}".format(Trainer_stats['F1%'].idxmax()), className='statboxcontent'),
                      html.Div("{}%".format(Trainer_stats['F1%'].max()), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'}),
        html.Div([
            html.Div("Jockey", className='mini_container',
                     style={'textAlign': 'center', 'font-family': 'Arial Black', 'width': '55px', 'padding-top': '60px',
                            'color': '#fbff00'}),
            html.Div([html.H6("Most wins", className='statboxheader'),
                      html.Div("{}".format(Jockey_stats['W'].idxmax()), className='statboxcontent'),
                      html.Div("{}".format(Jockey_stats['W'].max()), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest win%", className='statboxheader'),
                      html.Div("{}".format(Jockey_stats['W%'].idxmax()), className='statboxcontent'),
                      html.Div("{}%".format(Jockey_stats['W%'].max()), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest win streak", className='statboxheader'),
                      html.Div("{}".format(Jockey_stats['LWS'].idxmax()), className='statboxcontent'),
                      html.Div("{}".format(Jockey_stats['LWS'].max()), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest losing streak", className='statboxheader'),
                      html.Div("{}".format(Jockey_stats['LLS'].idxmax()), className='statboxcontent'),
                      html.Div("{}".format(Jockey_stats['LLS'].max()), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest F1%", className='statboxheader'),
                      html.Div("{}".format(Jockey_stats['F1%'].idxmax()), className='statboxcontent'),
                      html.Div("{}%".format(Jockey_stats['F1%'].max()), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'}),
        html.Div([
            html.Div("Trainer/ Jockey Combo", className='mini_container',
                     style={'textAlign': 'center', 'font-family': 'Arial Black', 'width': '55px', 'padding-top': '60px',
                            'color': '#fbff00'}),
            html.Div([html.H6("Most wins", className='statboxheader'),
                      html.Div(["{}/".format(Trainer_jockey_stats['W'].idxmax()[0]),
                                html.Br(), "{}".format(Trainer_jockey_stats['W'].idxmax()[1])],
                               className='statboxcontent'),
                      html.Div("{}".format(Trainer_jockey_stats['W'].max()), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest win%", className='statboxheader'),
                      html.Div(["{}/".format(
                          Trainer_jockey_stats[Trainer_jockey_stats['TR'] >= 5]['W%'].idxmax()[0]),
                                html.Br(), "{}".format(
                              Trainer_jockey_stats[Trainer_jockey_stats['TR'] >= 5]['W%'].idxmax()[1])],
                               className='statboxcontent'),
                      html.Div("{}%".format(Trainer_jockey_stats[Trainer_jockey_stats['TR'] >= 5]['W%'].max()),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest win streak", className='statboxheader'),
                      html.Div(["{}/".format(Trainer_jockey_stats['LWS'].idxmax()[0]),
                                html.Br(), "{}".format(Trainer_jockey_stats['LWS'].idxmax()[1])],
                               className='statboxcontent'),
                      html.Div("{}".format(Trainer_jockey_stats['LWS'].max()), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest losing streak", className='statboxheader'),
                      html.Div(["{}/".format(Trainer_jockey_stats['LLS'].idxmax()[0]),
                                html.Br(), "{}".format(Trainer_jockey_stats['LLS'].idxmax()[1])],
                               className='statboxcontent'),
                      html.Div("{}".format(Trainer_jockey_stats['LLS'].max()), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest F1%", className='statboxheader'),
                      html.Div(["{}/".format(
                          Trainer_jockey_stats[Trainer_jockey_stats['TR'] >= 5]['F1%'].idxmax()[0]),
                                html.Br(), "{}".format(
                              Trainer_jockey_stats[Trainer_jockey_stats['TR'] >= 5]['F1%'].idxmax()[1])],
                               className='statboxcontent'),
                      html.Div(
                          "{}%".format(Trainer_jockey_stats[Trainer_jockey_stats['TR'] >= 5]['F1%'].max()),
                          className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'}),
        html.H3("Media Tips", style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#e5ff00', 'margin': '0px',
                                     'font-family': 'Arial Black'}),
        html.Div([
            html.Div([html.H6("TurfBee", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('TurfBee', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('TurfBee', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('TurfBee', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('TurfBee', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("BOL", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('BOL', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('BOL', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('BOL', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('BOL', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Dina Thanthi", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('Dina Thanthi', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('Dina Thanthi', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('Dina Thanthi', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('Dina Thanthi', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Experts", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('Experts', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('Experts', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('Experts', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('Experts', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Media Man", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('Media Man', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('Media Man', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('Media Man', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('Media Man', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("S. Today", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('S. Today', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('S. Today', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('S. Today', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('S. Today', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'}),
        html.Div([
            html.Div([html.H6("TRS", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('TRS', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('TRS', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('TRS', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('TRS', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Telegraph", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('Telegraph', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('Telegraph', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('Telegraph', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('Telegraph', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Telangana Today", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('Telangana Today', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('Telangana Today', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('Telangana Today', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('Telangana Today', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Telangana Times", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('Telangana Times', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('Telangana Times', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('Telangana Times', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('Telangana Times', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Andhra Jyothi", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('Andhra Jyothi', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('Andhra Jyothi', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('Andhra Jyothi', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('Andhra Jyothi', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Asian Age", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('Asian Age', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('Asian Age', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('Asian Age', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('Asian Age', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'}),
        html.Div([
            html.Div([html.H6("Times of India", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('Times of India', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('Times of India', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('Times of India', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('Times of India', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("The Hindu", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('The Hindu', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('The Hindu', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('The Hindu', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('The Hindu', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Mumbai Mirror", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('Mumbai Mirror', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('Mumbai Mirror', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('Mumbai Mirror', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('Mumbai Mirror', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Deccan Herald", className='mediatipsheader'),
                      html.Div("TR: {}".format(media_df['TR'].get('Deccan Herald', np.nan)), className='mediatipTR'),
                      html.Div("F-->W: {}%".format(media_df['F'].get('Deccan Herald', np.nan)), className='mediatipF'),
                      html.Div("S-->W: {}%".format(media_df['S'].get('Deccan Herald', np.nan)), className='mediatipS'),
                      html.Div("T-->W: {}%".format(media_df['T'].get('Deccan Herald', np.nan)), className='mediatipT')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("", className='mediatipsheader'),
                      html.Div("", className='mediatipF'),
                      html.Div("", className='mediatipS'),
                      html.Div("", className='mediatipT')],
                     className="mini_container", style={'width': '200px', 'background-color': '#061e44'}),
            html.Div([html.H6("", className='mediatipsheader'),
                      html.Div("", className='mediatipF'),
                      html.Div("", className='mediatipS'),
                      html.Div("", className='mediatipT')],
                     className="mini_container", style={'width': '200px', 'background-color': '#061e44'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'})
    ]
    return overall_stats



def generate_trainer_jockey_stats(df):
    foo = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F17', 'F18', 'F19']
    pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    n = df.groupby(["Trainer","Jockey","Pl"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = df.groupby(["Trainer","Jockey"], as_index=True).size().to_frame()
    l = df.groupby(["Trainer", "Jockey", "FO"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in foo if i not in l.columns]
    for col in mis_col:
        l[col] = [0] * len(l)
    l = l[foo]
    k = df.groupby(["Trainer", "Jockey"], as_index=True)['Pl'].agg([maxwinstreaks,maxlosestreaks])
    j = df.groupby(["Trainer", "Jockey"], as_index=True)['Win_OPEN'].sum()
    i = df.groupby(["Trainer", "Jockey"], as_index=True)['Win_LATEST'].sum()
    h = df.groupby(["Trainer", "Jockey"], as_index=True)['Win_MIDDLE'].sum()
    g = df.groupby(["Trainer", "Jockey"], as_index=True)['Win_FINAL'].sum()
    Trainer_jockey_stats = pd.concat([n, m, l, k, j, i, h, g], axis=1)
    Trainer_jockey_stats.rename(columns={0:'TR', 1:"W", 2:"SHP", 3:"THP", 'maxwinstreaks':'LWS', 'maxlosestreaks':'LLS'}, inplace=True)
    Trainer_jockey_stats["W%"]=((Trainer_jockey_stats['W']/Trainer_jockey_stats['TR'])*100).round(2)
    Trainer_jockey_stats["SHP%"]=((Trainer_jockey_stats['SHP']/Trainer_jockey_stats['TR'])*100).round(2)
    Trainer_jockey_stats["THP%"]=((Trainer_jockey_stats['THP']/Trainer_jockey_stats['TR'])*100).round(2)
    Trainer_jockey_stats["Plc%"]=(((Trainer_jockey_stats['W']+Trainer_jockey_stats['SHP']+Trainer_jockey_stats['THP'])/Trainer_jockey_stats['TR'])*100).round(2)
    Trainer_jockey_stats["F1%"]=((Trainer_jockey_stats["F1"]/Trainer_jockey_stats['TR'])*100).round(2)
    Trainer_jockey_stats["F2%"]=((Trainer_jockey_stats["F2"]/Trainer_jockey_stats['TR'])*100).round(2)
    Trainer_jockey_stats["F3%"]=((Trainer_jockey_stats["F3"]/Trainer_jockey_stats['TR'])*100).round(2)
    Trainer_jockey_stats["Plc"]=Trainer_jockey_stats['W']+Trainer_jockey_stats['SHP']+Trainer_jockey_stats['THP']
    Trainer_jockey_stats["BO"]=Trainer_jockey_stats["TR"]-Trainer_jockey_stats["Plc"]
    Trainer_jockey_stats["TR%"] = ((Trainer_jockey_stats["TR"] / len(df)) * 100).round(2)
    Trainer_jockey_stats['Win_OPEN'] = Trainer_jockey_stats['Win_OPEN'] / Trainer_jockey_stats['W']
    Trainer_jockey_stats["edgeO"] = round(
        Trainer_jockey_stats['W%']/100 * Trainer_jockey_stats['Win_OPEN'] - (100 - Trainer_jockey_stats['W%'])/100, 2)
    Trainer_jockey_stats['Win_LATEST'] = Trainer_jockey_stats['Win_LATEST'] / Trainer_jockey_stats['W']
    Trainer_jockey_stats["edgeL"] = round(
        Trainer_jockey_stats['W%']/100 * Trainer_jockey_stats['Win_LATEST'] - (100 - Trainer_jockey_stats['W%'])/100, 2)
    Trainer_jockey_stats['Win_MIDDLE'] = Trainer_jockey_stats['Win_MIDDLE'] / Trainer_jockey_stats['W']
    Trainer_jockey_stats["edgeM"] = round(
        Trainer_jockey_stats['W%']/100 * Trainer_jockey_stats['Win_MIDDLE'] - (100 - Trainer_jockey_stats['W%'])/100, 2)
    Trainer_jockey_stats['Win_FINAL'] = Trainer_jockey_stats['Win_FINAL'] / Trainer_jockey_stats['W']
    Trainer_jockey_stats["edgeF"] = round(
        Trainer_jockey_stats['W%']/100 * Trainer_jockey_stats['Win_FINAL'] - (100 - Trainer_jockey_stats['W%'])/100, 2)
    return Trainer_jockey_stats

def generate_trainer_stats(df):
    foo = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F17',
          'F18', 'F19']
    pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    n=df.groupby(["Trainer","Pl"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = df.groupby(["Trainer"], as_index=True).size().to_frame()
    l=df.groupby(["Trainer","FO"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in foo if i not in l.columns]
    for col in mis_col:
        l[col] = [0] * len(l)
    l = l[foo]
    k=df.groupby(["Trainer"], as_index=True)['Pl'].agg([maxwinstreaks,maxlosestreaks])
    j = df.groupby(["Trainer"], as_index=True)['Win_OPEN'].sum()
    i = df.groupby(["Trainer"], as_index=True)['Win_LATEST'].sum()
    h = df.groupby(["Trainer"], as_index=True)['Win_MIDDLE'].sum()
    g = df.groupby(["Trainer"], as_index=True)['Win_FINAL'].sum()
    Trainer_stats = pd.concat([n, m, l, k, j, i, h, g], axis=1)
    Trainer_stats.rename(columns={0:'TR', 1:"W", 2:"SHP", 3:"THP", 'maxwinstreaks':'LWS', 'maxlosestreaks':'LLS'}, inplace=True)
    Trainer_stats["W%"]=((Trainer_stats["W"]/Trainer_stats['TR'])*100).round(2)
    Trainer_stats["SHP%"]=((Trainer_stats["SHP"]/Trainer_stats['TR'])*100).round(2)
    Trainer_stats["THP%"]=((Trainer_stats["THP"]/Trainer_stats['TR'])*100).round(2)
    Trainer_stats["Plc%"]=(((Trainer_stats["W"]+Trainer_stats["SHP"]+Trainer_stats["THP"])/Trainer_stats['TR'])*100).round(2)
    Trainer_stats["F1%"]=((Trainer_stats["F1"]/Trainer_stats['TR'])*100).round(2)
    Trainer_stats["F2%"]=((Trainer_stats["F2"]/Trainer_stats['TR'])*100).round(2)
    Trainer_stats["F3%"]=((Trainer_stats["F3"]/Trainer_stats['TR'])*100).round(2)
    Trainer_stats["Plc"]=Trainer_stats["W"]+Trainer_stats["SHP"]+Trainer_stats["THP"]
    Trainer_stats["BO"]=Trainer_stats["TR"]-Trainer_stats["Plc"]
    Trainer_stats["TR%"] = ((Trainer_stats["TR"] / len(df)) * 100).round(2)
    Trainer_stats['Win_OPEN'] = Trainer_stats['Win_OPEN']/Trainer_stats['W']
    Trainer_stats["edgeO"]=round(Trainer_stats['W%']/100*Trainer_stats['Win_OPEN'] - (100 - Trainer_stats['W%'])/100,2)
    Trainer_stats['Win_LATEST'] = Trainer_stats['Win_LATEST']/Trainer_stats['W']
    Trainer_stats["edgeL"]=round(Trainer_stats['W%']/100*Trainer_stats['Win_LATEST'] - (100 - Trainer_stats['W%'])/100,2)
    Trainer_stats['Win_MIDDLE'] = Trainer_stats['Win_MIDDLE']/Trainer_stats['W']
    Trainer_stats["edgeM"]=round(Trainer_stats['W%']/100*Trainer_stats['Win_MIDDLE'] - (100 - Trainer_stats['W%'])/100,2)
    Trainer_stats['Win_FINAL'] = Trainer_stats['Win_FINAL']/Trainer_stats['W']
    Trainer_stats["edgeF"]=round(Trainer_stats['W%']/100*Trainer_stats['Win_FINAL'] - (100 - Trainer_stats['W%'])/100,2)
    return Trainer_stats

def generate_jockey_stats(df):
    foo = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F17',
          'F18', 'F19']
    pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    n = df.groupby(["Jockey","Pl"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = df.groupby(["Jockey"], as_index=True).size().to_frame()
    l=df.groupby(["Jockey","FO"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in foo if i not in l.columns]
    for col in mis_col:
        l[col] = [0] * len(l)
    l = l[foo]
    k=df.groupby(["Jockey"], as_index=True)['Pl'].agg([maxwinstreaks,maxlosestreaks])
    j = df.groupby(["Jockey"], as_index=True)['Win_OPEN'].sum()
    i = df.groupby(["Jockey"], as_index=True)['Win_LATEST'].sum()
    h = df.groupby(["Jockey"], as_index=True)['Win_MIDDLE'].sum()
    g = df.groupby(["Jockey"], as_index=True)['Win_FINAL'].sum()
    Jockey_stats = pd.concat([n, m, l, k, j, i, h, g], axis=1)
    Jockey_stats.rename(columns={0:'TR', 1:"W", 2:"SHP", 3:"THP", 'maxwinstreaks':'LWS', 'maxlosestreaks':'LLS'}, inplace=True)
    Jockey_stats["W%"]=((Jockey_stats["W"]/Jockey_stats['TR'])*100).round(2)
    Jockey_stats["SHP%"]=((Jockey_stats["SHP"]/Jockey_stats['TR'])*100).round(2)
    Jockey_stats["THP%"]=((Jockey_stats["THP"]/Jockey_stats['TR'])*100).round(2)
    Jockey_stats["Plc%"]=(((Jockey_stats["W"]+Jockey_stats["SHP"]+Jockey_stats["THP"])/Jockey_stats['TR'])*100).round(2)
    Jockey_stats["F1%"]=((Jockey_stats["F1"]/Jockey_stats['TR'])*100).round(2)
    Jockey_stats["F2%"]=((Jockey_stats["F2"]/Jockey_stats['TR'])*100).round(2)
    Jockey_stats["F3%"]=((Jockey_stats["F3"]/Jockey_stats['TR'])*100).round(2)
    Jockey_stats["Plc"]=Jockey_stats["W"]+Jockey_stats["SHP"]+Jockey_stats["THP"]
    Jockey_stats["BO"]=Jockey_stats["TR"]-Jockey_stats["Plc"]
    Jockey_stats["TR%"] = ((Jockey_stats["TR"] / len(df)) * 100).round(2)
    Jockey_stats['Win_OPEN'] = Jockey_stats['Win_OPEN']/Jockey_stats['W']
    Jockey_stats["edgeO"]=round(Jockey_stats['W%']/100*Jockey_stats['Win_OPEN'] - (100 - Jockey_stats['W%'])/100,2)
    Jockey_stats['Win_LATEST'] = Jockey_stats['Win_LATEST']/Jockey_stats['W']
    Jockey_stats["edgeL"]=round(Jockey_stats['W%']/100*Jockey_stats['Win_LATEST'] - (100 - Jockey_stats['W%'])/100,2)
    Jockey_stats['Win_MIDDLE'] = Jockey_stats['Win_MIDDLE']/Jockey_stats['W']
    Jockey_stats["edgeM"]=round(Jockey_stats['W%']/100*Jockey_stats['Win_MIDDLE'] - (100 - Jockey_stats['W%'])/100,2)
    Jockey_stats['Win_FINAL'] = Jockey_stats['Win_FINAL']/Jockey_stats['W']
    Jockey_stats["edgeF"]=round(Jockey_stats['W%']/100*Jockey_stats['Win_FINAL'] - (100 - Jockey_stats['W%'])/100,2)
    return Jockey_stats

def generate_jockey_trainer_stats(df):
    foo = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F17',
           'F18', 'F19']
    pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    n = df.groupby(["Jockey", "Trainer", "Pl"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = df.groupby(["Jockey","Trainer"], as_index=True).size().to_frame()
    l = df.groupby(["Jockey", "Trainer", "FO"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in foo if i not in l.columns]
    for col in mis_col:
        l[col] = [0] * len(l)
    l = l[foo]
    k = df.groupby(["Jockey", "Trainer"], as_index=True)['Pl'].agg([maxwinstreaks,maxlosestreaks])
    j = df.groupby(["Jockey", "Trainer"], as_index=True)['Win_OPEN'].sum()
    i = df.groupby(["Jockey", "Trainer"], as_index=True)['Win_LATEST'].sum()
    h = df.groupby(["Jockey", "Trainer"], as_index=True)['Win_MIDDLE'].sum()
    g = df.groupby(["Jockey", "Trainer"], as_index=True)['Win_FINAL'].sum()
    Jockey_Trainer_stats = pd.concat([n, m, l, k, j, i, h, g], axis=1)
    Jockey_Trainer_stats.rename(columns={0:'TR', 1:"W", 2:"SHP", 3:"THP", 'maxwinstreaks':'LWS', 'maxlosestreaks':'LLS'}, inplace=True)
    Jockey_Trainer_stats["W%"]=((Jockey_Trainer_stats["W"]/Jockey_Trainer_stats['TR'])*100).round(2)
    Jockey_Trainer_stats["SHP%"]=((Jockey_Trainer_stats["SHP"]/Jockey_Trainer_stats['TR'])*100).round(2)
    Jockey_Trainer_stats["THP%"]=((Jockey_Trainer_stats["THP"]/Jockey_Trainer_stats['TR'])*100).round(2)
    Jockey_Trainer_stats["Plc%"]=(((Jockey_Trainer_stats["W"]+Jockey_Trainer_stats["SHP"]+Jockey_Trainer_stats["THP"])/Jockey_Trainer_stats['TR'])*100).round(2)
    Jockey_Trainer_stats["F1%"]=((Jockey_Trainer_stats["F1"]/Jockey_Trainer_stats['TR'])*100).round(2)
    Jockey_Trainer_stats["F2%"]=((Jockey_Trainer_stats["F2"]/Jockey_Trainer_stats['TR'])*100).round(2)
    Jockey_Trainer_stats["F3%"]=((Jockey_Trainer_stats["F3"]/Jockey_Trainer_stats['TR'])*100).round(2)
    Jockey_Trainer_stats["Plc"]=Jockey_Trainer_stats["W"]+Jockey_Trainer_stats["SHP"]+Jockey_Trainer_stats["THP"]
    Jockey_Trainer_stats["BO"]=Jockey_Trainer_stats["TR"]-Jockey_Trainer_stats["Plc"]
    Jockey_Trainer_stats["TR%"] = ((Jockey_Trainer_stats["TR"] / len(df)) * 100).round(2)
    Jockey_Trainer_stats['Win_OPEN'] = Jockey_Trainer_stats['Win_OPEN'] / Jockey_Trainer_stats['W']
    Jockey_Trainer_stats["edgeO"] = round(
        Jockey_Trainer_stats['W%']/100 * Jockey_Trainer_stats['Win_OPEN'] - (100 - Jockey_Trainer_stats['W%'])/100, 2)
    Jockey_Trainer_stats['Win_LATEST'] = Jockey_Trainer_stats['Win_LATEST'] / Jockey_Trainer_stats['W']
    Jockey_Trainer_stats["edgeL"] = round(
        Jockey_Trainer_stats['W%']/100 * Jockey_Trainer_stats['Win_LATEST'] - (100 - Jockey_Trainer_stats['W%'])/100, 2)
    Jockey_Trainer_stats['Win_MIDDLE'] = Jockey_Trainer_stats['Win_MIDDLE'] / Jockey_Trainer_stats['W']
    Jockey_Trainer_stats["edgeM"] = round(
        Jockey_Trainer_stats['W%']/100 * Jockey_Trainer_stats['Win_MIDDLE'] - (100 - Jockey_Trainer_stats['W%'])/100, 2)
    Jockey_Trainer_stats['Win_FINAL'] = Jockey_Trainer_stats['Win_FINAL'] / Jockey_Trainer_stats['W']
    Jockey_Trainer_stats["edgeF"] = round(
        Jockey_Trainer_stats['W%']/100 * Jockey_Trainer_stats['Win_FINAL'] - (100 - Jockey_Trainer_stats['W%'])/100, 2)
    return Jockey_Trainer_stats

def generate_FO_stats(df):
    foo = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F17',
           'F18', 'F19']
    pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    n = df.groupby(["FO","Pl"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = df.groupby(["FO"], as_index=True).size().to_frame()
    k = df.groupby(["FO"], as_index=True)['Pl'].agg([maxwinstreaks, maxlosestreaks])
    j=df.groupby(["FO"], as_index=True)['Win_OPEN'].sum()
    i=df.groupby(["FO"], as_index=True)['Win_LATEST'].sum()
    h=df.groupby(["FO"], as_index=True)['Win_MIDDLE'].sum()
    g=df.groupby(["FO"], as_index=True)['Win_FINAL'].sum()
    fodf = pd.concat([n, m, k, j, i, h, g], axis=1)
    fodf.rename(columns={0:'TR', 1:"W", 2:"SHP", 3:"THP", 'maxwinstreaks':'LWS', 'maxlosestreaks':'LLS'}, inplace=True)
    fodf["W%"]=((fodf["W"]/fodf['TR'])*100).round(2)
    fodf["SHP%"]=((fodf["SHP"]/fodf['TR'])*100).round(2)
    fodf["THP%"]=((fodf["THP"]/fodf['TR'])*100).round(2)
    fodf["Plc%"]=(((fodf["W"]+fodf["SHP"]+fodf["THP"])/fodf['TR'])*100).round(2)
    fodf["Plc"]=fodf["W"]+fodf["SHP"]+fodf["THP"]
    fodf["BO"]=fodf["TR"]-fodf["Plc"]
    fodf["TR%"] = ((fodf["TR"] / len(df)) * 100).round(2)
    fodf['Win_OPEN'] = fodf['Win_OPEN'] / fodf['W']
    fodf["edgeO"] = round(fodf['W%']/100 * fodf['Win_OPEN'] - (100 - fodf['W%'])/100, 2)
    fodf['Win_LATEST'] = fodf['Win_LATEST'] / fodf['W']
    fodf["edgeL"] = round(
        fodf['W%']/100 * fodf['Win_LATEST'] - (100 - fodf['W%'])/100, 2)
    fodf['Win_MIDDLE'] = fodf['Win_MIDDLE'] / fodf['W']
    fodf["edgeM"] = round(
        fodf['W%']/100 * fodf['Win_MIDDLE'] - (100 - fodf['W%'])/100, 2)
    fodf['Win_FINAL'] = fodf['Win_FINAL'] / fodf['W']
    fodf["edgeF"] = round(fodf['W%']/100 * fodf['Win_FINAL'] - (100 - fodf['W%'])/100,
                                        2)
    f = [i for i in foo if i in fodf.index]
    fodf = fodf.reindex(f)
    return fodf

def generate_MO_stats(df):
    foo = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F17',
           'F18', 'F19']
    pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    n = df.groupby(["MO","Pl"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = df.groupby(["MO"], as_index=True).size().to_frame()
    k = df.groupby(["MO"], as_index=True)['Pl'].agg([maxwinstreaks, maxlosestreaks])
    j=df.groupby(["MO"], as_index=True)['Win_OPEN'].sum()
    i=df.groupby(["MO"], as_index=True)['Win_LATEST'].sum()
    h=df.groupby(["MO"], as_index=True)['Win_MIDDLE'].sum()
    g=df.groupby(["MO"], as_index=True)['Win_FINAL'].sum()
    modf = pd.concat([n, m, k, j, i, h, g], axis=1)
    modf.rename(columns={0:'TR', 1:"W", 2:"SHP", 3:"THP", 'maxwinstreaks':'LWS', 'maxlosestreaks':'LLS'}, inplace=True)
    modf["W%"]=((modf["W"]/modf['TR'])*100).round(2)
    modf["SHP%"]=((modf["SHP"]/modf['TR'])*100).round(2)
    modf["THP%"]=((modf["THP"]/modf['TR'])*100).round(2)
    modf["Plc%"]=(((modf["W"]+modf["SHP"]+modf["THP"])/modf['TR'])*100).round(2)
    modf["Plc"]=modf["W"]+modf["SHP"]+modf["THP"]
    modf["BO"]=modf["TR"]-modf["Plc"]
    modf["TR%"] = ((modf["TR"] / len(df)) * 100).round(2)
    modf['Win_OPEN'] = modf['Win_OPEN'] / modf['W']
    modf["edgeO"] = round(modf['W%']/100 * modf['Win_OPEN'] - (100 - modf['W%'])/100, 2)
    modf['Win_LATEST'] = modf['Win_LATEST'] / modf['W']
    modf["edgeL"] = round(
        modf['W%']/100 * modf['Win_LATEST'] - (100 - modf['W%'])/100, 2)
    modf['Win_MIDDLE'] = modf['Win_MIDDLE'] / modf['W']
    modf["edgeM"] = round(
        modf['W%']/100 * modf['Win_MIDDLE'] - (100 - modf['W%'])/100, 2)
    modf['Win_FINAL'] = modf['Win_FINAL'] / modf['W']
    modf["edgeF"] = round(modf['W%']/100 * modf['Win_FINAL'] - (100 - modf['W%'])/100,
                                        2)
    f = [i for i in foo if i in modf.index]
    modf = modf.reindex(f)
    return modf

def generate_LO_stats(df):
    foo = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F17',
           'F18', 'F19']
    pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    n = df.groupby(["LO","Pl"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = df.groupby(["LO"], as_index=True).size().to_frame()
    k = df.groupby(["LO"], as_index=True)['Pl'].agg([maxwinstreaks, maxlosestreaks])
    j=df.groupby(["LO"], as_index=True)['Win_OPEN'].sum()
    i=df.groupby(["LO"], as_index=True)['Win_LATEST'].sum()
    h=df.groupby(["LO"], as_index=True)['Win_MIDDLE'].sum()
    g=df.groupby(["LO"], as_index=True)['Win_FINAL'].sum()
    lodf = pd.concat([n, m, k, j, i, h, g], axis=1)
    lodf.rename(columns={0:'TR', 1:"W", 2:"SHP", 3:"THP", 'maxwinstreaks':'LWS', 'maxlosestreaks':'LLS'}, inplace=True)
    lodf["W%"]=((lodf["W"]/lodf['TR'])*100).round(2)
    lodf["SHP%"]=((lodf["SHP"]/lodf['TR'])*100).round(2)
    lodf["THP%"]=((lodf["THP"]/lodf['TR'])*100).round(2)
    lodf["Plc%"]=(((lodf["W"]+lodf["SHP"]+lodf["THP"])/lodf['TR'])*100).round(2)
    lodf["Plc"]=lodf["W"]+lodf["SHP"]+lodf["THP"]
    lodf["BO"]=lodf["TR"]-lodf["Plc"]
    lodf["TR%"] = ((lodf["TR"] / len(df)) * 100).round(2)
    lodf['Win_OPEN'] = lodf['Win_OPEN'] / lodf['W']
    lodf["edgeO"] = round(lodf['W%']/100 * lodf['Win_OPEN'] - (100 - lodf['W%'])/100, 2)
    lodf['Win_LATEST'] = lodf['Win_LATEST'] / lodf['W']
    lodf["edgeL"] = round(
        lodf['W%']/100 * lodf['Win_LATEST'] - (100 - lodf['W%'])/100, 2)
    lodf['Win_MIDDLE'] = lodf['Win_MIDDLE'] / lodf['W']
    lodf["edgeM"] = round(
        lodf['W%']/100 * lodf['Win_MIDDLE'] - (100 - lodf['W%'])/100, 2)
    lodf['Win_FINAL'] = lodf['Win_FINAL'] / lodf['W']
    lodf["edgeF"] = round(lodf['W%']/100 * lodf['Win_FINAL'] - (100 - lodf['W%'])/100,
                                        2)
    f = [i for i in foo if i in lodf.index]
    lodf = lodf.reindex(f)
    return lodf

def generate_OO_stats(df):
    foo = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F17',
           'F18', 'F19']
    pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    n = df.groupby(["OO","Pl"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = df.groupby(["OO"], as_index=True).size().to_frame()
    k = df.groupby(["OO"], as_index=True)['Pl'].agg([maxwinstreaks, maxlosestreaks])
    j=df.groupby(["OO"], as_index=True)['Win_OPEN'].sum()
    i=df.groupby(["OO"], as_index=True)['Win_LATEST'].sum()
    h=df.groupby(["OO"], as_index=True)['Win_MIDDLE'].sum()
    g=df.groupby(["OO"], as_index=True)['Win_FINAL'].sum()
    oodf = pd.concat([n, m, k, j, i, h, g], axis=1)
    oodf.rename(columns={0:'TR', 1:"W", 2:"SHP", 3:"THP", 'maxwinstreaks':'LWS', 'maxlosestreaks':'LLS'}, inplace=True)
    oodf["W%"]=((oodf["W"]/oodf['TR'])*100).round(2)
    oodf["SHP%"]=((oodf["SHP"]/oodf['TR'])*100).round(2)
    oodf["THP%"]=((oodf["THP"]/oodf['TR'])*100).round(2)
    oodf["Plc%"]=(((oodf["W"]+oodf["SHP"]+oodf["THP"])/oodf['TR'])*100).round(2)
    oodf["Plc"]=oodf["W"]+oodf["SHP"]+oodf["THP"]
    oodf["BO"]=oodf["TR"]-oodf["Plc"]
    oodf["TR%"] = ((oodf["TR"] / len(df)) * 100).round(2)
    oodf['Win_OPEN'] = oodf['Win_OPEN'] / oodf['W']
    oodf["edgeO"] = round(oodf['W%']/100 * oodf['Win_OPEN'] - (100 - oodf['W%'])/100, 2)
    oodf['Win_LATEST'] = oodf['Win_LATEST'] / oodf['W']
    oodf["edgeL"] = round(
        oodf['W%']/100 * oodf['Win_LATEST'] - (100 - oodf['W%'])/100, 2)
    oodf['Win_MIDDLE'] = oodf['Win_MIDDLE'] / oodf['W']
    oodf["edgeM"] = round(
        oodf['W%']/100 * oodf['Win_MIDDLE'] - (100 - oodf['W%'])/100, 2)
    oodf['Win_FINAL'] = oodf['Win_FINAL'] / oodf['W']
    oodf["edgeF"] = round(oodf['W%']/100 * oodf['Win_FINAL'] - (100 - oodf['W%'])/100,
                                        2)
    f = [i for i in foo if i in oodf.index]
    oodf = oodf.reindex(f)
    return oodf

def generate_overall_scoreboard(overall_dff,dff):
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
            html.Td(html.Button("{}".format(len(overall_dff.loc[overall_dff['Pl'] == 1])), id='overall_first_button',
                                n_clicks=0),
                    id='overall_first_pos'),
            html.Td("{0:.2f}".format((len(overall_dff.loc[overall_dff['Pl'] == 1]) / len(overall_dff)) * 100),
                    id='overall_first_pct'),
            html.Td(html.Button("{}".format(len(overall_dff.loc[overall_dff['Pl'] == 2])), id='overall_second_button',
                                n_clicks=0),
                    id='overall_second_pos'),
            html.Td("{0:.2f}".format((len(overall_dff.loc[overall_dff['Pl'] == 2]) / len(overall_dff)) * 100),
                    id='overall_second_pct'),
            html.Td(html.Button("{}".format(len(overall_dff.loc[overall_dff['Pl'] == 3])), id='overall_third_button',
                                n_clicks=0),
                    id='overall_third_pos'),
            html.Td("{0:.2f}".format((len(overall_dff.loc[overall_dff['Pl'] == 3]) / len(overall_dff)) * 100),
                    id='overall_third_pct'),
            html.Td(html.Button("{}".format((len(overall_dff.loc[overall_dff['Pl'] == 1]) + len(
                overall_dff.loc[overall_dff['Pl'] == 2]) + len(overall_dff.loc[overall_dff['Pl'] == 3]))),
                                id='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(
                ((len(overall_dff.loc[overall_dff['Pl'] == 1]) + len(overall_dff.loc[overall_dff['Pl'] == 2]) + len(
                    overall_dff.loc[overall_dff['Pl'] == 3])) / len(overall_dff)) * 100), id='overall_place_pct'),
            html.Td(html.Button("{}".format(len(overall_dff) - (
                    len(overall_dff.loc[overall_dff['Pl'] == 1]) + len(overall_dff.loc[overall_dff['Pl'] == 2]) + len(
                overall_dff.loc[overall_dff['Pl'] == 3]))), id='overall_loss_button', n_clicks=0),
                    id='loss'),
            html.Td("{}".format(maxminstreaks(1, overall_dff)[0]), id='overall_longest_winning_streak_data'),
            html.Td("{}".format(maxminstreaks(1, overall_dff)[1]), id='overall_longest_losing_streak_data')
        ]),
        html.Tr([
            html.Td("Filtered", className='type_header'),
            html.Td("{}".format(len(dff))),
            html.Td("{}%".format(round(((len(dff)/len(overall_dff))*100),2))),
            html.Td(html.Button("{}".format(len(dff.loc[dff['Pl'] == 1])), id='first_button', n_clicks=0),
                    id='first_pos'),
            html.Td("{0:.2f}".format((len(dff.loc[dff['Pl'] == 1]) / len(dff)) * 100), id='first_pct'),
            html.Td(html.Button("{}".format(len(dff.loc[dff['Pl'] == 2])), id='second_button', n_clicks=0),
                    id='second_pos'),
            html.Td("{0:.2f}".format((len(dff.loc[dff['Pl'] == 2]) / len(dff)) * 100), id='second_pct'),
            html.Td(html.Button("{}".format(len(dff.loc[dff['Pl'] == 3])), id='third_button', n_clicks=0),
                    id='third_pos'),
            html.Td("{0:.2f}".format((len(dff.loc[dff['Pl'] == 3]) / len(dff)) * 100), id='third_pct'),
            html.Td(html.Button("{}".format((len(dff.loc[dff['Pl'] == 1]) + len(
                dff.loc[dff['Pl'] == 2]) + len(dff.loc[dff['Pl'] == 3]))), id='place_button', n_clicks=0),
                    id='place_pos'),
            html.Td("{0:.2f}".format(((len(dff.loc[dff['Pl'] == 1]) + len(dff.loc[dff['Pl'] == 2]) + len(
                dff.loc[dff['Pl'] == 3])) / len(dff)) * 100), id='place_pct'),
            html.Td(html.Button("{}".format(len(dff) - (
                    len(dff.loc[dff['Pl'] == 1]) + len(dff.loc[dff['Pl'] == 2]) + len(
                dff.loc[dff['Pl'] == 3]))), id='loss_button', n_clicks=0),
                    id='loss'),
            html.Td("{}".format(maxminstreaks(1, dff)[0]), id='longest_winning_streak_data'),
            html.Td("{}".format(maxminstreaks(1, dff)[1]), id='longest_losing_streak_data')
        ])
    ], id='overall_sb_table')]
    return Overall_SB

def safe_division(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0


def generate_overall_scoreboard_for_media(overall_media_race_stats,media_race_stats):
    overalltr = len(overall_media_race_stats)
    overalltrpct = 100.0

    overallwin = 0
    for i in ['F/W', 'S/W', 'T/W']:
        overallwin += overall_media_race_stats['Win'].value_counts().get(i, 0)
    overallwinpct = round((100 * safe_division(overallwin,
        len(overall_media_race_stats))), 2)
    overallshp = 0
    for i in ['F/2', 'S/2', 'T/2']:
        overallshp += overall_media_race_stats['SHP'].value_counts().get(i, 0)
    overallshppct = round((100 * safe_division(overallshp,
                                               len(overall_media_race_stats))), 2)
    overallthp = 0
    for i in ['F/3', 'S/3', 'T/3']:
        overallthp += overall_media_race_stats['THP'].value_counts().get(i, 0)
    overallthppct = round((100 * safe_division(overallthp,
                                               len(overall_media_race_stats))), 2)
    overallplc = overall_media_race_stats['Plc'].value_counts().get('W', 0)
    overallplcpct = round((100 * safe_division(overall_media_race_stats['Plc'].value_counts().get('W', 0),
                                               len(overall_media_race_stats))), 2)
    overallloss = overall_media_race_stats['Plc'].value_counts().get('L', 0)

    filteredtr = len(media_race_stats)
    filteredtrpct = round((100 * safe_division(len(media_race_stats), len(overall_media_race_stats))), 2)

    filteredwin = 0
    for i in ['F/W', 'S/W', 'T/W']:
        filteredwin += media_race_stats['Win'].value_counts().get(i, 0)
    filteredwinpct = round((100 * safe_division(filteredwin,
                                               len(media_race_stats))), 2)
    filteredshp = 0
    for i in ['F/2', 'S/2', 'T/2']:
        filteredshp += media_race_stats['SHP'].value_counts().get(i, 0)
    filteredshppct = round((100 * safe_division(filteredshp,
                                               len(media_race_stats))), 2)
    filteredthp = 0
    for i in ['F/3', 'S/3', 'T/3']:
        filteredthp += media_race_stats['THP'].value_counts().get(i, 0)
    filteredthppct = round((100 * safe_division(filteredthp,
                                                len(media_race_stats))), 2)
    filteredplc = media_race_stats['Plc'].value_counts().get('W', 0)
    filteredplcpct = round(
        (100 * safe_division(media_race_stats['Plc'].value_counts().get('W', 0), len(media_race_stats))), 2)
    filteredloss = media_race_stats['Plc'].value_counts().get('L', 0)

    firstdf = media_race_stats[media_race_stats['one'] == 1]
    seconddf = media_race_stats[media_race_stats['two'] == 1]
    thirddf = media_race_stats[media_race_stats['three'] == 1]

    firsttr = len(firstdf)
    firsttrpct = round((100 * safe_division(len(firstdf), len(media_race_stats))), 2)
    firstwin = firstdf['firstplc'].value_counts().get('W', 0)
    firstwinpct = round((100 * safe_division(firstdf['firstplc'].value_counts().get('W', 0), len(firstdf))), 2)
    firstshp = firstdf['firstplc'].value_counts().get('SHP', 0)
    firstshppct = round((100 * safe_division(firstdf['firstplc'].value_counts().get('SHP', 0), len(firstdf))), 2)
    firstthp = firstdf['firstplc'].value_counts().get('THP', 0)
    firstthppct = round((100 * safe_division(firstdf['firstplc'].value_counts().get('THP', 0), len(firstdf))), 2)
    firstplc = 0
    for i in ['W', 'SHP', 'THP']:
        firstplc += firstdf['firstplc'].value_counts().get(i, 0)
    firstplcpct = round(
        (100 * safe_division(firstplc, len(firstdf))), 2)
    firstloss = firstdf['firstplc'].value_counts().get('L', 0)

    secondtr = len(seconddf)
    secondtrpct = round((100 * safe_division(len(seconddf), len(media_race_stats))), 2)
    secondwin = seconddf['secondplc'].value_counts().get('W', 0)
    secondwinpct = round((100 * safe_division(seconddf['secondplc'].value_counts().get('W', 0), len(seconddf))), 2)
    secondshp = seconddf['secondplc'].value_counts().get('SHP', 0)
    secondshppct = round((100 * safe_division(seconddf['secondplc'].value_counts().get('SHP', 0), len(seconddf))), 2)
    secondthp = seconddf['secondplc'].value_counts().get('THP', 0)
    secondthppct = round((100 * safe_division(seconddf['secondplc'].value_counts().get('THP', 0), len(seconddf))), 2)
    secondplc = 0
    for i in ['W', 'SHP', 'THP']:
        secondplc += seconddf['secondplc'].value_counts().get(i, 0)
    secondplcpct = round(
        (100 * safe_division(secondplc, len(seconddf))), 2)
    secondloss = seconddf['secondplc'].value_counts().get('L', 0)

    thirdtr = len(thirddf)
    thirdtrpct = round((100 * safe_division(len(thirddf), len(media_race_stats))), 2)
    thirdwin = thirddf['thirdplc'].value_counts().get('W', 0)
    thirdwinpct = round((100 * safe_division(thirddf['thirdplc'].value_counts().get('W', 0), len(thirddf))), 2)
    thirdshp = thirddf['thirdplc'].value_counts().get('SHP', 0)
    thirdshppct = round((100 * safe_division(thirddf['thirdplc'].value_counts().get('SHP', 0), len(thirddf))), 2)
    thirdthp = thirddf['thirdplc'].value_counts().get('THP', 0)
    thirdthppct = round((100 * safe_division(thirddf['thirdplc'].value_counts().get('THP', 0), len(thirddf))), 2)
    thirdplc = 0
    for i in ['W', 'SHP', 'THP']:
        thirdplc += thirddf['thirdplc'].value_counts().get(i, 0)
    thirdplcpct = round(
        (100 * safe_division(thirdplc, len(thirddf))), 2)
    thirdloss = thirddf['thirdplc'].value_counts().get('L', 0)


    try:
        Overall_media_SB = [html.Div(id='selected_filters'),
                      html.Hr(className='othr'),
                      html.Table([
            html.Thead([
                html.Tr([
                    html.Th("", id='overall_header'),
                    html.Th("TR", id='overall_TR_header'),
                    html.Th("TR%", id='overall_TRpct_header'),
                    html.Th("W", id='overall_first_header'),
                    html.Th('W%', id='overall_firstpct_header'),
                    html.Th("SHP", id='overall_second_header'),
                    html.Th('SHP%', id='overall_secondpct_header'),
                    html.Th("THP", id='overall_third_header'),
                    html.Th('THP%', id='overall_thirdpct_header'),
                    html.Th('Plc', id='overall_place_header'),
                    html.Th('Plc%', id='overall_placepct_header'),
                    html.Th('BO', id='overall_loss_header'),
                    html.Th("LWS", id='overall_longest_winning_streak_header'),
                    html.Th("LLS", id='overall_longest_losing_streak_header')
                ])
            ]),
            html.Tr([
                html.Td("Overall", className='type_header'),
                html.Td("{}".format(overalltr)),
                html.Td("{}".format(overalltrpct)),
                html.Td(html.Button("{}".format(overallwin), id='overall_first_button',
                                    n_clicks=0),
                        id='overall_first_pos'),
                html.Td("{0:.2f}".format(overallwinpct),
                        id='overall_first_pct'),
                html.Td(html.Button("{}".format(overallshp), id='overall_second_button',
                                    n_clicks=0),
                        id='overall_second_pos'),
                html.Td("{0:.2f}".format(overallshppct),
                        id='overall_second_pct'),
                html.Td(html.Button("{}".format(overallthp), id='overall_third_button',
                                    n_clicks=0),
                        id='overall_third_pos'),
                html.Td("{0:.2f}".format(overallthppct),
                        id='overall_third_pct'),
                html.Td(html.Button("{}".format(overallplc),
                                    id='overall_place_button', n_clicks=0),
                        id='overall_place_pos'),
                html.Td("{0:.2f}".format(overallplcpct), id='overall_place_pct'),
                html.Td(html.Button("{}".format(overallloss), id='overall_loss_button', n_clicks=0),
                        id='loss'),
                html.Td("{}".format(maxminstreaks_media(['L'], overall_media_race_stats['Win'])[0]), id='overall_longest_winning_streak_data'),
                html.Td("{}".format(maxminstreaks_media(['L'], overall_media_race_stats['Win'])[1]), id='overall_longest_losing_streak_data')
            ]),
            html.Tr([
                html.Td("Filtered", className='type_header'),
                html.Td("{}".format(filteredtr)),
                html.Td("{}%".format(filteredtrpct)),
                html.Td(html.Button("{}".format(filteredwin), id='first_button', n_clicks=0),
                        id='first_pos'),
                html.Td("{0:.2f}".format(filteredwinpct), id='first_pct'),
                html.Td(html.Button("{}".format(filteredshp), id='second_button', n_clicks=0),
                        id='second_pos'),
                html.Td("{0:.2f}".format(filteredshppct), id='second_pct'),
                html.Td(html.Button("{}".format(filteredthp), id='third_button', n_clicks=0),
                        id='third_pos'),
                html.Td("{0:.2f}".format(filteredthppct), id='third_pct'),
                html.Td(html.Button("{}".format(filteredplc), id='place_button', n_clicks=0),
                        id='place_pos'),
                html.Td("{0:.2f}".format(filteredplcpct), id='place_pct'),
                html.Td(html.Button("{}".format(filteredloss), id='loss_button', n_clicks=0),
                        id='loss'),
                html.Td("{}".format(maxminstreaks_media(['L'], media_race_stats['Win'])[0]), id='longest_winning_streak_data'),
                html.Td("{}".format(maxminstreaks_media(['L'], media_race_stats['Win'])[1]), id='longest_losing_streak_data')
            ]),
            html.Tr([
                html.Td("First", className='type_header'),
                html.Td("{}".format(firsttr)),
                html.Td("{}%".format(firsttrpct)),
                html.Td(html.Button("{}".format(firstwin), id='first_first_button', n_clicks=0),
                        id='first_first_pos'),
                html.Td("{0:.2f}".format(firstwinpct), id='first_first_pct'),
                html.Td(html.Button("{}".format(firstshp), id='first_second_button', n_clicks=0),
                        id='first_second_pos'),
                html.Td("{0:.2f}".format(firstshppct), id='first_second_pct'),
                html.Td(html.Button("{}".format(firstthp), id='first_third_button', n_clicks=0),
                        id='first_third_pos'),
                html.Td("{0:.2f}".format(firstthppct), id='first_third_pct'),
                html.Td(html.Button("{}".format(firstplc), id='first_place_button', n_clicks=0),
                        id='first_place_pos'),
                html.Td("{0:.2f}".format(firstplcpct), id='first_place_pct'),
                html.Td(html.Button("{}".format(firstloss), id='first_loss_button', n_clicks=0),
                        id='first_loss'),
                html.Td("{}".format(maxminstreaks_media(['L', 'SHP', 'THP'], firstdf['firstplc'])[0]), id='first_longest_winning_streak_data'),
                html.Td("{}".format(maxminstreaks_media(['L', 'SHP', 'THP'], firstdf['firstplc'])[1]), id='first_longest_losing_streak_data')
            ]),
            html.Tr([
                html.Td("Second", className='type_header'),
                html.Td("{}".format(secondtr)),
                html.Td("{}%".format(secondtrpct)),
                html.Td(html.Button("{}".format(secondwin), id='second_first_button', n_clicks=0),
                        id='second_first_pos'),
                html.Td("{0:.2f}".format(secondwinpct), id='second_first_pct'),
                html.Td(html.Button("{}".format(secondshp), id='second_second_button', n_clicks=0),
                        id='second_second_pos'),
                html.Td("{0:.2f}".format(secondshppct), id='second_second_pct'),
                html.Td(html.Button("{}".format(secondthp), id='second_third_button', n_clicks=0),
                        id='second_third_pos'),
                html.Td("{0:.2f}".format(secondthppct), id='second_third_pct'),
                html.Td(html.Button("{}".format(secondplc), id='second_place_button', n_clicks=0),
                        id='second_place_pos'),
                html.Td("{0:.2f}".format(secondplcpct), id='place_pct'),
                html.Td(html.Button("{}".format(secondloss), id='second_loss_button', n_clicks=0),
                        id='second_loss'),
                html.Td("{}".format(maxminstreaks_media(['L', 'SHP', 'THP'], seconddf['secondplc'])[0]), id='second_longest_winning_streak_data'),
                html.Td("{}".format(maxminstreaks_media(['L', 'SHP', 'THP'], seconddf['secondplc'])[1]), id='second_longest_losing_streak_data')
            ]),
            html.Tr([
                html.Td("Third", className='type_header'),
                html.Td("{}".format(thirdtr)),
                html.Td("{}%".format(thirdtrpct)),
                html.Td(html.Button("{}".format(thirdwin), id='third_first_button', n_clicks=0),
                        id='third_first_pos'),
                html.Td("{0:.2f}".format(thirdwinpct), id='third_first_pct'),
                html.Td(html.Button("{}".format(thirdshp), id='third_second_button', n_clicks=0),
                        id='third_second_pos'),
                html.Td("{0:.2f}".format(thirdshppct), id='third_second_pct'),
                html.Td(html.Button("{}".format(thirdthp), id='third_third_button', n_clicks=0),
                        id='third_third_pos'),
                html.Td("{0:.2f}".format(thirdthppct), id='third_third_pct'),
                html.Td(html.Button("{}".format(thirdplc), id='third_place_button', n_clicks=0),
                        id='third_place_pos'),
                html.Td("{0:.2f}".format(thirdplcpct), id='third_place_pct'),
                html.Td(html.Button("{}".format(thirdloss), id='third_loss_button', n_clicks=0),
                        id='third_loss'),
                html.Td("{}".format(maxminstreaks_media(['L', 'SHP', 'THP'], thirddf['thirdplc'])[0]), id='third_longest_winning_streak_data'),
                html.Td("{}".format(maxminstreaks_media(['L', 'SHP', 'THP'], thirddf['thirdplc'])[1]), id='third_longest_losing_streak_data')
            ])
        ], id='overall_sb_table')]
        return Overall_media_SB
    except:
        Overall_media_SB = [html.Div(id='selected_filters'),
                            html.Hr(className='othr'),
                            html.Table([
                                html.Thead([
                                    html.Tr([
                                        html.Th("", id='overall_header'),
                                        html.Th("TR", id='overall_TR_header'),
                                        html.Th("TR%", id='overall_TRpct_header'),
                                        html.Th("F->W", id='overall_first_header'),
                                        html.Th('F->W%', id='overall_firstpct_header'),
                                        html.Th("S->W", id='overall_second_header'),
                                        html.Th('S->W%', id='overall_secondpct_header'),
                                        html.Th("T->W", id='overall_third_header'),
                                        html.Th('T->W%', id='overall_thirdpct_header'),
                                        html.Th('Any->W', id='overall_place_header'),
                                        html.Th('Any->W%', id='overall_placepct_header'),
                                        html.Th('Fail', id='overall_loss_header'),
                                        html.Th("LWS(F)", id='overall_longest_winning_streak_header'),
                                        html.Th("LLS(F)", id='overall_longest_losing_streak_header')
                                    ])
                                ]),
                                html.Tr([
                                    html.Td("Overall", className='type_header'),
                                    html.Td("{}".format(len(overall_media_race_stats))),
                                    html.Td("0%"),
                                    html.Td(html.Button(
                                        "{}".format(overall_media_race_stats['final'].value_counts().get('F/W', 0)),
                                        id='overall_first_button',
                                        n_clicks=0),
                                            id='overall_first_pos'),
                                    html.Td("{0:.2f}".format(round(
                                        overall_media_race_stats['final'].value_counts(normalize=True).get('F/W',
                                                                                                           0) * 100,
                                        2)),
                                            id='overall_first_pct'),
                                    html.Td(html.Button(
                                        "{}".format(overall_media_race_stats['final'].value_counts().get('S/W', 0)),
                                        id='overall_second_button',
                                        n_clicks=0),
                                            id='overall_second_pos'),
                                    html.Td("{0:.2f}".format(round(
                                        overall_media_race_stats['final'].value_counts(normalize=True).get('S/W',
                                                                                                           0) * 100,
                                        2)),
                                            id='overall_second_pct'),
                                    html.Td(html.Button(
                                        "{}".format(overall_media_race_stats['final'].value_counts().get('T/W', 0)),
                                        id='overall_third_button',
                                        n_clicks=0),
                                            id='overall_third_pos'),
                                    html.Td("{0:.2f}".format(round(
                                        overall_media_race_stats['final'].value_counts(normalize=True).get('T/W',
                                                                                                           0) * 100,
                                        2)),
                                            id='overall_third_pct'),
                                    html.Td(html.Button("{}".format((overall_media_race_stats[
                                                                         'final'].value_counts().get('F/W', 0) +
                                                                     overall_media_race_stats[
                                                                         'final'].value_counts().get('S/W', 0) +
                                                                     overall_media_race_stats[
                                                                         'final'].value_counts().get('T/W', 0))),
                                                        id='overall_place_button', n_clicks=0),
                                            id='overall_place_pos'),
                                    html.Td("{0:.2f}".format(round(
                                        overall_media_race_stats['final'].value_counts(normalize=True).get('F/W',
                                                                                                           0) * 100 +
                                        overall_media_race_stats['final'].value_counts(normalize=True).get('S/W',
                                                                                                           0) * 100 +
                                        overall_media_race_stats['final'].value_counts(normalize=True).get('T/W',
                                                                                                           0) * 100,
                                        2)), id='overall_place_pct'),
                                    html.Td(html.Button(
                                        "{}".format(overall_media_race_stats['final'].value_counts().get('L', 0)),
                                        id='overall_loss_button', n_clicks=0),
                                            id='loss'),
                                    html.Td(
                                        "{}".format(maxminstreaks_media('F/W', overall_media_race_stats['final'])[0]),
                                        id='overall_longest_winning_streak_data'),
                                    html.Td(
                                        "{}".format(maxminstreaks_media('F/W', overall_media_race_stats['final'])[1]),
                                        id='overall_longest_losing_streak_data')
                                ]),
                                html.Tr([
                                    html.Td("Filtered", className='type_header'),
                                    html.Td("{}".format(len(media_race_stats))),
                                    html.Td("{}%".format(0)),
                                    html.Td(
                                        html.Button("{}".format(media_race_stats['final'].value_counts().get('F/W', 0)),
                                                    id='first_button', n_clicks=0),
                                        id='first_pos'),
                                    html.Td("{0:.2f}".format(round(
                                        media_race_stats['final'].value_counts(normalize=True).get('F/W', 0) * 100, 2)),
                                            id='first_pct'),
                                    html.Td(
                                        html.Button("{}".format(media_race_stats['final'].value_counts().get('S/W', 0)),
                                                    id='second_button', n_clicks=0),
                                        id='second_pos'),
                                    html.Td("{0:.2f}".format(round(
                                        media_race_stats['final'].value_counts(normalize=True).get('S/W', 0) * 100, 2)),
                                            id='second_pct'),
                                    html.Td(
                                        html.Button("{}".format(media_race_stats['final'].value_counts().get('T/W', 0)),
                                                    id='third_button', n_clicks=0),
                                        id='third_pos'),
                                    html.Td("{0:.2f}".format(round(
                                        media_race_stats['final'].value_counts(normalize=True).get('T/W', 0) * 100, 2)),
                                            id='third_pct'),
                                    html.Td(html.Button("{}".format((media_race_stats['final'].value_counts().get('F/W',
                                                                                                                  0) +
                                                                     media_race_stats['final'].value_counts().get('S/W',
                                                                                                                  0) +
                                                                     media_race_stats['final'].value_counts().get('T/W',
                                                                                                                  0))),
                                                        id='place_button', n_clicks=0),
                                            id='place_pos'),
                                    html.Td("{0:.2f}".format(round(
                                        media_race_stats['final'].value_counts(normalize=True).get('F/W', 0) * 100 +
                                        media_race_stats['final'].value_counts(normalize=True).get('S/W', 0) * 100 +
                                        media_race_stats['final'].value_counts(normalize=True).get('T/W', 0) * 100, 2)),
                                            id='place_pct'),
                                    html.Td(
                                        html.Button("{}".format(media_race_stats['final'].value_counts().get('L', 0)),
                                                    id='loss_button', n_clicks=0),
                                        id='loss'),
                                    html.Td("{}".format(maxminstreaks_media('F/W', media_race_stats['final'])[0]),
                                            id='longest_winning_streak_data'),
                                    html.Td("{}".format(maxminstreaks_media('F/W', media_race_stats['final'])[1]),
                                            id='longest_losing_streak_data')
                                ])
                            ], id='overall_sb_table')]
        return Overall_media_SB

def generate_scoreboard(dff):
    SB = [html.Table([
        html.Thead([
            html.Tr([
                html.Th("TR", id='TR_header'),
                html.Th("WIN", id='first_header'),
                html.Th('W%', id='firstpct_header'),
                html.Th("SHP", id='second_header'),
                html.Th('2%', id='secondpct_header'),
                html.Th("THP", id='third_header'),
                html.Th('3%', id='thirdpct_header'),
                html.Th('Plc', id='place_header'),
                html.Th('P%', id='placepct_header'),
                html.Th('BO', id='loss_header'),
                html.Th("LWS(WIN)", id='longest_winning_streak_header'),
                html.Th("LLS(WIN)", id='longest_losing_streak_header')
            ])
        ]),
        html.Tr([
            html.Td("{}".format(len(dff))),
            html.Td(html.Button("{}".format(len(dff.loc[dff['Pl'] == 1])), id='first_button', n_clicks=0),
                    id='first_pos'),
            html.Td("{0:.2f}".format((len(dff.loc[dff['Pl'] == 1]) / len(dff)) * 100), id='first_pct'),
            html.Td(html.Button("{}".format(len(dff.loc[dff['Pl'] == 2])), id='second_button', n_clicks=0),
                    id='second_pos'),
            html.Td("{0:.2f}".format((len(dff.loc[dff['Pl'] == 2]) / len(dff)) * 100), id='second_pct'),
            html.Td(html.Button("{}".format(len(dff.loc[dff['Pl'] == 3])), id='third_button', n_clicks=0),
                    id='third_pos'),
            html.Td("{0:.2f}".format((len(dff.loc[dff['Pl'] == 3]) / len(dff)) * 100), id='third_pct'),
            html.Td(html.Button("{}".format((len(dff.loc[dff['Pl'] == 1]) + len(
                dff.loc[dff['Pl'] == 2]) + len(dff.loc[dff['Pl'] == 3]))), id='place_button', n_clicks=0),
                    id='place_pos'),
            html.Td("{0:.2f}".format(((len(dff.loc[dff['Pl'] == 1]) + len(dff.loc[dff['Pl'] == 2]) + len(
                dff.loc[dff['Pl'] == 3])) / len(dff)) * 100), id='place_pct'),
            html.Td(html.Button("{}".format(len(dff) - (
                    len(dff.loc[dff['Pl'] == 1]) + len(dff.loc[dff['Pl'] == 2]) + len(
                dff.loc[dff['Pl'] == 3]))), id='loss_button', n_clicks=0),
                    id='loss'),
            html.Td("{}".format(maxminstreaks(1, dff)[0]), id='longest_winning_streak_data'),
            html.Td("{}".format(maxminstreaks(1, dff)[1]), id='longest_losing_streak_data')
        ])
    ])]
    return SB

def generate_favourite_table(dff):
    fav_table = [html.Table([
        html.Thead([
            html.Tr([
                html.Td(),
                html.Th("F1", id='F1_header', className='statboxcolheader'),
                html.Th("F2", id='F2_header', className='statboxcolheader'),
                html.Th('F3', id='F3_header', className='statboxcolheader'),
                html.Th("F4", id='F4_header', className='statboxcolheader'),
                html.Th('F5', id='F5_header', className='statboxcolheader'),
                html.Th("F6", id='F6_header', className='statboxcolheader'),
                html.Th('F7', id='F7_header', className='statboxcolheader'),
                html.Th('F8', id='F8_header', className='statboxcolheader'),
                html.Th('F9', id='F9_header', className='statboxcolheader'),
                html.Th('F10', id='F10_header', className='statboxcolheader'),
                html.Th("F11", id='F11_header', className='statboxcolheader'),
                html.Th("F12", id='F12_header', className='statboxcolheader'),
                html.Th("F13", id='F13_header', className='statboxcolheader'),
                html.Th("F14", id='F14_header', className='statboxcolheader'),
                html.Th("F15", id='F15_header', className='statboxcolheader'),
                html.Th("F16", id='F16_header', className='statboxcolheader')
            ])
        ]),
        html.Tr([
            html.Th("Opening Odds", id='OO_header', className='statboxrowheader'),
            html.Td("{}".format(table_counts(dff, "OO", "F1")), id="OOF1"),
            html.Td("{}".format(table_counts(dff, "OO", "F2")), id="OOF2"),
            html.Td("{}".format(table_counts(dff, "OO", "F3")), id="OOF3"),
            html.Td("{}".format(table_counts(dff, "OO", "F4")), id="OOF4"),
            html.Td("{}".format(table_counts(dff, "OO", "F5")), id="OOF5"),
            html.Td("{}".format(table_counts(dff, "OO", "F6")), id="OOF6"),
            html.Td("{}".format(table_counts(dff, "OO", "F7")), id="OOF7"),
            html.Td("{}".format(table_counts(dff, "OO", "F8")), id="OOF8"),
            html.Td("{}".format(table_counts(dff, "OO", "F9")), id="OOF9"),
            html.Td("{}".format(table_counts(dff, "OO", "F10")), id="OOF10"),
            html.Td("{}".format(table_counts(dff, "OO", "F11")), id="OOF11"),
            html.Td("{}".format(table_counts(dff, "OO", "F12")), id="OOF12"),
            html.Td("{}".format(table_counts(dff, "OO", "F13")), id="OOF13"),
            html.Td("{}".format(table_counts(dff, "OO", "F14")), id="OOF14"),
            html.Td("{}".format(table_counts(dff, "OO", "F15")), id="OOF15"),
            html.Td("{}".format(table_counts(dff, "OO", "F16")), id="OOF16")

        ]),
        html.Tr([
            html.Th("Latest Odds", id='LO_header', className='statboxrowheader'),
            html.Td("{}".format(table_counts(dff, "LO", "F1")), id="LOF1"),
            html.Td("{}".format(table_counts(dff, "LO", "F2")), id="LOF2"),
            html.Td("{}".format(table_counts(dff, "LO", "F3")), id="LOF3"),
            html.Td("{}".format(table_counts(dff, "LO", "F4")), id="LOF4"),
            html.Td("{}".format(table_counts(dff, "LO", "F5")), id="LOF5"),
            html.Td("{}".format(table_counts(dff, "LO", "F6")), id="LOF6"),
            html.Td("{}".format(table_counts(dff, "LO", "F7")), id="LOF7"),
            html.Td("{}".format(table_counts(dff, "LO", "F8")), id="LOF8"),
            html.Td("{}".format(table_counts(dff, "LO", "F9")), id="LOF9"),
            html.Td("{}".format(table_counts(dff, "LO", "F10")), id="LOF10"),
            html.Td("{}".format(table_counts(dff, "LO", "F11")), id="LOF11"),
            html.Td("{}".format(table_counts(dff, "LO", "F12")), id="LOF12"),
            html.Td("{}".format(table_counts(dff, "LO", "F13")), id="LOF13"),
            html.Td("{}".format(table_counts(dff, "LO", "F14")), id="LOF14"),
            html.Td("{}".format(table_counts(dff, "LO", "F15")), id="LOF15"),
            html.Td("{}".format(table_counts(dff, "LO", "F16")), id="LOF16")
        ]),
        html.Tr([
            html.Th("Middle Odds", id='MO_header', className='statboxrowheader'),
            html.Td("{}".format(table_counts(dff, "MO", "F1")), id="MOF1"),
            html.Td("{}".format(table_counts(dff, "MO", "F2")), id="MOF2"),
            html.Td("{}".format(table_counts(dff, "MO", "F3")), id="MOF3"),
            html.Td("{}".format(table_counts(dff, "MO", "F4")), id="MOF4"),
            html.Td("{}".format(table_counts(dff, "MO", "F5")), id="MOF5"),
            html.Td("{}".format(table_counts(dff, "MO", "F6")), id="MOF6"),
            html.Td("{}".format(table_counts(dff, "MO", "F7")), id="MOF7"),
            html.Td("{}".format(table_counts(dff, "MO", "F8")), id="MOF8"),
            html.Td("{}".format(table_counts(dff, "MO", "F9")), id="MOF9"),
            html.Td("{}".format(table_counts(dff, "MO", "F10")), id="MOF10"),
            html.Td("{}".format(table_counts(dff, "MO", "F11")), id="MOF11"),
            html.Td("{}".format(table_counts(dff, "MO", "F12")), id="MOF12"),
            html.Td("{}".format(table_counts(dff, "MO", "F13")), id="MOF13"),
            html.Td("{}".format(table_counts(dff, "MO", "F14")), id="MOF14"),
            html.Td("{}".format(table_counts(dff, "MO", "F15")), id="MOF15"),
            html.Td("{}".format(table_counts(dff, "MO", "F16")), id="MOF16")
        ]),
        html.Tr([
            html.Th("Final Odds", id='FO_header', className='statboxrowheader'),
            html.Td("{}".format(table_counts(dff, "FO", "F1")), id="FOF1"),
            html.Td("{}".format(table_counts(dff, "FO", "F2")), id="FOF2"),
            html.Td("{}".format(table_counts(dff, "FO", "F3")), id="FOF3"),
            html.Td("{}".format(table_counts(dff, "FO", "F4")), id="FOF4"),
            html.Td("{}".format(table_counts(dff, "FO", "F5")), id="FOF5"),
            html.Td("{}".format(table_counts(dff, "FO", "F6")), id="FOF6"),
            html.Td("{}".format(table_counts(dff, "FO", "F7")), id="FOF7"),
            html.Td("{}".format(table_counts(dff, "FO", "F8")), id="FOF8"),
            html.Td("{}".format(table_counts(dff, "FO", "F9")), id="FOF9"),
            html.Td("{}".format(table_counts(dff, "FO", "F10")), id="FOF10"),
            html.Td("{}".format(table_counts(dff, "FO", "F11")), id="FOF11"),
            html.Td("{}".format(table_counts(dff, "FO", "F12")), id="FOF12"),
            html.Td("{}".format(table_counts(dff, "FO", "F13")), id="FOF13"),
            html.Td("{}".format(table_counts(dff, "FO", "F14")), id="FOF14"),
            html.Td("{}".format(table_counts(dff, "FO", "F15")), id="FOF15"),
            html.Td("{}".format(table_counts(dff, "FO", "F16")), id="FOF16")
        ])
    ])]
    return fav_table

def generate_split_scoreboard(xdf,searchterm, title):
    if searchterm == 'Jockey':
        ttype = "Trainer"
    else:
        ttype = "Jockey"
    header = html.H3("{} stats".format(title),
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    split_sb = dash_table.DataTable(
                       # id='split_table',
                       columns=[{"name": i, "id": i} for i in xdf[['TR','TR%','W','W%','SHP','SHP%','THP','THP%','Plc','Plc%','BO','F1%','F2%','F3%','LWS','LLS','edgeO','edgeL','edgeM','edgeF']].loc[searchterm[0]].reset_index().columns],
                       data=xdf[['TR','TR%','W','W%','SHP','SHP%','THP','THP%','Plc','Plc%','BO','F1%','F2%','F3%','LWS','LLS','edgeO','edgeL','edgeM','edgeF']].loc[searchterm[0]].reset_index().to_dict('records'),
                       editable=False,
                       export_columns='all',
                       export_format='xlsx',
                       #filterable=False,
                       #filter_action="none",
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
                                   'filter_query': '{{W}} = {}'.format(xdf['W'].loc[searchterm[0]].max()),
                                   'column_id': 'W'
                               },
                               # 'border':'2px solid 2ECC71',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{W%}} = {}'.format(xdf['W%'].loc[searchterm[0]].max()),
                                   'column_id': 'W%'
                               },
                               # 'border':'2px solid 2ECC71',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{SHP}} = {}'.format(xdf['SHP'].loc[searchterm[0]].max()),
                                   'column_id': 'SHP'
                               },
                               # 'border': '2px solid e8e40e',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{SHP%}} = {}'.format(xdf['SHP%'].loc[searchterm[0]].max()),
                                   'column_id': 'SHP%'
                               },
                               # 'border': '2px solid e8e40e',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{THP}} = {}'.format(xdf['THP'].loc[searchterm[0]].max()),
                                   'column_id': 'THP'
                               },
                               # 'border': '2px solid 3498DB',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{THP%}} = {}'.format(xdf['THP%'].loc[searchterm[0]].max()),
                                   'column_id': 'THP%'
                               },
                               # 'border': '2px solid 3498DB',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{Plc}} = {}'.format(xdf['Plc'].loc[searchterm[0]].max()),
                                   'column_id': 'Plc'
                               },
                               # 'border': '2px solid white',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{Plc%}} = {}'.format(xdf['Plc%'].loc[searchterm[0]].max()),
                                   'column_id': 'Plc%'
                               },
                               # 'border': '2px solid white',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{F1%}} = {}'.format(xdf['F1%'].loc[searchterm[0]].max()),
                                   'column_id': 'F1%'
                               },
                               # 'border': '2px solid abcdef',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{F2%}} = {}'.format(xdf['F2%'].loc[searchterm[0]].max()),
                                   'column_id': 'F2%'
                               },
                               # 'border': '2px solid abcdef',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{F3%}} = {}'.format(xdf['F3%'].loc[searchterm[0]].max()),
                                   'column_id': 'F3%'
                               },
                               # 'border': '2px solid abcdef',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{LWS}} = {}'.format(xdf['LWS'].loc[searchterm[0]].max()),
                                   'column_id': 'LWS'
                               },
                               # 'border': '2px solid 2ECC71',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{LLS}} = {}'.format(xdf['LLS'].loc[searchterm[0]].max()),
                                   'column_id': 'LLS'
                               },
                               # 'border': '2px solid ff2200',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{edgeO}} = {}'.format(xdf['edgeO'].loc[searchterm[0]].max()),
                                   'column_id': 'edgeO'
                               },
                               # 'border': '2px solid ff2200',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{edgeL}} = {}'.format(xdf['edgeL'].loc[searchterm[0]].max()),
                                   'column_id': 'edgeL'
                               },
                               # 'border': '2px solid ff2200',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{edgeM}} = {}'.format(xdf['edgeM'].loc[searchterm[0]].max()),
                                   'column_id': 'edgeM'
                               },
                               # 'border': '2px solid ff2200',
                               'font-weight':'bold',
                               'font-size':18
                           },
                           {
                               'if': {
                                   'filter_query': '{{edgeF}} = {}'.format(xdf['edgeF'].loc[searchterm[0]].max()),
                                   'column_id': 'edgeF'
                               },
                               # 'border': '2px solid ff2200',
                               'font-weight':'bold',
                               'font-size':18
                           }
                       ],
                       style_filter={
                           'textAlign': 'center'
                       },
                       style_as_list_view=True
                   )
    return header, split_sb


def generate_split_scoreboard_media(xdf,ttype):
    header = html.H3("{} stats".format(ttype),
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    split_sb = dash_table.DataTable(
        id=ttype+'table',
        columns=[{"name": i, "id": i} for i in xdf[
            ['TR', 'TR%', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS',
             'LLS']].reset_index().columns],
        data=xdf[['TR', 'TR%', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS',
                  'LLS']].reset_index().to_dict('records'),
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

    return header, split_sb


def generate_favourite_scoreboard(xdf):
    header = html.H3("F1-F16 performance",
            style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                   'font-family': 'Arial Black'})
    fav_sb =    dash_table.DataTable(
                   # id='table',
                   columns=[{"name": i, "id": i} for i in xdf[['TR','TR%','W','W%','SHP','SHP%','THP','THP%','Plc','Plc%','BO','LWS','LLS','edgeO','edgeL','edgeM','edgeF']].reset_index().columns],
                   data=xdf[['TR','TR%','W','W%','SHP','SHP%','THP','THP%','Plc','Plc%','BO','LWS','LLS','edgeO','edgeL','edgeM','edgeF']].reset_index().to_dict('records'),
                   editable=False,
                   export_columns='all',
                   export_format='xlsx',
                   #filterable=False,
                   #filter_action="none",
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
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{W%}} = {}'.format(xdf['W%'].max()),
                               'column_id': 'W%'
                           },
                           # 'border':'2px solid 2ECC71',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{SHP}} = {}'.format(xdf['SHP'].max()),
                               'column_id': 'SHP'
                           },
                           # 'border': '2px solid e8e40e',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{SHP%}} = {}'.format(xdf['SHP%'].max()),
                               'column_id': 'SHP%'
                           },
                           # 'border': '2px solid e8e40e',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{THP}} = {}'.format(xdf['THP'].max()),
                               'column_id': 'THP'
                           },
                           # 'border': '2px solid 3498DB',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{THP%}} = {}'.format(xdf['THP%'].max()),
                               'column_id': 'THP%'
                           },
                           # 'border': '2px solid 3498DB',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{Plc}} = {}'.format(xdf['Plc'].max()),
                               'column_id': 'Plc'
                           },
                           # 'border': '2px solid white',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{Plc%}} = {}'.format(xdf['Plc%'].max()),
                               'column_id': 'Plc%'
                           },
                           # 'border': '2px solid white',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{LWS}} = {}'.format(xdf['LWS'].max()),
                               'column_id': 'LWS'
                           },
                           # 'border': '2px solid 2ECC71',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{LLS}} = {}'.format(xdf['LLS'].max()),
                               'column_id': 'LLS'
                           },
                           # 'border': '2px solid ff2200',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{BO}} = {}'.format(xdf['BO'].max()),
                               'column_id': 'BO'
                           },
                           # 'border': '2px solid ff2200',
                           'font-weight':'bold',
                           'font-size':18
                       }
                   ],
                   style_filter={
                       'textAlign': 'center'
                   },
                   style_as_list_view=True
               )

    return header, fav_sb

def generate_additional_scoreboards(xdf, type):
    header = html.H3("{} stats".format(type),
            style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                   'font-family': 'Arial Black'})
    fav_sb =    dash_table.DataTable(
                   # id='table',
                   columns=[{"name": i, "id": i} for i in xdf[['TR','TR%','W','W%','SHP','SHP%','THP','THP%','Plc','Plc%','BO','F1%','F2%','F3%','LWS','LLS','edgeO','edgeL','edgeM','edgeF']].reset_index().columns],
                   data=xdf[['TR','TR%','W','W%','SHP','SHP%','THP','THP%','Plc','Plc%','BO','F1%','F2%','F3%','LWS','LLS','edgeO','edgeL','edgeM','edgeF']].reset_index().to_dict('records'),
                   editable=False,
                   export_columns='all',
                   export_format='xlsx',
                   #filterable=False,
                   #filter_action="none",
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
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{W%}} = {}'.format(xdf['W%'].max()),
                               'column_id': 'W%'
                           },
                           # 'border':'2px solid 2ECC71',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{SHP}} = {}'.format(xdf['SHP'].max()),
                               'column_id': 'SHP'
                           },
                           # 'border': '2px solid e8e40e',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{SHP%}} = {}'.format(xdf['SHP%'].max()),
                               'column_id': 'SHP%'
                           },
                           # 'border': '2px solid e8e40e',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{THP}} = {}'.format(xdf['THP'].max()),
                               'column_id': 'THP'
                           },
                           # 'border': '2px solid 3498DB',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{THP%}} = {}'.format(xdf['THP%'].max()),
                               'column_id': 'THP%'
                           },
                           # 'border': '2px solid 3498DB',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{Plc}} = {}'.format(xdf['Plc'].max()),
                               'column_id': 'Plc'
                           },
                           # 'border': '2px solid white',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{Plc%}} = {}'.format(xdf['Plc%'].max()),
                               'column_id': 'Plc%'
                           },
                           # 'border': '2px solid white',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{LWS}} = {}'.format(xdf['LWS'].max()),
                               'column_id': 'LWS'
                           },
                           # 'border': '2px solid 2ECC71',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{LLS}} = {}'.format(xdf['LLS'].max()),
                               'column_id': 'LLS'
                           },
                           # 'border': '2px solid ff2200',
                           'font-weight':'bold',
                           'font-size':18
                       },
                       {
                           'if': {
                               'filter_query': '{{BO}} = {}'.format(xdf['BO'].max()),
                               'column_id': 'BO'
                           },
                           # 'border': '2px solid ff2200',
                           'font-weight':'bold',
                           'font-size':18
                       }
                   ],
                   style_filter={
                       'textAlign': 'center'
                   },
                   style_as_list_view=True
               )

    return header, fav_sb

def generate_favourite_detailed_table(dff,FO_stats):
    split_by_type = 'Jockey'
    header = html.H3("F1-F16 performance(detailed)",
            style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                   'font-family': 'Arial Black'})
    fav_sb = html.Table([
        html.Thead([
            html.Tr([
                html.Th(""),
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
                html.Th("LWS", id='overall_longest_winning_streak_header'),
                html.Th("LLS", id='overall_longest_losing_streak_header')
            ])
        ], className='fav_sb_header'),
        html.Tr([
            html.Td("F0", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F0',0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F0', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F0', 0)), id='F0_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F0',0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F0', 0)), id='F0_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F0',0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F0', 0)), id='F0_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F0',0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F0', 0)), id='F0_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F0',0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F0', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F0',0)), id='F0_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F0',0)), id='F0_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F0'])[0]), className='table_streak'),
                html.Div(html.Ul(id='F0_form', className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F0']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F0']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F0'],split_by_type,1),className='fav_table_split_space', id='F0_split')
            ],
                colSpan=14,
                style={'padding':'0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F1", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F1', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F1', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F1', 0)), id='F1_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F1', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F1', 0)), id='F1_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F1', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F1', 0)),  id='F1_third_button',className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F1', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F1', 0)), id='F1_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F1', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F1', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F1', 0)), id='F1_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F1', 0)), id='F1_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F1'])[0]), className='table_streak'),
                html.Div(html.Ul(id='F1_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F1']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F1']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F1'],split_by_type,1),className='fav_table_split_space', id='F1_split')
            ],
                colSpan=14,
                style={'padding':'0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F2", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F2', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F2', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F2', 0)), id='F2_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F2', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F2', 0)), id='F2_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F2', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F2', 0)), id='F2_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F2', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F2', 0)), id='F2_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F2', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F2', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F2', 0)), id='F2_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F2', 0)), id='F2_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F2'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F2_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F2']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F2']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F2'],split_by_type,1),className='fav_table_split_space', id='F2_split')
            ],
                colSpan=14,
                style={'padding':'0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F3", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F3', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F3', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F3', 0)), id='F3_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F3', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F3', 0)), id='F3_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F3', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F3', 0)), id='F3_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F3', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F3', 0)), id='F3_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F3', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F3', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F3', 0)),  id='F3_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F3', 0)), id='F3_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F3'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F3_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F3']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F3']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F3'],split_by_type,1),className='fav_table_split_space', id='F3_split')
            ],
                colSpan=14,
                style={'padding':'0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F4", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F4', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F4', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F4', 0)), id='F4_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F4', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F4', 0)), id='F4_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F4', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F4', 0)), id='F4_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F4', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F4', 0)), id='F4_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F4', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F4', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F4', 0)),  id='F4_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F4', 0)), id='F4_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F4'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F4_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F4']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F4']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F4'],split_by_type,1),className='fav_table_split_space', id='F4_split')
            ],
                colSpan=14,
                style={'padding': '0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F5", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F5', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F5', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F5', 0)), id='F5_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F5', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F5', 0)), id='F5_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F5', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F5', 0)), id='F5_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F5', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F5', 0)), id='F5_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F5', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F5', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F5', 0)),  id='F5_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F5', 0)), id='F5_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F5'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F5_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F5']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F5']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F5'],split_by_type,1),className='fav_table_split_space', id='F5_split')
            ],
                colSpan=14,
                style={'padding': '0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F6", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F6', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F6', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F6', 0)), id='F6_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F6', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F6', 0)), id='F6_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F6', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F6', 0)), id='F6_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F6', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F6', 0)), id='F6_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F6', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F6', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F6', 0)),  id='F6_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F6', 0)), id='F6_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F6'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F6_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F6']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F6']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F6'],split_by_type,1),className='fav_table_split_space', id='F6_split')
            ],
                colSpan=14,
                style={'padding': '0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F7", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F7', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F7', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F7', 0)), id='F7_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F7', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F7', 0)), id='F7_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F7', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F7', 0)), id='F7_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F7', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F7', 0)), id='F7_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F7', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F7', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F7', 0)),  id='F7_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F7', 0)), id='F7_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F7'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F7_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F7']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F7']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F7'],split_by_type,1),className='fav_table_split_space', id='F7_split')
            ],
                colSpan=14,
                style={'padding': '0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F8", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F8', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F8', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F8', 0)), id='F8_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F8', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F8', 0)), id='F8_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F8', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F8', 0)), id='F8_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F8', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F8', 0)), id='F8_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F8', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F8', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F8', 0)),  id='F8_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F8', 0)), id='F8_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F8'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F8_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F8']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F8']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F8'],split_by_type,1),className='fav_table_split_space', id='F8_split')
            ],
                colSpan=14,
                style={'padding': '0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F9", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F9', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F9', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F9', 0)), id='F9_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F9', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F9', 0)), id='F9_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F9', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F9', 0)), id='F9_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F9', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F9', 0)), id='F9_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F9', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F9', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F9', 0)),  id='F9_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F9', 0)), id='F9_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F9'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F9_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F9']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F9']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F9'],split_by_type,1),className='fav_table_split_space', id='F9_split')
            ],
                colSpan=14,
                style={'padding': '0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F10", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F10', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F10', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F10', 0)), id='F10_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F10', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F10', 0)), id='F10_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F10', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F10', 0)), id='F10_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F10', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F10', 0)), id='F10_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F10', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F10', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F10', 0)),  id='F10_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F10', 0)), id='F10_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F10'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F10_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F10']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F10']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F10'],split_by_type,1),className='fav_table_split_space', id='F10_split')
            ],
                colSpan=14,
                style={'padding': '0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F11", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F11', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F11', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F11', 0)), id='F11_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F11', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F11', 0)), id='F11_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F11', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F11', 0)), id='F11_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F11', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F11', 0)), id='F11_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F11', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F11', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F11', 0)),  id='F11_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F11', 0)), id='F11_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F11'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F11_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F11']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F11']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F11'],split_by_type,1),className='fav_table_split_space', id='F11_split')
            ],
                colSpan=14,
                style={'padding': '0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F12", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F12', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F12', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F12', 0)), id='F12_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F12', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F12', 0)), id='F12_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F12', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F12', 0)), id='F12_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F12', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F12', 0)), id='F12_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F1', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F12', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F12', 0)),  id='F12_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F12', 0)), id='F12_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F12'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F12_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F12']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F12']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F12'],split_by_type,1),className='fav_table_split_space', id='F12_split')
            ],
                colSpan=14,
                style={'padding': '0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F13", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F13', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F13', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F13', 0)), id='F13_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F13', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F13', 0)), id='F13_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F13', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F13', 0)), id='F13_third_button',className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F13', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F13', 0)), id='F13_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F13', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F13', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F13', 0)),  id='F13_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F13', 0)), id='F13_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F13'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F13_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F13']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F13']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F13'],split_by_type,1),className='fav_table_split_space', id='F13_split')
            ],
                colSpan=14,
                style={'padding': '0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F14", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F14', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F14', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F14', 0)), id='F14_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F14', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F14', 0)), id='F14_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F14', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F14', 0)), id='F14_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F14', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F14', 0)), id='F14_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F14', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F14', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F14', 0)),  id='F14_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F14', 0)), id='F14_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F14'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F14_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F14']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F14']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F14'],split_by_type,1),className='fav_table_split_space', id='F14_split')
            ],
                colSpan=14,
                style={'padding': '0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F15", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F15', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F15', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F15', 0)), id='F15_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F15', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F15', 0)), id='F15_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F15', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F15', 0)), id='F15_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F15', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F15', 0)), id='F15_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F15', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F15', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F15', 0)),  id='F15_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F15', 0)), id='F15_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F15'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F15_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F15']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F15']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F15'],split_by_type,1),className='fav_table_split_space', id='F15_split')
            ],
                colSpan=14,
                style={'padding': '0px'},
                className='fgcontainer'
            )
        ]),
        html.Tr([
            html.Td("F16", className='fav_no_header'),
            html.Td("{}".format(int(FO_stats['TR'].get('F16', 0)))),
            html.Td("{}".format(FO_stats['TR%'].get('F16', 0))),
            html.Td(html.Button("{}".format(FO_stats['W'].get('F16', 0)), id='F16_first_button', className='overall_first_button',
                                n_clicks=0),
                    className='overall_first_pos'),
            html.Td("{0:.2f}".format(FO_stats['W%'].get('F16', 0)),
                    className='overall_first_pct'),
            html.Td(html.Button("{}".format(FO_stats['SHP'].get('F16', 0)), id='F16_second_button', className='overall_second_button',
                                n_clicks=0),
                    className='overall_second_pos'),
            html.Td("{0:.2f}".format(FO_stats['SHP%'].get('F16', 0)),
                    className='overall_second_pct'),
            html.Td(html.Button("{}".format(FO_stats['THP'].get('F16', 0)), id='F16_third_button', className='overall_third_button',
                                n_clicks=0),
                    className='overall_third_pos'),
            html.Td("{0:.2f}".format(FO_stats['THP%'].get('F16', 0)),
                    className='overall_third_pct'),
            html.Td(html.Button("{}".format(FO_stats['Plc'].get('F16', 0)), id='F16_place_button',
                                className='overall_place_button', n_clicks=0),
                    id='overall_place_pos'),
            html.Td("{0:.2f}".format(FO_stats['Plc%'].get('F16', 0)), className='overall_place_pct'),
            html.Td(html.Button("{}".format(FO_stats['BO'].get('F16', 0)), className='overall_loss_button', n_clicks=0),
                    className='loss'),
            html.Td("{}".format(FO_stats['LWS'].get('F16', 0)),  id='F16_lws', className='overall_longest_winning_streak_data'),
            html.Td("{}".format(FO_stats['LLS'].get('F16', 0)), id='F16_lls', className='overall_longest_winning_streak_data'),
        ]),
        html.Tr([
            html.Td(children=[
                html.Div("Streaks: {}".format(streak(dff.loc[dff['FO'] == 'F16'])[0]), className='table_streak'),
                html.Div(html.Ul( id='F16_form',className='fav_standings-table__form',
                        children=[generate_li_for_fav_sb(1, i, j, n) for n, (i, j) in
                                  enumerate(zip(dff.loc[dff['FO'] == 'F16']['Pl'][::-1],
                                                dff.loc[dff['FO'] == 'F16']['FINAL'][::-1]))]),
                        className='table_form'),
                html.Br(),
                html.Div(children=generate_fav_table_split(dff.loc[dff['FO'] == 'F16'],split_by_type,1),className='fav_table_split_space', id='F16_split')
            ],
                colSpan=14,
                style={'padding': '0px'},
                className='fgcontainer'
            )
        ])
    ], id='fav_sb_table')

    return header, fav_sb

def generate_media_stats(df):
    media_list = ['TurfBee', 'BOL', 'Dina Thanthi', 'Experts', 'Media Man',
       'S. Today', 'TRS', 'Telegraph', 'BNG', 'Telangana Today',
       'Telangana Times', 'Andhra Jyothi', 'Asian Age', 'Times of India',
       'The Hindu', 'Mumbai Mirror', 'Deccan Herald']
    dicts = defaultdict(dict)
    for m in media_list:
        race=0
        for i in ['F', 'S', 'T']:
            temp = df[df[m] == i]
            if len(temp) == 0:
                # print(m)
                continue
            if len(temp) > race:
                dicts[m]['TR'] = len(temp)
                race = len(temp)
            dicts[m][i] = round(100 * len(temp.loc[temp["Pl"] == 1]) / len(temp), 2)

    media_df = pd.DataFrame.from_dict(dicts, 'index')
    return media_df

# def generate_distance_stats(dff):
#     pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#
#     m = dff.groupby(['Distance'], as_index=True).size().to_frame()
#     n = dff.groupby(['Distance', 'Pl'], as_index=True).size().unstack(fill_value=0)
#     mis_col = [i for i in pl if i not in n.columns]
#     for col in mis_col:
#         n[col] = [0] * len(n)
#     n = n[pl]
#
#     dis = pd.concat([m, n], axis=1)
#     dis.rename(columns={0: 'TR', 1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'},
#                inplace=True)
#     dis["W%"] = ((dis["W"] / dis['TR']) * 100).round(2)
#     dis["SHP%"] = ((dis["SHP"] / dis['TR']) * 100).round(2)
#     dis["THP%"] = ((dis["THP"] / dis['TR']) * 100).round(2)
#     dis["Plc%"] = (((dis["W"] + dis["SHP"] + dis["THP"]) / dis['TR']) * 100).round(2)
#     dis["Plc"] = dis["W"] + dis["SHP"] + dis["THP"]
#     dis["BO"] = dis["TR"] - dis["Plc"]
#     return dis

def generate_distance_stats(dff):
    foo = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F17',
           'F18', 'F19']
    pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    n = dff.groupby(["Distance", "Pl"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = dff.groupby(["Distance"], as_index=True).size().to_frame()
    l = dff.groupby(["Distance", "FO"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in foo if i not in l.columns]
    for col in mis_col:
        l[col] = [0] * len(l)
    l = l[foo]
    k = dff.groupby(["Distance"], as_index=True)['Pl'].agg([maxwinstreaks, maxlosestreaks])
    j = dff.groupby(["Distance"], as_index=True)['Win_OPEN'].sum()
    i = dff.groupby(["Distance"], as_index=True)['Win_LATEST'].sum()
    h = dff.groupby(["Distance"], as_index=True)['Win_MIDDLE'].sum()
    g = dff.groupby(["Distance"], as_index=True)['Win_FINAL'].sum()
    dist_stats = pd.concat([n, m, l, k, j , i , h , g], axis=1)
    dist_stats.rename(columns={0: 'TR', 1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'},
                      inplace=True)
    dist_stats["W%"] = ((dist_stats["W"] / dist_stats['TR']) * 100).round(2)
    dist_stats["SHP%"] = ((dist_stats["SHP"] / dist_stats['TR']) * 100).round(2)
    dist_stats["THP%"] = ((dist_stats["THP"] / dist_stats['TR']) * 100).round(2)
    dist_stats["Plc%"] = (((dist_stats["W"] + dist_stats["SHP"] + dist_stats["THP"]) / dist_stats['TR']) * 100).round(2)
    dist_stats["F1%"] = ((dist_stats["F1"] / dist_stats['TR']) * 100).round(2)
    dist_stats["F2%"] = ((dist_stats["F2"] / dist_stats['TR']) * 100).round(2)
    dist_stats["F3%"] = ((dist_stats["F3"] / dist_stats['TR']) * 100).round(2)
    dist_stats["Plc"] = dist_stats["W"] + dist_stats["SHP"] + dist_stats["THP"]
    dist_stats["BO"] = dist_stats["TR"] - dist_stats["Plc"]
    dist_stats["TR%"] = ((dist_stats["TR"] / len(dff)) * 100).round(2)
    dist_stats['Win_OPEN'] = dist_stats['Win_OPEN'] / dist_stats['W']
    dist_stats["edgeO"] = round(dist_stats['W%']/100 * dist_stats['Win_OPEN'] - (100 - dist_stats['W%'])/100, 2)
    dist_stats['Win_LATEST'] = dist_stats['Win_LATEST'] / dist_stats['W']
    dist_stats["edgeL"] = round(
        dist_stats['W%']/100 * dist_stats['Win_LATEST'] - (100 - dist_stats['W%'])/100, 2)
    dist_stats['Win_MIDDLE'] = dist_stats['Win_MIDDLE'] / dist_stats['W']
    dist_stats["edgeM"] = round(
        dist_stats['W%']/100 * dist_stats['Win_MIDDLE'] - (100 - dist_stats['W%'])/100, 2)
    dist_stats['Win_FINAL'] = dist_stats['Win_FINAL'] / dist_stats['W']
    dist_stats["edgeF"] = round(dist_stats['W%']/100 * dist_stats['Win_FINAL'] - (100 - dist_stats['W%'])/100,
                                        2)
    return dist_stats

def generate_dr_stats(dff):
    foo = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F17',
           'F18', 'F19']
    pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    n = dff.groupby(["Dr", "Pl"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = dff.groupby(["Dr"], as_index=True).size().to_frame()
    l = dff.groupby(["Dr", "FO"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in foo if i not in l.columns]
    for col in mis_col:
        l[col] = [0] * len(l)
    l = l[foo]
    k = dff.groupby(["Dr"], as_index=True)['Pl'].agg([maxwinstreaks, maxlosestreaks])
    j = dff.groupby(["Dr"], as_index=True)['Win_OPEN'].sum()
    i = dff.groupby(["Dr"], as_index=True)['Win_LATEST'].sum()
    h = dff.groupby(["Dr"], as_index=True)['Win_MIDDLE'].sum()
    g = dff.groupby(["Dr"], as_index=True)['Win_FINAL'].sum()
    dr_stats = pd.concat([n, m, l, k, j , i , h , g], axis=1)
    dr_stats.rename(columns={0: 'TR', 1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'},
                    inplace=True)
    dr_stats["W%"] = ((dr_stats["W"] / dr_stats['TR']) * 100).round(2)
    dr_stats["SHP%"] = ((dr_stats["SHP"] / dr_stats['TR']) * 100).round(2)
    dr_stats["THP%"] = ((dr_stats["THP"] / dr_stats['TR']) * 100).round(2)
    dr_stats["Plc%"] = (((dr_stats["W"] + dr_stats["SHP"] + dr_stats["THP"]) / dr_stats['TR']) * 100).round(2)
    dr_stats["F1%"] = ((dr_stats["F1"] / dr_stats['TR']) * 100).round(2)
    dr_stats["F2%"] = ((dr_stats["F2"] / dr_stats['TR']) * 100).round(2)
    dr_stats["F3%"] = ((dr_stats["F3"] / dr_stats['TR']) * 100).round(2)
    dr_stats["Plc"] = dr_stats["W"] + dr_stats["SHP"] + dr_stats["THP"]
    dr_stats["BO"] = dr_stats["TR"] - dr_stats["Plc"]
    dr_stats["TR%"] = ((dr_stats["TR"] / len(dff)) * 100).round(2)
    dr_stats['Win_OPEN'] = dr_stats['Win_OPEN'] / dr_stats['W']
    dr_stats["edgeO"] = round(dr_stats['W%']/100 * dr_stats['Win_OPEN'] - (100 - dr_stats['W%'])/100, 2)
    dr_stats['Win_LATEST'] = dr_stats['Win_LATEST'] / dr_stats['W']
    dr_stats["edgeL"] = round(
        dr_stats['W%']/100 * dr_stats['Win_LATEST'] - (100 - dr_stats['W%'])/100, 2)
    dr_stats['Win_MIDDLE'] = dr_stats['Win_MIDDLE'] / dr_stats['W']
    dr_stats["edgeM"] = round(
        dr_stats['W%']/100 * dr_stats['Win_MIDDLE'] - (100 - dr_stats['W%'])/100, 2)
    dr_stats['Win_FINAL'] = dr_stats['Win_FINAL'] / dr_stats['W']
    dr_stats["edgeF"] = round(dr_stats['W%']/100 * dr_stats['Win_FINAL'] - (100 - dr_stats['W%'])/100,
                                        2)
    dr_stats.index = dr_stats.index.astype('int32')
    return dr_stats

def generate_age_stats(dff):
    foo = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F17',
           'F18', 'F19']
    pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    n = dff.groupby(["Age", "Pl"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = dff.groupby(["Age"], as_index=True).size().to_frame()
    l = dff.groupby(["Age", "FO"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in foo if i not in l.columns]
    for col in mis_col:
        l[col] = [0] * len(l)
    l = l[foo]
    k = dff.groupby(["Age"], as_index=True)['Pl'].agg([maxwinstreaks, maxlosestreaks])
    j = dff.groupby(["Age"], as_index=True)['Win_OPEN'].sum()
    i = dff.groupby(["Age"], as_index=True)['Win_LATEST'].sum()
    h = dff.groupby(["Age"], as_index=True)['Win_MIDDLE'].sum()
    g = dff.groupby(["Age"], as_index=True)['Win_FINAL'].sum()
    age_stats = pd.concat([n, m, l, k, j , i , h , g], axis=1)
    age_stats.rename(columns={0: 'TR', 1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'},
                     inplace=True)
    age_stats["W%"] = ((age_stats["W"] / age_stats['TR']) * 100).round(2)
    age_stats["SHP%"] = ((age_stats["SHP"] / age_stats['TR']) * 100).round(2)
    age_stats["THP%"] = ((age_stats["THP"] / age_stats['TR']) * 100).round(2)
    age_stats["Plc%"] = (((age_stats["W"] + age_stats["SHP"] + age_stats["THP"]) / age_stats['TR']) * 100).round(2)
    age_stats["F1%"] = ((age_stats["F1"] / age_stats['TR']) * 100).round(2)
    age_stats["F2%"] = ((age_stats["F2"] / age_stats['TR']) * 100).round(2)
    age_stats["F3%"] = ((age_stats["F3"] / age_stats['TR']) * 100).round(2)
    age_stats["Plc"] = age_stats["W"] + age_stats["SHP"] + age_stats["THP"]
    age_stats["BO"] = age_stats["TR"] - age_stats["Plc"]
    age_stats["TR%"] = ((age_stats["TR"] / len(dff)) * 100).round(2)
    age_stats['Win_OPEN'] = age_stats['Win_OPEN'] / age_stats['W']
    age_stats["edgeO"] = round(age_stats['W%']/100 * age_stats['Win_OPEN'] - (100 - age_stats['W%'])/100, 2)
    age_stats['Win_LATEST'] = age_stats['Win_LATEST'] / age_stats['W']
    age_stats["edgeL"] = round(
        age_stats['W%']/100 * age_stats['Win_LATEST'] - (100 - age_stats['W%'])/100, 2)
    age_stats['Win_MIDDLE'] = age_stats['Win_MIDDLE'] / age_stats['W']
    age_stats["edgeM"] = round(
        age_stats['W%']/100 * age_stats['Win_MIDDLE'] - (100 - age_stats['W%'])/100, 2)
    age_stats['Win_FINAL'] = age_stats['Win_FINAL'] / age_stats['W']
    age_stats["edgeF"] = round(age_stats['W%']/100 * age_stats['Win_FINAL'] - (100 - age_stats['W%'])/100,
                                        2)
    return age_stats

def generate_distance_boxes(dff, distance_stats):
    best = {1000: '2px solid #082255', 1100: '2px solid #082255', 1200: '2px solid #082255', 1400: '2px solid #082255',
            1600: '2px solid #082255', 1800: '2px solid #082255', 2000: '2px solid #082255', 2200: '2px solid #082255',
            2400: '2px solid #082255', 2800: '2px solid #082255', 3000: '2px solid #082255', 3200: '2px solid #082255',
            distance_stats['W%'].idxmax(): '2px solid #59ff00'}

    distance_boxes = [
        html.H3("Distance-wise performance",
                style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                       'font-family': 'Arial Black'}),
        html.Div([
            html.Div([html.H6("1000M", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(distance_stats['TR'].get(1000,0))), className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(distance_stats['W'].get(1000, '-')), className='foboxwin'),
                          html.Div("2: {}".format(distance_stats['SHP'].get(1000, '-')), className='foboxshp'),
                          html.Div("3: {}".format(distance_stats['THP'].get(1000, '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(distance_stats['Plc'].get(1000, '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(distance_stats['W%'].get(1000, '-')), className='foboxwin'),
                          html.Div("2%: {}".format(distance_stats['SHP%'].get(1000, '-')), className='foboxshp'),
                          html.Div("3%: {}".format(distance_stats['THP%'].get(1000, '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(distance_stats['Plc%'].get(1000, '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px', 'border':best[1000]}),
            html.Div([html.H6("1100M", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(distance_stats['TR'].get(1100, 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(distance_stats['W'].get(1100, '-')), className='foboxwin'),
                          html.Div("2: {}".format(distance_stats['SHP'].get(1100, '-')), className='foboxshp'),
                          html.Div("3: {}".format(distance_stats['THP'].get(1100, '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(distance_stats['Plc'].get(1100, '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(distance_stats['W%'].get(1100, '-')), className='foboxwin'),
                          html.Div("2%: {}".format(distance_stats['SHP%'].get(1100, '-')), className='foboxshp'),
                          html.Div("3%: {}".format(distance_stats['THP%'].get(1100, '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(distance_stats['Plc%'].get(1100, '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px', 'border':best[1100]}),
            html.Div([html.H6("1200M", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(distance_stats['TR'].get(1200, 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(distance_stats['W'].get(1200, '-')), className='foboxwin'),
                          html.Div("2: {}".format(distance_stats['SHP'].get(1200, '-')), className='foboxshp'),
                          html.Div("3: {}".format(distance_stats['THP'].get(1200, '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(distance_stats['Plc'].get(1200, '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(distance_stats['W%'].get(1200, '-')), className='foboxwin'),
                          html.Div("2%: {}".format(distance_stats['SHP%'].get(1200, '-')), className='foboxshp'),
                          html.Div("3%: {}".format(distance_stats['THP%'].get(1200, '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(distance_stats['Plc%'].get(1200, '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px', 'border':best[1200]}),
            html.Div([html.H6("1400M", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(distance_stats['TR'].get(1400, 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(distance_stats['W'].get(1400, '-')), className='foboxwin'),
                          html.Div("2: {}".format(distance_stats['SHP'].get(1400, '-')), className='foboxshp'),
                          html.Div("3: {}".format(distance_stats['THP'].get(1400, '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(distance_stats['Plc'].get(1400, '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(distance_stats['W%'].get(1400, '-')), className='foboxwin'),
                          html.Div("2%: {}".format(distance_stats['SHP%'].get(1400, '-')), className='foboxshp'),
                          html.Div("3%: {}".format(distance_stats['THP%'].get(1400, '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(distance_stats['Plc%'].get(1400, '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px', 'border':best[1400]}),
            html.Div([html.H6("1600M", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(distance_stats['TR'].get(1600, 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(distance_stats['W'].get(1600, '-')), className='foboxwin'),
                          html.Div("2: {}".format(distance_stats['SHP'].get(1600, '-')), className='foboxshp'),
                          html.Div("3: {}".format(distance_stats['THP'].get(1600, '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(distance_stats['Plc'].get(1600, '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(distance_stats['W%'].get(1600, '-')), className='foboxwin'),
                          html.Div("2%: {}".format(distance_stats['SHP%'].get(1600, '-')), className='foboxshp'),
                          html.Div("3%: {}".format(distance_stats['THP%'].get(1600, '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(distance_stats['Plc%'].get(1600, '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px', 'border':best[1600]}),
            html.Div([html.H6("1800M", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(distance_stats['TR'].get(1800, 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(distance_stats['W'].get(1800, '-')), className='foboxwin'),
                          html.Div("2: {}".format(distance_stats['SHP'].get(1800, '-')), className='foboxshp'),
                          html.Div("3: {}".format(distance_stats['THP'].get(1800, '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(distance_stats['Plc'].get(1800, '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(distance_stats['W%'].get(1800, '-')), className='foboxwin'),
                          html.Div("2%: {}".format(distance_stats['SHP%'].get(1800, '-')), className='foboxshp'),
                          html.Div("3%: {}".format(distance_stats['THP%'].get(1800, '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(distance_stats['Plc%'].get(1800, '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px', 'border':best[1800]})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'}),
        html.Div([
            html.Div([html.H6("2000M", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(distance_stats['TR'].get(2000, 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(distance_stats['W'].get(2000, '-')), className='foboxwin'),
                          html.Div("2: {}".format(distance_stats['SHP'].get(2000, '-')), className='foboxshp'),
                          html.Div("3: {}".format(distance_stats['THP'].get(2000, '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(distance_stats['Plc'].get(2000, '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(distance_stats['W%'].get(2000, '-')), className='foboxwin'),
                          html.Div("2%: {}".format(distance_stats['SHP%'].get(2000, '-')), className='foboxshp'),
                          html.Div("3%: {}".format(distance_stats['THP%'].get(2000, '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(distance_stats['Plc%'].get(2000, '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px', 'border':best[2000]}),
            html.Div([html.H6("2200M", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(distance_stats['TR'].get(2200, 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(distance_stats['W'].get(2200, '-')), className='foboxwin'),
                          html.Div("2: {}".format(distance_stats['SHP'].get(2200, '-')), className='foboxshp'),
                          html.Div("3: {}".format(distance_stats['THP'].get(2200, '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(distance_stats['Plc'].get(2200, '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(distance_stats['W%'].get(2200, '-')), className='foboxwin'),
                          html.Div("2%: {}".format(distance_stats['SHP%'].get(2200, '-')), className='foboxshp'),
                          html.Div("3%: {}".format(distance_stats['THP%'].get(2200, '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(distance_stats['Plc%'].get(2200, '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px', 'border':best[2200]}),
            html.Div([html.H6("2400M", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(distance_stats['TR'].get(2400, 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(distance_stats['W'].get(2400, '-')), className='foboxwin'),
                          html.Div("2: {}".format(distance_stats['SHP'].get(2400, '-')), className='foboxshp'),
                          html.Div("3: {}".format(distance_stats['THP'].get(2400, '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(distance_stats['Plc'].get(2400, '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(distance_stats['W%'].get(2400, '-')), className='foboxwin'),
                          html.Div("2%: {}".format(distance_stats['SHP%'].get(2400, '-')), className='foboxshp'),
                          html.Div("3%: {}".format(distance_stats['THP%'].get(2400, '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(distance_stats['Plc%'].get(2400, '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px', 'border':best[2400]}),
            html.Div([html.H6("2800M", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(distance_stats['TR'].get(2800, 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(distance_stats['W'].get(2800, '-')), className='foboxwin'),
                          html.Div("2: {}".format(distance_stats['SHP'].get(2800, '-')), className='foboxshp'),
                          html.Div("3: {}".format(distance_stats['THP'].get(2800, '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(distance_stats['Plc'].get(2800, '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(distance_stats['W%'].get(2800, '-')), className='foboxwin'),
                          html.Div("2%: {}".format(distance_stats['SHP%'].get(2800, '-')), className='foboxshp'),
                          html.Div("3%: {}".format(distance_stats['THP%'].get(2800, '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(distance_stats['Plc%'].get(2800, '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px', 'border':best[2800]}),
            html.Div([html.H6("3000M", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(distance_stats['TR'].get(3000, 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(distance_stats['W'].get(3000, '-')), className='foboxwin'),
                          html.Div("2: {}".format(distance_stats['SHP'].get(3000, '-')), className='foboxshp'),
                          html.Div("3: {}".format(distance_stats['THP'].get(3000, '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(distance_stats['Plc'].get(3000, '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(distance_stats['W%'].get(3000, '-')), className='foboxwin'),
                          html.Div("2%: {}".format(distance_stats['SHP%'].get(3000, '-')), className='foboxshp'),
                          html.Div("3%: {}".format(distance_stats['THP%'].get(3000, '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(distance_stats['Plc%'].get(3000, '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px', 'border':best[3000]}),
            html.Div([html.H6("3200M", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(distance_stats['TR'].get(3200, 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(distance_stats['W'].get(3200, '-')), className='foboxwin'),
                          html.Div("2: {}".format(distance_stats['SHP'].get(3200, '-')), className='foboxshp'),
                          html.Div("3: {}".format(distance_stats['THP'].get(3200, '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(distance_stats['Plc'].get(3200, '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(distance_stats['W%'].get(3200, '-')), className='foboxwin'),
                          html.Div("2%: {}".format(distance_stats['SHP%'].get(3200, '-')), className='foboxshp'),
                          html.Div("3%: {}".format(distance_stats['THP%'].get(3200, '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(distance_stats['Plc%'].get(3200, '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px', 'border':best[3200]})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'})
    ]
    return distance_boxes

def generate_single_distance_box(dist, distance_stats):
    box = html.Div([html.H6("{}M".format(dist), className='distanceboxheader'),
                      html.Div("Total Races: {}".format(int(distance_stats.loc[dist]['TR'])), className='statboxcontent'),
                      html.Div("W%: {}".format(distance_stats.loc[dist]['W%']), className='statboxnumcontent'),
                      html.Div("SHP%: {}".format(distance_stats.loc[dist]['SHP%']), className='statboxnumcontent'),
                      html.Div("THP%: {}".format(distance_stats.loc[dist]['THP%']), className='statboxnumcontent'),
                      html.Div("Plc%: {}".format(distance_stats.loc[dist]['Plc%']), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'})
    return box

def generate_distance_boxes_test(dff, distance_stats):
    distance_boxes = [
        html.H3("Distance-wise performance",
                style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                       'font-family': 'Arial Black'}),
        html.Div([generate_single_distance_box(i, distance_stats) for i in dff['Distance'].unique()[:int(len(dff['Distance'].unique())/2)]], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'}),
        html.Div([generate_single_distance_box(i, distance_stats) for i in dff['Distance'].unique()[int(len(dff['Distance'].unique())/2):]], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'})
        ]
    return distance_boxes

def generate_FO_boxes(dff, FO_stats):
    fo_boxes = [
        html.H3("F1-F16 performance",
                style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                       'font-family': 'Arial Black'}),
        html.Div([
            html.Div([html.H6("F1", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F1',0))), className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F1', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F1', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F1', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F1', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F1', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F1', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F1', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F1', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("F2", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F2', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F2', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F2', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F2', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F2', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F2', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F2', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F2', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F2', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("F3", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F3', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F3', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F3', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F3', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F3', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F3', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F3', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F3', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F3', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("F4", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F4', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F4', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F4', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F4', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F4', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F4', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F4', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F4', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F4', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("F5", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F5', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F5', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F5', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F5', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F5', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F5', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F5', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F5', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F5', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("F6", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F6', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F6', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F6', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F6', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F6', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F6', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F6', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F6', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F6', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'}),
        html.Div([
            html.Div([html.H6("F7", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F7', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F7', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F7', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F7', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F7', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F7', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F7', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F7', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F7', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("F8", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F8', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F8', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F8', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F8', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F8', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F8', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F8', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F8', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F8', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("F9", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F9', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F9', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F9', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F9', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F9', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F9', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F9', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F9', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F9', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("F10", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F10', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F10', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F10', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F10', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F10', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F10', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F10', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F10', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F10', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("F11", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F11', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F11', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F11', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F11', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F11', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F11', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F11', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F11', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F11', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("F12", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F12', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F12', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F12', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F12', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F12', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F12', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F12', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F12', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F12', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'}),
        html.Div([
            html.Div(className="mini_container", style={'width': '200px', 'background-color':'#061e44'}),
            html.Div([html.H6("F13", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F13', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F13', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F13', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F13', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F13', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F13', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F13', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F13', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F13', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("F14", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F14', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F14', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F14', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F14', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F14', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F14', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F14', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F14', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F14', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("F15", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F15', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F15', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F15', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F15', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F15', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F15', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F15', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F15', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F15', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("F16", className='distanceboxheader'),
                      html.Div("TR: {}".format(int(FO_stats['TR'].get('F16', 0))),
                               className='statboxcontent'),
                      html.Span([
                          html.Div("W: {}".format(FO_stats['W'].get('F16', '-')), className='foboxwin'),
                          html.Div("2: {}".format(FO_stats['SHP'].get('F16', '-')), className='foboxshp'),
                          html.Div("3: {}".format(FO_stats['THP'].get('F16', '-')), className='foboxthp'),
                          html.Div("Plc: {}".format(FO_stats['Plc'].get('F16', '-')), className='foboxplc')
                      ], className='boxspan'
                      ),
                      html.Span([
                          html.Div("W%: {}".format(FO_stats['W%'].get('F16', '-')), className='foboxwin'),
                          html.Div("2%: {}".format(FO_stats['SHP%'].get('F16', '-')), className='foboxshp'),
                          html.Div("3%: {}".format(FO_stats['THP%'].get('F16', '-')), className='foboxthp'),
                          html.Div("Plc%: {}".format(FO_stats['Plc%'].get('F16', '-')), className='foboxplc')
                      ], className='boxspan'
                      )],
                     className="mini_container", style={'width': '200px'}),
            html.Div(className="mini_container", style={'width': '200px', 'background-color':'#061e44'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'})
    ]
    return fo_boxes


def gen_idxmax_for_f1_f4_nonPC(stats, col, searchterm):
    if len(stats) == 0:
        return "-"
    else:
        return stats.loc[searchterm[0]][col].idxmax()

def gen_max_for_f1_f4_nonPC(stats, col, searchterm):
    if len(stats) == 0:
        return "-"
    else:
        return stats.loc[searchterm[0]][col].max()


def gen_idxmax_for_f1_f4_PC(stats, col, searchterm):
    if len(stats) == 0:
        return "-"
    else:
        return stats[stats['TR']>=1].loc[searchterm[0]][col].idxmax()


def gen_max_for_f1_f4_PC(stats, col, searchterm):
    if len(stats) == 0:
        return "-"
    else:
        return stats[stats['TR']>=1].loc[searchterm[0]][col].max()

def generate_f1_f4_stats(df, searchtype, searchterm):
    minimum_races=1
    if searchtype == 'Trainer':
        Trainer_stats = generate_trainer_jockey_stats(df)
        F1_Trainer_jockey_stats = generate_trainer_jockey_stats(df[df['FO']=='F1'])
        F2_Trainer_jockey_stats = generate_trainer_jockey_stats(df[df['FO'] == 'F2'])
        F3_Trainer_jockey_stats = generate_trainer_jockey_stats(df[df['FO'] == 'F3'])
        F4_Trainer_jockey_stats = generate_trainer_jockey_stats(df[df['FO'] == 'F4'])
    elif searchtype == 'Jockey':
        Trainer_stats = generate_jockey_trainer_stats(df)
        F1_Trainer_jockey_stats = generate_jockey_trainer_stats(df[df['FO']=='F1'])
        F2_Trainer_jockey_stats = generate_jockey_trainer_stats(df[df['FO'] == 'F2'])
        F3_Trainer_jockey_stats = generate_jockey_trainer_stats(df[df['FO'] == 'F3'])
        F4_Trainer_jockey_stats = generate_jockey_trainer_stats(df[df['FO'] == 'F4'])
    print("This is:",searchterm)
    f1_f4_stats = [
        html.H3("Partnership stats",
                style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                       'font-family': 'Arial Black'}),
        html.Div([
            html.Div("All", className='mini_container',
                     style={'textAlign': 'center', 'font-family': 'Arial Black', 'width': '55px', 'padding-top': '60px',
                            'color': '#fbff00'}),
            html.Div([html.H6("Most wins", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(Trainer_stats, 'W', searchterm)), className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(Trainer_stats, 'W', searchterm)), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest win%", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_PC(Trainer_stats, 'W%', searchterm)), className='statboxcontent'),
                      html.Div("{}%".format(gen_max_for_f1_f4_PC(Trainer_stats, 'W%', searchterm)), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest win streak", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(Trainer_stats, 'LWS', searchterm)), className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(Trainer_stats, 'LWS', searchterm)), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest losing streak", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(Trainer_stats, 'LLS', searchterm)), className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(Trainer_stats, 'LLS', searchterm)), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest Plc%", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_PC(Trainer_stats, 'Plc%', searchterm)), className='statboxcontent'),
                      html.Div("{}%".format(gen_max_for_f1_f4_PC(Trainer_stats, 'Plc%', searchterm)), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'}),
        html.Div([
            html.Div("F1", className='mini_container',
                     style={'textAlign': 'center', 'font-family': 'Arial Black', 'width': '55px', 'padding-top': '60px',
                            'color': '#fbff00'}),
            html.Div([html.H6("Most wins", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(F1_Trainer_jockey_stats, 'W', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(F1_Trainer_jockey_stats, 'W', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest win%", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_PC(F1_Trainer_jockey_stats, 'W%', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}%".format(gen_max_for_f1_f4_PC(F1_Trainer_jockey_stats, 'W%', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest win streak", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(F1_Trainer_jockey_stats, 'LWS', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(F1_Trainer_jockey_stats, 'LWS', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest losing streak", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(F1_Trainer_jockey_stats, 'LLS', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(F1_Trainer_jockey_stats, 'LLS', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest Plc%", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_PC(F1_Trainer_jockey_stats, 'Plc%', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}%".format(gen_max_for_f1_f4_PC(F1_Trainer_jockey_stats, 'Plc%', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'}),
        html.Div([
            html.Div("F2", className='mini_container',
                     style={'textAlign': 'center', 'font-family': 'Arial Black', 'width': '55px', 'padding-top': '60px',
                            'color': '#fbff00'}),
            html.Div([html.H6("Most wins", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(F2_Trainer_jockey_stats, 'W', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(F2_Trainer_jockey_stats, 'W', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest win%", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_PC(F2_Trainer_jockey_stats, 'W%', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}%".format(gen_max_for_f1_f4_PC(F2_Trainer_jockey_stats, 'W%', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest win streak", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(F2_Trainer_jockey_stats, 'LWS', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(F2_Trainer_jockey_stats, 'LWS', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest losing streak", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(F2_Trainer_jockey_stats, 'LLS', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(F2_Trainer_jockey_stats, 'LLS', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest Plc%", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_PC(F2_Trainer_jockey_stats, 'Plc%', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}%".format(gen_max_for_f1_f4_PC(F2_Trainer_jockey_stats, 'Plc%', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'}),
        html.Div([
            html.Div("F3", className='mini_container',
                     style={'textAlign': 'center', 'font-family': 'Arial Black', 'width': '55px', 'padding-top': '60px',
                            'color': '#fbff00'}),
            html.Div([html.H6("Most wins", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(F3_Trainer_jockey_stats, 'W', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(F3_Trainer_jockey_stats, 'W', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest win%", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_PC(F3_Trainer_jockey_stats, 'W%', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}%".format(gen_max_for_f1_f4_PC(F3_Trainer_jockey_stats, 'W%', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest win streak", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(F3_Trainer_jockey_stats, 'LWS', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(F3_Trainer_jockey_stats, 'LWS', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest losing streak", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(F3_Trainer_jockey_stats, 'LLS', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(F3_Trainer_jockey_stats, 'LLS', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest Plc%", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_PC(F3_Trainer_jockey_stats, 'Plc%', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}%".format(gen_max_for_f1_f4_PC(F3_Trainer_jockey_stats, 'Plc%', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'}),
        html.Div([
            html.Div("F4", className='mini_container',
                     style={'textAlign': 'center', 'font-family': 'Arial Black', 'width': '55px', 'padding-top': '60px',
                            'color': '#fbff00'}),
            html.Div([html.H6("Most wins", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(F4_Trainer_jockey_stats, 'W', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(F4_Trainer_jockey_stats, 'W', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest win%", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_PC(F4_Trainer_jockey_stats, 'W%', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}%".format(gen_max_for_f1_f4_PC(F4_Trainer_jockey_stats, 'W%', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest win streak", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(F4_Trainer_jockey_stats, 'LWS', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(F4_Trainer_jockey_stats, 'LWS', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Longest losing streak", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_nonPC(F4_Trainer_jockey_stats, 'LLS', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}".format(gen_max_for_f1_f4_nonPC(F4_Trainer_jockey_stats, 'LLS', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("Highest Plc%", className='statboxheader'),
                      html.Div("{}".format(gen_idxmax_for_f1_f4_PC(F4_Trainer_jockey_stats, 'Plc%', searchterm)),
                               className='statboxcontent'),
                      html.Div("{}%".format(gen_max_for_f1_f4_PC(F4_Trainer_jockey_stats, 'Plc%', searchterm)),
                               className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'})
    ]
    return f1_f4_stats

def generate_edge_stats(edge_for_win):
    minimum_races=1
    edge_stats = [
        html.H3("Edge formula stats",
                style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                       'font-family': 'Arial Black'}),
        html.Div([
            html.Div("ODDS", className='mini_container',
                     style={'textAlign': 'center', 'font-family': 'Arial Black', 'width': '55px', 'padding-top': '60px',
                            'color': '#fbff00'}),
            html.Div([html.H6("OPENING", className='statboxheader'),
                      html.Div("{}".format(edge_for_win[0]), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("LATEST", className='statboxheader'),
                      html.Div("{}".format(edge_for_win[1]), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("MIDDLE", className='statboxheader'),
                      html.Div("{}".format(edge_for_win[2]), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'}),
            html.Div([html.H6("FINAL", className='statboxheader'),
                      html.Div("{}".format(edge_for_win[3]), className='statboxnumcontent')],
                     className="mini_container", style={'width': '200px'})
        ], className="pretty_container",
            style={'display': 'flex', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
                   'padding': '0px'})
    ]
    return edge_stats


def generate_fav_table_split(df, split_by_type, pl):
    split_div_for_fav = []
    if pl in [1,2,3]:
        for i in df[split_by_type].unique():
            dff = df.loc[df[split_by_type] == i]
            a = html.Div(children=[
                html.Span("{}".format(i), className='split_fav_main_span'),
                "|",
                html.Span("TR: {} | {}%".format(len(dff),round(100*len(dff)/len(df),1)), className='split_fav_tr_span'),
                "|",
                html.Span("W: {} | {}%".format(len(dff[dff['Pl']==1]),round(100 * len(dff[dff['Pl']==1]) / len(dff), 1)), className='split_fav_win_span'),
                "|",
                html.Span("SHP: {} | {}%".format(len(dff[dff['Pl'] == 2]),round(100 * len(dff[dff['Pl'] == 2]) / len(dff), 1)), className='split_fav_shp_span'),
                "|",
                html.Span("THP: {} | {}%".format(len(dff[dff['Pl'] == 3]),round(100 * len(dff[dff['Pl'] == 3]) / len(dff), 1)), className='split_fav_thp_span'),
                "|",
                html.Span("Plc: {} | {}%".format(len(dff[(dff['Pl']>0) & (dff['Pl'] <= 3)]),round(100 * len(dff[(dff['Pl']>0) & (dff['Pl'] <= 3)]) / len(dff), 1)), className='split_fav_plc_span'),
                "|",
                html.Span("BO: {}".format(len(dff[(dff['Pl']==0) | (dff['Pl'] > 3)])), className='split_fav_bo_span'),
                "|",
                html.Span("LWS: {}".format(maxminstreaks(pl, dff)[0]), className='split_fav_ls_span'),
                html.Span("LLS: {}".format(maxminstreaks(pl, dff)[1]), className='split_fav_ls_span'),
                "|",
                html.Br(),
                html.Span("Streaks: {}".format(streak(dff)[0]), className='split_fav_streak_span'),
                html.Br(),
                html.Span(html.Ul(children=[generate_li_for_fav_split(pl, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))], className='split_fg_ul'), className='split_fav_formguide_span')
            ], className='split_fav_individual_div')
            split_div_for_fav.append(a)
    elif pl == 4:
        for i in df[split_by_type].unique():
            dff = df.loc[df[split_by_type] == i]
            a = html.Div(children=[
                html.Span("{}".format(i), className='split_fav_main_span'),
                "|",
                html.Span("TR: {} | {}%".format(len(dff), round(100 * len(dff) / len(df), 1)),
                          className='split_fav_tr_span'),
                "|",
                html.Span(
                    "W: {} | {}%".format(len(dff[dff['Pl'] == 1]), round(100 * len(dff[dff['Pl'] == 1]) / len(dff), 1)),
                    className='split_fav_win_span'),
                "|",
                html.Span("SHP: {} | {}%".format(len(dff[dff['Pl'] == 2]),
                                                 round(100 * len(dff[dff['Pl'] == 2]) / len(dff), 1)),
                          className='split_fav_shp_span'),
                "|",
                html.Span("THP: {} | {}%".format(len(dff[dff['Pl'] == 3]),
                                                 round(100 * len(dff[dff['Pl'] == 3]) / len(dff), 1)),
                          className='split_fav_thp_span'),
                "|",
                html.Span("Plc: {} | {}%".format(len(dff[(dff['Pl'] > 0) & (dff['Pl'] <= 3)]),
                                                 round(100 * len(dff[(dff['Pl'] > 0) & (dff['Pl'] <= 3)]) / len(dff),
                                                       1)), className='split_fav_plc_span'),
                "|",
                html.Span("BO: {}".format(len(dff[(dff['Pl'] == 0) | (dff['Pl'] > 3)])), className='split_fav_bo_span'),
                "|",
                html.Span("LWS: {}".format(streak(dff)[1]), className='split_fav_ls_span'),
                html.Span("LLS: {}".format(streak(dff)[2]), className='split_fav_ls_span'),
                "|",
                html.Br(),
                html.Span("Streaks: {}".format(streak(dff)[0]), className='split_fav_streak_span'),
                html.Br(),
                html.Span(html.Ul(children=[generate_li_for_fav_split(pl, i, j, n) for n, (i, j) in
                                            enumerate(zip(dff['Pl'][::-1], dff['FINAL'][::-1]))],
                                  className='split_fg_ul'), className='split_fav_formguide_span')
            ], className='split_fav_individual_div')
            split_div_for_fav.append(a)
    return split_div_for_fav


def generate_media_race_stats(df, paper):
    paper_df = df[['Season', 'Date', 'Race No', 'Classification', 'Pl', paper]].copy()
    if not paper_df[paper].isnull().all().all():
        paper_df['num_pred'] = paper_df[paper].replace(['F', 'S', 'T'], [1, 2, 3])
    else:
        paper_df['num_pred'] = paper_df[paper]
    paper_df.dropna(inplace=True)
    paper_df['num_pred'] = paper_df['num_pred'].astype(int)
    paper_df['check_res'] = list(zip(paper_df.num_pred, paper_df.Pl))
    combo = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    race_stats = paper_df.groupby(["Season", "Date", "Race No", "Classification", "check_res"], as_index=True).size().unstack(
        fill_value=0)
    mis_col = [i for i in combo if i not in race_stats.columns]
    for col in mis_col:
        #     print(col)
        race_stats["{}".format(col)] = [0] * len(race_stats)

    for n, col in enumerate(race_stats.columns):
        if type(col) is not tuple:
            print(tuple(map(int, col[1:-1].split(', '))))
            print(race_stats.columns[n])
            race_stats.rename(columns={col: tuple(map(int, col[1:-1].split(', ')))}, inplace=True)
    # race_stats = race_stats[combo]

    race_stats.reset_index(inplace=True)

    race_stats[1, 2].replace(1, 2, inplace=True)
    race_stats[1, 3].replace(1, 3, inplace=True)

    race_stats[2, 2].replace(1, 2, inplace=True)
    race_stats[2, 3].replace(1, 3, inplace=True)

    race_stats[3, 2].replace(1, 2, inplace=True)
    race_stats[3, 3].replace(1, 3, inplace=True)

    race_stats['firstplc'] = race_stats[1, 1] + race_stats[1, 2] + race_stats[1, 3]
    race_stats['secondplc'] = race_stats[2, 1] + race_stats[2, 2] + race_stats[2, 3]
    race_stats['thirdplc'] = race_stats[3, 1] + race_stats[3, 2] + race_stats[3, 3]

    race_stats['firstplc'].replace([0, 1, 2, 3], ['L', 'W', 'SHP', 'THP'], inplace=True)
    race_stats['secondplc'].replace([0, 1, 2, 3], ['L', 'W', 'SHP', 'THP'], inplace=True)
    race_stats['thirdplc'].replace([0, 1, 2, 3], ['L', 'W', 'SHP', 'THP'], inplace=True)

    race_stats[1, 2].replace(2, 1, inplace=True)
    race_stats[1, 3].replace(3, 1, inplace=True)

    race_stats[2, 2].replace(2, 1, inplace=True)
    race_stats[2, 3].replace(3, 1, inplace=True)

    race_stats[3, 2].replace(2, 1, inplace=True)
    race_stats[3, 3].replace(3, 1, inplace=True)

    race_stats[(2, 1)].replace(1, 2, inplace=True)
    race_stats[(3, 1)].replace(1, 3, inplace=True)

    race_stats[(2, 2)].replace(1, 2, inplace=True)
    race_stats[(3, 2)].replace(1, 3, inplace=True)

    race_stats[(2, 3)].replace(1, 2, inplace=True)
    race_stats[(3, 3)].replace(1, 3, inplace=True)

    race_stats['Win'] = race_stats[(1, 1)] + race_stats[(2, 1)] + race_stats[(3, 1)]
    race_stats['SHP'] = race_stats[(1, 2)] + race_stats[(2, 2)] + race_stats[(3, 2)]
    race_stats['THP'] = race_stats[(1, 3)] + race_stats[(2, 3)] + race_stats[(3, 3)]

    race_stats['Plc'] = race_stats['Win'] + race_stats['SHP'] + race_stats['THP']

    race_stats['Win'].replace([0, 1, 2, 3], ['L', 'F/W', 'S/W', 'T/W'], inplace=True)
    race_stats['SHP'].replace([0, 1, 2, 3], ['L', 'F/2', 'S/2', 'T/2'], inplace=True)
    race_stats['THP'].replace([0, 1, 2, 3], ['L', 'F/3', 'S/3', 'T/3'], inplace=True)
    race_stats['Plc'].replace([0, 1, 2, 3, 4, 5, 6], ['L', 'W', 'W', 'W', 'W', 'W', 'W'], inplace=True)

    for i in race_stats.columns:
        if type(i) is tuple:
            race_stats[i].replace([2, 3, 4, 5], [1, 1, 1, 1], inplace=True)

    race_stats['one'] = [0] * len(race_stats)
    race_stats['two'] = [0] * len(race_stats)
    race_stats['three'] = [0] * len(race_stats)
    for i in race_stats.columns:
        if i[0] == 3:
            race_stats['three'] = race_stats['three'] | race_stats[i[0], i[1]]
        elif i[0] == 2:
            race_stats['two'] = race_stats['two'] | race_stats[i[0], i[1]]
        elif i[0] == 1:
            race_stats['one'] = race_stats['one'] | race_stats[i[0], i[1]]

    return race_stats


def generate_li_for_media(n, pl, odds, box_number):
    codes = []
    cl = []
    if n in ['first', 'second', 'third']:
        codes = ["W", "SHP", "THP", "L"]
        cl = ['form_guide_f1f4__outcome form_guide_f1f4__outcome--win',
         'form_guide_f1f4__outcome form_guide_f1f4__outcome--shp',
         'form_guide_f1f4__outcome form_guide_f1f4__outcome--thp',
         'form_guide_f1f4__outcome form_guide_f1f4__outcome--bo']
    elif n == 'allwin':
        codes = ['F/W', 'S/W', 'T/W', 'L']
        cl = ['form_guide_f1f4__outcome form_guide_f1f4__outcome--f',
              'form_guide_f1f4__outcome form_guide_f1f4__outcome--s',
              'form_guide_f1f4__outcome form_guide_f1f4__outcome--t',
              'form_guide_f1f4__outcome form_guide_f1f4__outcome--l']
    elif n == 'allshp':
        codes = ['F/2', 'S/2', 'T/2', 'L']
        cl = ['form_guide_f1f4__outcome form_guide_f1f4__outcome--f',
              'form_guide_f1f4__outcome form_guide_f1f4__outcome--s',
              'form_guide_f1f4__outcome form_guide_f1f4__outcome--t',
              'form_guide_f1f4__outcome form_guide_f1f4__outcome--l']
    elif n == 'allthp':
        codes = ['F/3', 'S/3', 'T/3', 'L']
        cl = ['form_guide_f1f4__outcome form_guide_f1f4__outcome--f',
              'form_guide_f1f4__outcome form_guide_f1f4__outcome--s',
              'form_guide_f1f4__outcome form_guide_f1f4__outcome--t',
              'form_guide_f1f4__outcome form_guide_f1f4__outcome--l']
    elif n == 'allplc':
        codes = ['W','','','L']
        cl = ['form_guide_f1f4__outcome form_guide_f1f4__outcome--f',
              '',
              '',
              'form_guide_f1f4__outcome form_guide_f1f4__outcome--l']
    if box_number % 125 >= 25:
        if box_number % 125 == 0 and box_number != 0:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0],
                               style={"margin-top": "15px"})
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1],
                               style={"margin-top": "15px"})
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2],
                               style={"margin-top": "15px"})
            elif pl == codes[3]:
                return html.Li("{}".format(pl),
                               className=cl[3],
                               style={"margin-top": "15px"})
        elif box_number % 25 == 0 and box_number != 0:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0])
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1])
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2])
            elif pl == codes[3]:
                return html.Li("{}".format(pl),
                               className=cl[3])
        elif box_number%5 == 0 and box_number !=0:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0],
                               style={"margin-left": "20px"})
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1],
                               style={"margin-left": "20px"})
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2],
                               style={"margin-left": "20px"})
            elif pl == codes[3]:
                return html.Li("{}".format(pl),
                               className=cl[3],
                               style={"margin-left": "20px"})
        else:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0])
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1])
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2])
            elif pl == codes[3]:
                return html.Li("{}".format(pl),
                               className=cl[3])
    else:
        if box_number % 125 == 0 and box_number != 0:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0],
                               style={"margin-top": "15px"})
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1],
                               style={"margin-top": "15px"})
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2],
                               style={"margin-top": "15px"})
            elif pl == codes[3]:
                return html.Li("{}".format(pl),
                               className=cl[3],
                               style={"margin-top": "15px"})
        elif box_number % 25 == 0 and box_number != 0:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0],
                               style={"margin-top": "15px"})
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1],
                               style={"margin-top": "15px"})
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2],
                               style={"margin-top": "15px"})
            elif pl == codes[3]:
                return html.Li("{}".format(pl),
                               className=cl[3],
                               style={"margin-top": "15px"})
        elif box_number % 5 == 0 and box_number != 0:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0],
                               style={"margin-top": "15px", "margin-left": "20px"})
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1],
                               style={"margin-top": "15px", "margin-left": "20px"})
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2],
                               style={"margin-top": "15px", "margin-left": "20px"})
            elif pl == codes[3]:
                return html.Li("{}".format(pl),
                               className=cl[3],
                               style={"margin-top": "15px", "margin-left": "20px"})
        else:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0],
                               style={"margin-top": "15px"})
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1],
                               style={"margin-top": "15px"})
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2],
                               style={"margin-top": "15px"})
            elif pl == codes[3]:
                return html.Li("{}".format(pl),
                               className=cl[3],
                               style={"margin-top": "15px"})
            
            
def generate_li_for_media_split(n, pl, odds, box_number):
    codes = []
    cl = []
    if n in ['first']:
        codes = [1, 2, 3]
        cl = ['form_guide_f1f4_split__outcome form_guide_f1f4__outcome--win',
         'form_guide_f1f4_split__outcome form_guide_f1f4__outcome--bo',
         'form_guide_f1f4_split__outcome form_guide_f1f4__outcome--bo',
         'form_guide_f1f4_split__outcome form_guide_f1f4__outcome--bo']
    elif n in ['second']:
        codes = [1,2,3]
        cl = ['form_guide_f1f4_split__outcome form_guide_f1f4__outcome--bo',
              'form_guide_f1f4_split__outcome form_guide_f1f4__outcome--win',
              'form_guide_f1f4_split__outcome form_guide_f1f4__outcome--bo',
              'form_guide_f1f4_split__outcome form_guide_f1f4__outcome--bo']
    elif n in ['third']:
        codes = [1,2,3]
        cl = ['form_guide_f1f4_split__outcome form_guide_f1f4__outcome--bo',
              'form_guide_f1f4_split__outcome form_guide_f1f4__outcome--bo',
              'form_guide_f1f4_split__outcome form_guide_f1f4__outcome--win',
              'form_guide_f1f4_split__outcome form_guide_f1f4__outcome--bo']
    elif n in ['place']:
        codes = [1,2,3]
        cl = ['form_guide_f1f4_split__outcome form_guide_f1f4__outcome--win',
              'form_guide_f1f4_split__outcome form_guide_f1f4__outcome--win',
              'form_guide_f1f4_split__outcome form_guide_f1f4__outcome--win',
              'form_guide_f1f4_split__outcome form_guide_f1f4__outcome--bo']
    if box_number % 175 >= 35:
        if box_number % 175 == 0 and box_number != 0:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0],
                               style={"margin-top": "15px"})
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1],
                               style={"margin-top": "15px"})
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2],
                               style={"margin-top": "15px"})
            else:
                return html.Li("{}".format(pl),
                               className=cl[3],
                               style={"margin-top": "15px"})
        elif box_number % 35 == 0 and box_number != 0:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0])
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1])
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2])
            else:
                return html.Li("{}".format(pl),
                               className=cl[3])
        elif box_number%5 == 0 and box_number !=0:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0],
                               style={"margin-left": "20px"})
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1],
                               style={"margin-left": "20px"})
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2],
                               style={"margin-left": "20px"})
            else:
                return html.Li("{}".format(pl),
                               className=cl[3],
                               style={"margin-left": "20px"})
        else:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0])
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1])
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2])
            else:
                return html.Li("{}".format(pl),
                               className=cl[3])
    else:
        if box_number % 175 == 0 and box_number != 0:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0],
                               style={"margin-top": "15px"})
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1],
                               style={"margin-top": "15px"})
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2],
                               style={"margin-top": "15px"})
            else:
                return html.Li("{}".format(pl),
                               className=cl[3],
                               style={"margin-top": "15px"})
        elif box_number % 35 == 0 and box_number != 0:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0],
                               style={"margin-top": "15px"})
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1],
                               style={"margin-top": "15px"})
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2],
                               style={"margin-top": "15px"})
            else:
                return html.Li("{}".format(pl),
                               className=cl[3],
                               style={"margin-top": "15px"})
        elif box_number % 5 == 0 and box_number != 0:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0],
                               style={"margin-top": "15px", "margin-left": "20px"})
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1],
                               style={"margin-top": "15px", "margin-left": "20px"})
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2],
                               style={"margin-top": "15px", "margin-left": "20px"})
            else:
                return html.Li("{}".format(pl),
                               className=cl[3],
                               style={"margin-top": "15px", "margin-left": "20px"})
        else:
            if pl == codes[0]:
                return html.Li("{}".format(pl),
                               className=cl[0],
                               style={"margin-top": "15px"})
            elif pl == codes[1]:
                return html.Li("{}".format(pl),
                               className=cl[1],
                               style={"margin-top": "15px"})
            elif pl == codes[2]:
                return html.Li("{}".format(pl),
                               className=cl[2],
                               style={"margin-top": "15px"})
            else:
                return html.Li("{}".format(pl),
                               className=cl[3],
                               style={"margin-top": "15px"})


def generate_tjstats_for_media(df,basedon):
    pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    n = df.groupby([basedon, "Pl"], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    n = n[pl]
    m = df.groupby([basedon], as_index=True).size().to_frame()
    k = df.groupby([basedon], as_index=True)['Pl'].agg([maxwinstreaks, maxlosestreaks])
    Trainer_stats = pd.concat([n, m, k], axis=1)
    Trainer_stats.rename(columns={0: 'TR', 1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'},
                         inplace=True)
    Trainer_stats["W%"] = ((Trainer_stats["W"] / Trainer_stats['TR']) * 100).round(2)
    Trainer_stats["SHP%"] = ((Trainer_stats["SHP"] / Trainer_stats['TR']) * 100).round(2)
    Trainer_stats["THP%"] = ((Trainer_stats["THP"] / Trainer_stats['TR']) * 100).round(2)
    Trainer_stats["Plc%"] = (
                ((Trainer_stats["W"] + Trainer_stats["SHP"] + Trainer_stats["THP"]) / Trainer_stats['TR']) * 100).round(
        2)
    Trainer_stats["Plc"] = Trainer_stats["W"] + Trainer_stats["SHP"] + Trainer_stats["THP"]
    Trainer_stats["BO"] = Trainer_stats["TR"] - Trainer_stats["Plc"]
    Trainer_stats["TR%"] = ((Trainer_stats["TR"] / len(df)) * 100).round(2)
    return Trainer_stats[['TR', 'TR%', 'W', 'W%', 'SHP', 'SHP%', 'THP', 'THP%', 'Plc', 'Plc%', 'BO', 'LWS', 'LLS']]




filters = [
    html.P("Centre:", className="control_label"),
    html.Div(
        dcc.Dropdown(
                id='Centre',
                className='dcc_control',
                options=[
                    {'label': i, 'value': i} for i in sorted(all_df['Centre'].unique())
                ],
                value='BLR',
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
                    {'label': 'Horse', 'value': 'Horse'},
                    {'label': 'Trainer', 'value': 'Trainer'},
                    {'label': 'Jockey', 'value': 'Jockey'},
                    {'label': 'Opening Odds', 'value': 'OO'},
                    {'label': 'Latest Odds', 'value': 'LO'},
                    {'label': 'Middle Odds', 'value': 'MO'},
                    {'label': 'Final Odds', 'value': 'FO'},
                    {'label': 'Media', 'value': 'Media'}
                ],
                value='Trainer',
                clearable=False
            ),
        className='dash-dropdown'
    ),
    html.P("Search:", className="control_label"),
    html.Div(
        dcc.Dropdown(
                id='identity',
                className='dcc_control',
                value=None,
                placeholder='Search',
                multi=True
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
                id='Classification',
                className='dcc_control',
                value=None,
                placeholder='Classification',
                multi=True,
                style={'font-size': '1.2rem'}
            ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
                id='Trainer',
                className='dcc_control',
                value=None,
                placeholder='Trainer',
                multi=True
            ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
                id='Jockey',
                className='dcc_control',
                value=None,
                placeholder='Jockey',
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
                id='Dr',
                className='dcc_control',
                value=None,
                placeholder='Dr',
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
            id='Weight',
            className='dcc_control',
            value=None,
            placeholder='Weight',
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.P("Odds:", className="control_label"),
    html.Div(
        dcc.Dropdown(
                id='Odds',
                className='dcc_control',
                options=[
                    {'label': 'Opening Odds', 'value': 'OO'},
                    {'label': 'Latest Odds', 'value': 'LO'},
                    {'label': 'Middle Odds', 'value': 'MO'},
                    {'label': 'Final Odds', 'value': 'FO'}
                ],
                value='OO',
                clearable=False
            ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
                id='Favourite',
                className='dcc_control',
                value=None,
                placeholder='F1-F16',
                multi=True,
            ),
        className='dash-dropdown'
    )

]


season_select = [
    html.Div(
        dcc.Dropdown(
                id='fav_season',
                className='dcc_fav_control',
                value='All',
                clearable=False
            ),
        className='fav_dropdown'
    ),
    html.Div(
        dcc.Dropdown(
                id='fav_jockey',
                className='dcc_fav_control',
                value='All',
                clearable=False
            ),
        className='fav_dropdown'
    ),
    html.Div(
        dcc.Dropdown(
                id='fav_trainer',
                className='dcc_fav_control',
                value='All',
                clearable=False
            ),
        className='fav_dropdown'
    ),
    html.Div(
        dcc.Dropdown(
                id='fav_distance',
                className='dcc_fav_control',
                value='All',
                clearable=False
            ),
        className='fav_dropdown'
    ),
    html.Div(
        dcc.Dropdown(
                id='fav_raceno',
                className='dcc_fav_control',
                value='All',
                clearable=False
            ),
        className='fav_dropdown'
    ),
    html.Div(
        dcc.Dropdown(
                id='fav_dr',
                className='dcc_fav_control',
                value='All',
                clearable=False
            ),
        className='fav_dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='fav_age',
            className='dcc_fav_control',
            value='All',
            clearable=False
        ),
        className='fav_dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='fav_weight',
            className='dcc_fav_control',
            value='All',
            clearable=False
        ),
        className='fav_dropdown'
    ),
    html.Div(
        dcc.Dropdown(
                id='fav_class',
                className='dcc_fav_control',
                value='All',
                clearable=False
            ),
        className='fav_dropdown'
    )
]

media_predtype_select_trainer = html.Div(
        dcc.Dropdown(
                id='media_FST_trainer',
                className='dcc_fav_control',
                options=[
                    {'label': 'F', 'value': 'F'},
                    {'label': 'S', 'value': 'S'},
                    {'label': 'T', 'value': 'T'},
                    {'label': 'All', 'value': 'All'}
                ],
                value='F',
                clearable=False
            ),
        className='fav_dropdown'
    )



media_predtype_select_jockey = html.Div(
        dcc.Dropdown(
                id='media_FST_jockey',
                className='dcc_fav_control',
                options=[
                    {'label': 'F', 'value': 'F'},
                    {'label': 'S', 'value': 'S'},
                    {'label': 'T', 'value': 'T'},
                    {'label': 'All', 'value': 'All'}
                ],
                value='F',
                clearable=False
            ),
        className='fav_dropdown'
    )


row0 = dbc.Row(
    [
        dbc.Col(
            [
                html.Div(id='Overall_scoreboard', className='pretty_container', style={'display': 'none'})
            ],
            width=12, className='pretty_container twelve columns', id='base_right-column'
        )
    ], id='fixedontop', className='flex-display fixedontop'
)



row1 = dbc.Row(
    [
        dbc.Col(
                    children=[html.H2("Racing Analytics", id='title'),
                              html.Div(filters, id='cross-filter-options', className='pretty_container'),
                              html.Div(id='odds_list', className='pretty_container', style={'display': 'none'})],
                    width=3, className='three columns', style={'display': 'flex', 'flex-direction': 'column'}
                ),
        dbc.Col(
            [
                html.Div(id='scoreboard', className='pretty_container', style={'display': 'none'}),
                html.Div(id='Favourite_table', className='pretty_container', style={'display': 'none'}),
                html.Div(id='f1_f4_display', children=[], className='pretty_container', style={'display': 'none'})
            ],
            width=9, className='nine columns', id='right-column'
        )
    ], className='flex-display'
)

row1point5 = dbc.Row(
    [
        dbc.Col(html.Div(id='form_display', className='pretty_container'), width=12, className='pretty_container twelve columns', id='oneandhalfleft', style={'display': 'none'})
    ], className='flex-display'
)

row1point6 = dbc.Row(
    [
        dbc.Col(html.Div(id='overallsplitSB', className='pretty_container'), width=12, className='pretty_container twelve columns', id='onepointsix', style={'display': 'none'})
    ], className='flex-display'
)

row1point7 = dbc.Row(
    [
        dbc.Col(html.Div(id='splitSB', className='pretty_container'), width=12, className='pretty_container twelve columns', id='five', style={'display': 'none'})
    ], className='flex-display'
)

row2 = dbc.Row(
    [
        dbc.Col(html.Div(id='firstgraph', className='dash-graph'), width=12, className='pretty_container twelve columns', id='twoleft', style={'display': 'none'})
    ], className='flex-display'
)

row3 = dbc.Row(
    [
        dbc.Col(html.Div(id='thirdgraph', className='dash-graph'), width=12, className='pretty_container twelve columns', id='threeleft', style={'display': 'none'})
    ], className='flex-display'
)



row34 = dbc.Row(
    [
        dbc.Col(html.Div(id='thirdfourthgraph', className='dash-graph'), width=12, className='pretty_container twelve columns', id='threefourleft', style={'display': 'none'})
    ], className='flex-display'
)


row4 = dbc.Row(
    dbc.Col(html.Div(id='fifthgraph', className='dash-graph'), width=12, className='pretty_container twelve columns', id='fourleft', style={'display': 'none'}), className='flex-display'
)

# row5 = dbc.Row(
#     dbc.Col(html.Div(id='splitSB', className='pretty_container'), width=12, className='pretty_container twelve columns', id='five', style={'display': 'none'}), className='flex-display'
# )

row2point5 = dbc.Row(
    [
        dbc.Col(html.Div(id='twopointfivegraph', className='dash-graph'), width=12, className='pretty_container twelve columns', id='twopointfiveleft', style={'display': 'none'})
    ], className='flex-display'
)

row2point75 = dbc.Row(
    [
        dbc.Col(html.Div(children=[html.Div(season_select), html.Div(id='twopointsevenfivegraph', className='dash-graph')], id='fav_detailed_container'), width=12, className='pretty_container twelve columns', id='twopointsevenfiveleft', style={'display': 'none'})
    ], className='flex-display'
)

row2point85 = dbc.Row(
    [
        dbc.Col(html.Div(id='twopointeightfivegraph', className='dash-graph'), width=12, className='pretty_container twelve columns', id='twopointeightfiveleft', style={'display': 'none'})
    ], className='flex-display'
)

row2point95 = dbc.Row(
    [
        dbc.Col(html.Div(id='twopointninefivegraph', className='dash-graph'), width=12, className='pretty_container twelve columns', id='twopointninefiveleft', style={'display': 'none'})
    ], className='flex-display'
)

row2point995 = dbc.Row(
    [
        dbc.Col(html.Div(id='twopointnineninefivegraph', className='dash-graph'), width=12, className='pretty_container twelve columns', id='twopointnineninefiveleft', style={'display': 'none'})
    ], className='flex-display'
)



app.layout = html.Div([
                       row0,
                       row1,
                       row1point5,
                       row1point6,
                       row1point7,
                       row2,
                       row2point5,
                       row2point75,
                       row2point85,
                       row2point95,
                       row2point995,
                       row3,
                       row34,
                       row4,
                       # row5,
                       html.Div(id="store", style={'display': 'none'}),
                       html.Div(id="store_centre_df", style={'display': 'none'}),
                       html.Div(id='store_media_race_stats', style={'display': 'none'})],
                      id='mainContainer', style={'display': 'flex', 'flex-direction': 'column'})

@app.callback(
    [Output(component_id='store_centre_df', component_property='children'),
     Output(component_id='identity', component_property='value'),
     Output(component_id='Season', component_property='value'),
     Output(component_id='Distance', component_property='value'),
     Output(component_id='Classification', component_property='value'),
     Output(component_id='Trainer', component_property='value'),
     Output(component_id='Jockey', component_property='value'),
     Output(component_id='Race_no', component_property='value'),
     Output(component_id='Dr', component_property='value'),
     Output(component_id='Age', component_property='value'),
     Output(component_id='Weight', component_property='value'),
     Output(component_id='Favourite', component_property='value')],
    [Input(component_id='Centre', component_property='value')]
)
def update_all_dropdowns(Centre):
    current_centre_df = all_df[all_df['Centre']==Centre]
    # print(len(current_centre_df))
    return[[current_centre_df.to_json(orient='split')],None, None, None, None, None, None, None, None, None, None, None]



@app.callback(
    [Output(component_id='identity', component_property='options'),
     Output(component_id='identity', component_property='value'),
     Output(component_id='Season', component_property='value'),
     Output(component_id='Distance', component_property='value'),
     Output(component_id='Classification', component_property='value'),
     Output(component_id='Trainer', component_property='value'),
     Output(component_id='Jockey', component_property='value'),
     Output(component_id='Race_no', component_property='value'),
     Output(component_id='Dr', component_property='value'),
     Output(component_id='Age', component_property='value'),
     Output(component_id='Weight', component_property='value'),
     Output(component_id='Favourite', component_property='value'),
     Output(component_id='identity', component_property='multi')],
    [Input(component_id='type', component_property='value'),
     Input(component_id='store_centre_df', component_property='children')],
    prevent_initial_call=True
)
def update_main_dropdown(searchtype, frame):
    if frame is None:
        print("id failed")
        raise PreventUpdate
    df = pd.read_json(frame[0], orient='split')
    if searchtype in ['OO','LO','MO','FO']:
        options = [{"label": i, "value": i} for i in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F0']]
    elif searchtype == 'Media':
        options = [{"label": i, "value": i} for i in sorted(media_list)]
    else:
        try:
            options = [{'label': i, 'value': i} for i in sorted([x for x in list(df[searchtype].unique()) if str(x) != 'nan'])]
        except:
            options = [{'label': i, 'value': i} for i in df[searchtype].unique()]
    if searchtype == 'Media':
        multiple = False
    else:
        multiple = True
    return[options, None, None, None, None, None, None, None, None, None, None, None, multiple]


@app.callback(
    [Output(component_id='Season', component_property='options'),
     Output(component_id='Distance', component_property='options'),
     Output(component_id='Classification', component_property='options'),
     Output(component_id='Trainer', component_property='options'),
     Output(component_id='Jockey', component_property='options'),
     Output(component_id='Jockey', component_property='disabled'),
     Output(component_id='Race_no', component_property='options'),
     Output(component_id='Dr', component_property='options'),
     Output(component_id='Age', component_property='options'),
     Output(component_id='Weight', component_property='options'),
     Output(component_id='Season', component_property='value'),
     Output(component_id='Distance', component_property='value'),
     Output(component_id='Classification', component_property='value'),
     Output(component_id='Trainer', component_property='value'),
     Output(component_id='Jockey', component_property='value'),
     Output(component_id='Race_no', component_property='value'),
     Output(component_id='Dr', component_property='value'),
     Output(component_id='Age', component_property='value'),
     Output(component_id='Weight', component_property='value'),
     Output(component_id='Favourite', component_property='value')],
    [Input(component_id='identity', component_property='value')],
    [State(component_id='type', component_property='value'),
     State(component_id='store_centre_df', component_property='children')],
    prevent_initial_call=True
)
def update_filter_dropdowns(searchterm,searchtype, frame):
    if frame is None:
        print("filter failed")
        raise PreventUpdate
    df = pd.read_json(frame[0], orient='split')
    dis = False
    if searchterm is None or searchterm == []:
        return [[],[],[],[],[],dis,[],[],[],[],None,None,None,None,None,None,None,None,None,None]
    if searchtype == "Jockey":
        dis = True
    if searchtype == 'Media':
        try:
            s = [{"label": i, "value": i} for i in df['Season'].unique()]
            d = [{"label": i, "value": i} for i in sorted([x for x in list(df['Distance'].unique()) if str(x) != 'nan'])]
            c = [{"label": i, "value": i} for i in df['Classification'].unique()]
            t = [{"label": i, "value": i} for i in sorted([x for x in list(df['Trainer'].unique()) if str(x) != 'nan'])]
            j = [{"label": i, "value": i} for i in sorted([x for x in list(df['Jockey'].unique()) if str(x) != 'nan'])]
            rn = [{"label": i, "value": i} for i in sorted([x for x in list(df['Race No'].unique()) if str(x) != 'nan'])]
            dr = [{"label": i, "value": i} for i in sorted([x for x in list(df['Dr'].unique()) if str(x) != 'nan'])]
            age = [{"label": i, "value": i} for i in sorted([x for x in list(df['Age'].unique()) if str(x) != 'nan'])]
            weight = [{"label": i, "value": i} for i in sorted([x for x in list(df['Wt'].unique()) if str(x) != 'nan'])]
        except:
            s = [{"label": i, "value": i} for i in df['Season'].unique()]
            d = [{"label": i, "value": i} for i in df['Distance'].unique()]
            c = [{"label": i, "value": i} for i in df['Classification'].unique()]
            t = [{"label": i, "value": i} for i in df['Trainer'].unique()]
            j = [{"label": i, "value": i} for i in df['Jockey'].unique()]
            rn = [{"label": i, "value": i} for i in df['Race No'].unique()]
            dr = [{"label": i, "value": i} for i in df['Dr'].unique()]
            age = [{"label": i, "value": i} for i in df['Age'].unique()]
            weight = [{"label": i, "value": i} for i in df['Wt'].unique()]
    else:
        try:
            s = [{"label": i, "value": i} for i in df.loc[df[searchtype].isin(searchterm)]['Season'].unique()]
            d = [{"label": i, "value": i} for i in sorted([x for x in list(df.loc[df[searchtype].isin(searchterm)]['Distance'].unique()) if str(x) != 'nan'])]
            c = [{"label": i, "value": i} for i in df.loc[df[searchtype].isin(searchterm)]['Classification'].unique()]
            t = [{"label": i, "value": i} for i in sorted([x for x in list(df.loc[df[searchtype].isin(searchterm)]['Trainer'].unique()) if str(x) != 'nan'])]
            j = [{"label": i, "value": i} for i in sorted([x for x in list(df.loc[df[searchtype].isin(searchterm)]['Jockey'].unique()) if str(x) != 'nan'])]
            rn = [{"label": i, "value": i} for i in sorted([x for x in list(df.loc[df[searchtype].isin(searchterm)]['Race No'].unique()) if str(x) != 'nan'])]
            dr = [{"label": i, "value": i} for i in sorted([x for x in list(df.loc[df[searchtype].isin(searchterm)]['Dr'].unique()) if str(x) != 'nan'])]
            age = [{"label": i, "value": i} for i in sorted([x for x in list(df.loc[df[searchtype].isin(searchterm)]['Age'].unique()) if str(x) != 'nan'])]
            weight = [{"label": i, "value": i} for i in sorted([x for x in list(df.loc[df[searchtype].isin(searchterm)]['Wt'].unique()) if str(x) != 'nan'])]
        except:
            s = [{"label": i, "value": i} for i in df.loc[df[searchtype].isin(searchterm)]['Season'].unique()]
            d = [{"label": i, "value": i} for i in df.loc[df[searchtype].isin(searchterm)]['Distance'].unique()]
            c = [{"label": i, "value": i} for i in df.loc[df[searchtype].isin(searchterm)]['Classification'].unique()]
            t = [{"label": i, "value": i} for i in df.loc[df[searchtype].isin(searchterm)]['Trainer'].unique()]
            j = [{"label": i, "value": i} for i in df.loc[df[searchtype].isin(searchterm)]['Jockey'].unique()]
            rn = [{"label": i, "value": i} for i in df.loc[df[searchtype].isin(searchterm)]['Race No'].unique()]
            dr = [{"label": i, "value": i} for i in df.loc[df[searchtype].isin(searchterm)]['Dr'].unique()]
            age = [{"label": i, "value": i} for i in df.loc[df[searchtype].isin(searchterm)]['Age'].unique()]
            weight = [{"label": i, "value": i} for i in df.loc[df[searchtype].isin(searchterm)]['Wt'].unique()]
    return[s, d, c, t, j, dis, rn, dr, age, weight,None,None,None,None,None,None,None,None,None,None]


@app.callback(
    [Output(component_id='Favourite', component_property='options'),
     Output(component_id='Favourite', component_property='value')],
    [Input(component_id='Odds', component_property='value'),
     Input(component_id='type', component_property='value'),
     Input(component_id='identity', component_property='value')]
)
def update_favourite_dropdown(odds_type, searchtype, searchterm):
    print("I am hear", odds_type)
    if searchterm is None or searchterm == []:
        return [[],None]
    # fav = [{"label": i, "value": i} for i in unique_non_null(df.loc[df[searchtype].isin(searchterm)][odds_type])]
    fav = [{"label": i, "value": i} for i in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F0']]
    return [fav, None]



@app.callback(
    [Output(component_id='Overall_scoreboard', component_property='children'),
     Output(component_id='Overall_scoreboard', component_property='style'),
     Output(component_id='scoreboard', component_property='children'),
     Output(component_id='Favourite_table', component_property='children'),
     Output(component_id='form_display', component_property='children'),
     Output(component_id='splitSB', component_property='children'),
     Output(component_id='store', component_property='children'),
     Output(component_id='scoreboard', component_property='style'),
     Output(component_id='Favourite_table', component_property='style'),
     Output(component_id='oneandhalfleft', component_property='style'),
     Output(component_id='twoleft', component_property='style'),
     Output(component_id='twopointfiveleft', component_property='style'),
     Output(component_id='twopointsevenfiveleft', component_property='style'),
     Output(component_id='threeleft', component_property='style'),
     Output(component_id='threefourleft', component_property='style'),
     Output(component_id='fourleft', component_property='style'),
     Output(component_id='five', component_property='style'),
     Output(component_id='odds_list', component_property='children'),
     Output(component_id='odds_list', component_property='style'),
     Output(component_id='f1_f4_display', component_property='children'),
     Output(component_id='f1_f4_display', component_property='style'),
     Output(component_id='overallsplitSB', component_property='children'),
     Output(component_id='onepointsix', component_property='style'),
     Output(component_id='twopointeightfiveleft', component_property='style'),
     Output(component_id='twopointninefiveleft', component_property='style'),
     Output(component_id='twopointnineninefiveleft', component_property='style'),
     Output(component_id='store_media_race_stats', component_property='children'),
     Output(component_id='fixedontop', component_property='className')],
    [Input(component_id='store_centre_df', component_property='children'),
     Input(component_id='identity', component_property='value'),
     Input(component_id='type', component_property='value'),
     Input(component_id='Odds', component_property='value'),
     Input(component_id='Season', component_property='value'),
     Input(component_id='Distance', component_property='value'),
     Input(component_id='Classification', component_property='value'),
     Input(component_id='Trainer', component_property='value'),
     Input(component_id='Jockey', component_property='value'),
     Input(component_id='Race_no', component_property='value'),
     Input(component_id='Dr', component_property='value'),
     Input(component_id='Age', component_property='value'),
     Input(component_id='Weight', component_property='value'),
     Input(component_id='Favourite', component_property='value')],
    prevent_initial_call=True
)
def display_scoreboard(frame, searchterm,searchtype,odds_type,*searchparam):
    #Generates scoreboard based on search filters
    if frame is None:
        print("Hello:", frame)
        raise PreventUpdate
    df = pd.read_json(frame[0], orient='split')

    Jockey_Trainer_stats = generate_jockey_trainer_stats(df)
    Trainer_jockey_stats = generate_trainer_jockey_stats(df)
    Trainer_stats = generate_trainer_stats(df)
    Jockey_stats = generate_jockey_stats(df)
    media_stats = generate_media_stats(df)
    overall_stats = generate_overall_stats(Trainer_stats, Jockey_stats, Trainer_jockey_stats, df['Centre'].unique()[0], media_stats)
    if searchterm is None or searchterm == [] or searchterm == "":
        print("A return")
        return [[], {'display': 'none'}, overall_stats, [], [], generate_additional_scoreboards(Trainer_stats, "Trainer"), [], {'display': 'initial', 'background-color': '#061e44', 'margin-right':'0px'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'initial'}, [], {'display': 'none'}, [], {'display': 'none'}, generate_additional_scoreboards(Jockey_stats, "Jockey"), {'display': 'initial'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, [], 'flex-display fixedontop']
    print(searchterm,'-',searchtype, odds_type, searchparam[-1])
    if searchtype != 'Media':
        dff = pd.concat([group for (name, group) in df.groupby(searchtype) if name in searchterm])
    else:
        dff = df
    overall_dff = dff
    if searchtype == 'Media':
        overall_media_race_stats = generate_media_race_stats(overall_dff, searchterm)
    if searchtype == 'Jockey':
        XDF = generate_jockey_trainer_stats(overall_dff)
    elif searchtype == 'Trainer':
        XDF = generate_trainer_jockey_stats(overall_dff)
    cond = []
    cols = ['Season', 'Distance', 'Classification', 'Trainer', 'Jockey', 'Race No', 'Dr', 'Age', 'Wt', odds_type]
    condition = []
    for (key, value) in zip(cols, searchparam):
        #print(key,'=',value)
        if value != [] and value is not None and value !="":
            for v in value:
                # print('V--->',v)
                if (v != "All" and v is not None and v!=""):
                    cond.append(dff[key] == v)
            if cond != []:
                condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    if len(dff)==0:
        print("B return")
        return [[], {'display': 'none'},html.H1("No Races to show"), [], [], [], [],
                {'display': 'initial', 'background-color': '#061e44', 'margin-right': '0px', 'textAlign': 'center'}, {'display': 'none'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, {'display': 'none'}, [], {'display': 'none'}, [], {'display': 'none'}, [], {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},[], 'flex-display fixedontop']
    foo = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14']
    pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # distance_stats = generate_distance_stats(dff)
    if searchtype == 'Media':
        media_race_stats = generate_media_race_stats(dff, searchterm)
        firstdf = media_race_stats[media_race_stats['one'] == 1]
        seconddf = media_race_stats[media_race_stats['two'] == 1]
        thirddf = media_race_stats[media_race_stats['three'] == 1]
    if searchtype=='Jockey':
        xdf = generate_jockey_trainer_stats(dff)
    elif searchtype=='Trainer':
        xdf = generate_trainer_jockey_stats(dff)
    elif searchtype=='OO':
        xdf = generate_OO_stats(df)
    elif searchtype=='LO':
        xdf = generate_LO_stats(df)
    elif searchtype=='MO':
        xdf = generate_MO_stats(df)
    elif searchtype=='FO':
        xdf = generate_FO_stats(df)
    if searchtype == 'Media':
        Overall_SB = generate_overall_scoreboard_for_media(overall_media_race_stats, media_race_stats)
    else:
        Overall_SB = generate_overall_scoreboard(overall_dff,dff)
    # SB = generate_scoreboard()
    fav_table = generate_favourite_table(dff)
    odds_summary = [
        html.Div("Average Odds for W:", style={'font-weight':'bold', 'color': '#2ECC71'}),
        html.Div("Open  : {}".format(round(dff.loc[dff['Pl']==1]['OPEN'].mean(),2)), style={'font-weight':'bold', 'color': 'white'}),
        html.Div("Latest: {}".format(round(dff.loc[dff['Pl'] == 1]['LATEST'].mean(),2)), style={'font-weight':'bold', 'color': 'white'}),
        html.Div("Middle: {}".format(round(dff.loc[dff['Pl'] == 1]['MIDDLE'].mean(),2)), style={'font-weight':'bold', 'color': 'white'}),
        html.Div("Final : {}".format(round(dff.loc[dff['Pl'] == 1]['FINAL'].mean(),2)), style={'font-weight':'bold', 'color': 'white'}),
    ]
    edge_for_win = [round((dff.loc[dff['Pl'] == 1]['OPEN'].mean()*(len(dff.loc[dff['Pl'] == 1]) / len(dff))-(1 - (len(dff.loc[dff['Pl'] == 1]) / len(dff)))), 2),
                    round((dff.loc[dff['Pl'] == 1]['LATEST'].mean()*(len(dff.loc[dff['Pl'] == 1]) / len(dff))-(1 - (len(dff.loc[dff['Pl'] == 1]) / len(dff)))),2),
                    round((dff.loc[dff['Pl'] == 1]['MIDDLE'].mean()*(len(dff.loc[dff['Pl'] == 1]) / len(dff))-(1 - (len(dff.loc[dff['Pl'] == 1]) / len(dff)))),2),
                    round((dff.loc[dff['Pl'] == 1]['FINAL'].mean()*(len(dff.loc[dff['Pl'] == 1]) / len(dff))-(1 - (len(dff.loc[dff['Pl'] == 1]) / len(dff)))),2)]
    edge_for_win = ['No wins' if math.isnan(x) else x for x in edge_for_win]
    print(edge_for_win)
    if searchtype == 'Media':
        print("C return")
        # print("This return")
        return [
            Overall_SB,
            {'display': 'initial'},
            [],
            [
            html.Div(id="form", children=[
                 html.Div('Form Guide(Any->W):', className='form_guide_header'),
                 html.Ul(id='form_guide_ul',
                     className='form_guide_ul_first',
                         children=[generate_li_for_media('allwin', i, j, n) for n, (i, j) in
                                  enumerate(zip(media_race_stats['Win'][::-1],
                                                media_race_stats['Win'][::-1]))])]),
            html.Div([html.Div("Streaks:", className='streak_header'),
                      "{}".format(streak_media('win',media_race_stats['Win'])[0])],
                     id='streak_display'),
            html.Div(id="form_first", children=[
                html.Div('Form Guide(F):', className='form_guide_header'),
                html.Ul(id='form_guide_ul_f',
                        className='form_guide_ul_win',
                        children=[generate_li_for_media('first', i, j, n) for n, (i, j) in
                                  enumerate(zip(firstdf['firstplc'][::-1],
                                                firstdf['firstplc'][::-1]))])]),
            html.Div(id="form_second", children=[
                html.Div('Form Guide(S):', className='form_guide_header'),
                html.Ul(id='form_guide_ul_s',
                        className='form_guide_ul_win',
                        children=[generate_li_for_media('second', i, j, n) for n, (i, j) in
                                  enumerate(zip(seconddf['secondplc'][::-1],
                                                seconddf['secondplc'][::-1]))])]),
            html.Div(id="form_third", children=[
                html.Div('Form Guide(T):', className='form_guide_header'),
                html.Ul(id='form_guide_ul_t',
                        className='form_guide_ul_win',
                        children=[generate_li_for_media('third', i, j, n) for n, (i, j) in
                                  enumerate(zip(thirddf['thirdplc'][::-1],
                                                thirddf['thirdplc'][::-1]))])])
             ],
            [
                media_predtype_select_trainer,
                html.Div(id='media_trainer_split')
            ],
            [], [dff.to_json(orient='split')], {'display': 'none'}, {'display': 'initial'}, {'display': 'initial'},
            {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
            {'display': 'none'}, {'display': 'none'},
            [], {'display': 'initial'}, [], {'display': 'none'},
            [
                media_predtype_select_jockey,
                html.Div(id='media_jockey_split')
            ],
            {'display': 'initial'}, {'display': 'none'},
            {'display': 'none'}, {'display': 'none'}, [media_race_stats.to_json(orient='index')],
        'flex-display notfixedontop']
    elif len(searchterm)==1 and searchtype not in ['OO', 'LO', 'MO', 'FO', 'Horse']:
        print("D return")
        return[
            Overall_SB,
            {'display':'initial'},
            generate_edge_stats(edge_for_win),
            fav_table,
            [html.Div([html.Div("Streaks:", className='streak_header'), "{}".format(streak(dff)[0])], id='streak_display'),
             html.Div(id="form", children=[
                html.Div('Form Guide(W):', className='form_guide_header'),
                html.Ul(id='form_guide_ul',className='standings-table__form',
                     children=[generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))])])],
            generate_split_scoreboard(xdf, searchterm, "Filtered"),
            [dff.to_json(orient='split')],
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            odds_summary,
            {'display': 'initial'},
            generate_f1_f4_stats(dff, searchtype, searchterm),
            {'display': 'initial'},
            generate_split_scoreboard(XDF, searchterm, "Overall"),
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            [],
            'flex-display fixedontop'
        ]
    elif searchtype in ['OO', 'LO', 'MO', 'FO']:
        print("E return")
        return [
                Overall_SB,
                {'display': 'initial'},
                [],
                [html.Div(), dcc.Graph(figure=generatebar(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                [], [], [dff.to_json(orient='split')], {'display': 'none'}, {'display': 'initial'}, {'display': 'none'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                odds_summary, {'display': 'initial'}, [], {'display': 'none'},[], {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, [], 'flex-display fixedontop']
    elif len(searchterm)>=2 and searchtype not in ['OO', 'LO', 'MO', 'FO', 'Horse']:
        print("F return")
        return [
                Overall_SB,
                {'display': 'initial'},
                generate_edge_stats(edge_for_win),
                fav_table,
                [],
                [],
                [dff.to_json(orient='split')],
                {'display': 'initial'},
                {'display': 'initial'},
                {'display': 'none'},
                {'display': 'initial'},
                {'display': 'initial'},
                {'display': 'initial'},
                {'display': 'initial'},
                {'display': 'initial'},
                {'display': 'none'},
                {'display': 'none'},
                odds_summary,
                {'display': 'initial'},
                [],
                {'display': 'none'},
                [],
                {'display': 'none'},
                {'display': 'initial'},
                {'display': 'initial'},
                {'display': 'initial'},
                [],
                'flex-display fixedontop'
            ]
    elif len(searchterm)==1 and searchtype=="Horse":
        print("G return")
        return[
            Overall_SB,
            {'display': 'initial'},
            generate_edge_stats(edge_for_win),
            fav_table,
            [html.Div([html.Div("Streaks:", className='streak_header'), "{}".format(streak(dff)[0])], id='streak_display'),
             # html.Hr(id='streak_separator'),
             html.Div(id="form", children=[
                html.Div('Form Guide(W):', className='form_guide_header'),
                html.Ul(id='form_guide_ul',className='standings-table__form',
                     children=[generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))])])],
            [],
            [dff.to_json(orient='split')],
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'none'},
            odds_summary,
            {'display': 'initial'},
            [],
            {'display': 'none'},
            [],
            {'display': 'none'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            [],
            'flex-display fixedontop'
        ]
    elif len(searchterm)>1 and searchtype=="Horse":
        print("H return")
        return[
            Overall_SB,
            {'display': 'initial'},
            generate_edge_stats(edge_for_win),
            fav_table,
            [],
            [],
            [dff.to_json(orient='split')],
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'none'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'none'},
            {'display': 'none'},
            odds_summary,
            {'display': 'initial'},
            [],
            {'display': 'none'},
            [],
            {'display': 'none'},
            {'display': 'initial'},
            {'display': 'initial'},
            {'display': 'initial'},
            [],
            'flex-display fixedontop'
        ]


@app.callback(
    Output('media_trainer_split', 'children'),
    [Input('media_FST_trainer', 'value'),
    Input('store', 'children')],
    [State('identity', 'value')]
)
def update_media_trainer_split(predtype, frame, paper):
    if frame == [] or frame is None:
        print('somehow this is empty')
        raise PreventUpdate
    # print(frame)
    df = pd.read_json(frame[0], orient='split')
    if predtype in ['F', 'S', 'T']:
        df = df[df[paper]==predtype]
    else:
        df = df[(df[paper] == 'F') | (df[paper] == 'S') | (df[paper] == 'T')]
    dff = generate_tjstats_for_media(df, 'Trainer')
    return [
        html.Div(id='trainer_form'),
        html.Br(),
        generate_split_scoreboard_media(dff, "Trainer")[0],
        generate_split_scoreboard_media(dff, "Trainer")[1]
    ]

@app.callback(
    Output('media_jockey_split', 'children'),
    [Input('media_FST_jockey', 'value'),
    Input('store', 'children')],
    [State('identity', 'value')]
)
def update_media_jockey_split(predtype, frame, paper):
    if frame == [] or frame is None:
        raise PreventUpdate
    df = pd.read_json(frame[0], orient='split')
    print(predtype)
    if predtype in ['F', 'S', 'T']:
        print('WE are here')
        df = df[df[paper] == predtype]
    else:
        print('WE are there')
        df = df[(df[paper] == 'F') | (df[paper] == 'S') | (df[paper] == 'T')]
    dff = generate_tjstats_for_media(df, 'Jockey')
    return [
        html.Div(id='jockey_form'),
        html.Br(),
        generate_split_scoreboard_media(dff, "Jockey")[0],
        generate_split_scoreboard_media(dff, "Jockey")[1]
    ]


@app.callback(
    Output('trainer_form', 'children'),
    [Input('Trainertable', 'active_cell')],
    [State('Trainertable', 'derived_viewport_data'),
     State('store', 'children'),
     State('media_FST_trainer', 'value'),
     State('identity', 'value')]
)
def updatetrainer(active_cell,data,frame,predtype,paper):
    if active_cell is None:
        raise PreventUpdate
    df = pd.read_json(frame[0], orient='split')
    print(df['Season'].unique())
    if predtype in ['F', 'S', 'T']:
        df = df[df[paper]==predtype]
    else:
        df = df[(df[paper] == 'F') | (df[paper] == 'S') | (df[paper] == 'T')]
    row = active_cell['row']
    column = active_cell['column']
    column_id = active_cell['column_id']
    numero = 'first'
    if column_id in ['SHP', 'SHP%']:
        numero = 'second'
    elif column_id in ['THP', 'THP%']:
        numero = 'third'
    elif column_id in ['Plc', 'Plc%']:
        numero = 'place'
    else:
        numero = 'first'
    trainer = data[row]['Trainer']
    temp = df[df['Trainer']==trainer]
    print(column_id, trainer, len(temp))
    print(temp[['Centre', 'Season', 'Race No', 'Trainer', 'Jockey', 'Pl', paper]])
    return [
                html.Div('Form Guide({}):'.format(trainer), className='form_guide_header'),
                html.Ul(id='form_guide_ul_f',
                        className='form_guide_ul_win',
                        children=[generate_li_for_media_split(numero, i, j, n) for n, (i, j) in
                                  enumerate(zip(temp['Pl'][::-1],
                                                temp['Pl'][::-1]))]),
    ]



@app.callback(
    Output('jockey_form', 'children'),
    [Input('Jockeytable', 'active_cell')],
    [State('Jockeytable', 'derived_viewport_data'),
     State('store', 'children'),
     State('media_FST_jockey', 'value'),
     State('identity', 'value')]
)
def updatejockey(active_cell,data,frame,predtype,paper):
    if active_cell is None:
        raise PreventUpdate
    df = pd.read_json(frame[0], orient='split')
    print(predtype)
    if predtype in ['F', 'S', 'T']:
        df = df[df[paper]==predtype]
    else:
        df = df[(df[paper] == 'F') | (df[paper] == 'S') | (df[paper] == 'T')]
    print(len(df))
    row = active_cell['row']
    column = active_cell['column']
    column_id = active_cell['column_id']
    numero = 'first'
    if column_id in ['SHP', 'SHP%']:
        numero = 'second'
    elif column_id in ['THP', 'THP%']:
        numero = 'third'
    elif column_id in ['Plc', 'Plc%']:
        numero = 'place'
    else:
        numero = 'first'
    jockey = data[row]['Jockey']
    temp = df[df['Jockey'] == jockey]
    # print(jockey, len(temp))
    # print(temp[['Centre', 'Season', 'Race No', 'Trainer', 'Jockey', 'Pl', paper]])
    print(data)
    return [
        html.Div('Form Guide({}):'.format(jockey), className='form_guide_header'),
        html.Ul(id='form_guide_ul_f',
                className='form_guide_ul_win',
                children=[generate_li_for_media_split(numero, i, j, n) for n, (i, j) in
                          enumerate(zip(temp['Pl'][::-1],
                                        temp['Pl'][::-1]))])
    ]



@app.callback(
    [Output(component_id='longest_winning_streak_header', component_property='children'),
     Output(component_id='longest_losing_streak_header', component_property='children'),
     Output(component_id='longest_winning_streak_data', component_property='children'),
     Output(component_id='longest_losing_streak_data', component_property='children'),
     Output(component_id='form', component_property='children'),
     Output(component_id='second_button', component_property='n_clicks'),
     Output(component_id='third_button', component_property='n_clicks'),
     Output(component_id='place_button', component_property='n_clicks'),
     Output(component_id='form_guide_ul', component_property='className'),
     Output('streak_display','children')],
    [Input(component_id='first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State('store_media_race_stats', 'children'),
     State(component_id='second_button', component_property='n_clicks'),
     State(component_id='third_button', component_property='n_clicks'),
     State(component_id='place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')]
)
def update_form_win(clicks, frame, media_race_frame, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    if searchterm in media_list:
        # print("It is part of the media list")
        # print(media_race_frame)
        # x=media_race_frame[0]
        media_race_stats = pd.read_json(media_race_frame[0], orient='index')
        return ["LWS",
                "LLS",
                "{}".format(maxminstreaks_media(['L'], media_race_stats['Win'])[0]),
                "{}".format(maxminstreaks_media(['L'], media_race_stats['Win'])[1]),
                html.Div(id="form", children=[
                    html.Div('Form Guide(Any->W):', className='form_guide_header'),
                    html.Ul(id='form_guide_ul',
                            className='form_guide_ul_first',
                            children=[generate_li_for_media('allwin', i, j, n) for n, (i, j) in
                                      enumerate(zip(media_race_stats['Win'][::-1],
                                                    media_race_stats['Win'][::-1]))])]),
                0,
                0,
                0,
                'form_guide_ul_first',
                [html.Div("Streaks:", className='streak_header'),
                 "{}".format(streak_media('win', media_race_stats['Win'])[0])]
                ]
    # print(frame)
    dff = pd.read_json(frame[0], orient='split')
    if len(searchterm)==1:
        return ["LWS",
                "LLS",
                "{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [html.Div('Form Guide(W):', className='form_guide_header'),
                html.Ul(id='form_guide_ul', className='standings-table__form', children=[generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))])],
                0,
                0,
                0,
                'standings-table__form',
                dash.no_update
                ]
    else:
        return ["LWS",
                "LLS",
                "{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                'standings-table__form',
                dash.no_update
                ]


@app.callback(
    [Output(component_id='longest_winning_streak_header', component_property='children'),
     Output(component_id='longest_losing_streak_header', component_property='children'),
     Output(component_id='longest_winning_streak_data', component_property='children'),
     Output(component_id='longest_losing_streak_data', component_property='children'),
     Output(component_id='form', component_property='children'),
     Output(component_id='first_button', component_property='n_clicks'),
     Output(component_id='third_button', component_property='n_clicks'),
     Output(component_id='place_button', component_property='n_clicks'),
     Output(component_id='form_guide_ul', component_property='className'),
     Output('streak_display','children')],
    [Input(component_id='second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State('store_media_race_stats', 'children'),
     State(component_id='first_button', component_property='n_clicks'),
     State(component_id='third_button', component_property='n_clicks'),
     State(component_id='place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')]
)
def update_form_shp(clicks, frame, media_race_frame, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    if searchterm in media_list:
        print("It is part of the media list second")
        media_race_stats = pd.read_json(media_race_frame[0], orient='index')
        return ["LWS",
                "LLS",
                "{}".format(maxminstreaks_media(['L'], media_race_stats['SHP'])[0]),
                "{}".format(maxminstreaks_media(['L'], media_race_stats['SHP'])[1]),
                html.Div(id="form", children=[
                    html.Div('Form Guide(Any->SHP):', className='form_guide_header'),
                    html.Ul(id='form_guide_ul',
                            className='form_guide_ul_first',
                            children=[generate_li_for_media('allshp', i, j, n) for n, (i, j) in
                                      enumerate(zip(media_race_stats['SHP'][::-1],
                                                    media_race_stats['SHP'][::-1]))])]),
                0,
                0,
                0,
                'form_guide_ul_second',
                [html.Div("Streaks:", className='streak_header'),
                 "{}".format(streak_media('SHP', media_race_stats['SHP'])[0])]
                ]
    dff = pd.read_json(frame[0], orient='split')
    if len(searchterm) == 1:
        return ["LWS",
                "LLS",
                "{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [html.Div('Form Guide(SHP):', className='form_guide_header'),
                html.Ul(id='form_guide_ul', className='standings-table__form', children=[generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))])],
                0,
                0,
                0,
                'standings-table__form',
                dash.no_update
                ]
    else:
        return ["LWS",
                "LLS",
                "{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                'standings-table__form',
                dash.no_update
                ]


@app.callback(
    [Output(component_id='longest_winning_streak_header', component_property='children'),
     Output(component_id='longest_losing_streak_header', component_property='children'),
     Output(component_id='longest_winning_streak_data', component_property='children'),
     Output(component_id='longest_losing_streak_data', component_property='children'),
     Output(component_id='form', component_property='children'),
     Output(component_id='first_button', component_property='n_clicks'),
     Output(component_id='second_button', component_property='n_clicks'),
     Output(component_id='place_button', component_property='n_clicks'),
     Output(component_id='form_guide_ul', component_property='className'),
     Output('streak_display','children')],
    [Input(component_id='third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State('store_media_race_stats', 'children'),
     State(component_id='first_button', component_property='n_clicks'),
     State(component_id='second_button', component_property='n_clicks'),
     State(component_id='place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')]
)
def update_form_thp(clicks, frame, media_race_frame, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    if searchterm in media_list:
        print("It is part of the media list third")
        media_race_stats = pd.read_json(media_race_frame[0], orient='index')
        return ["LWS",
                "LLS",
                "{}".format(maxminstreaks_media(['L'], media_race_stats['THP'])[0]),
                "{}".format(maxminstreaks_media(['L'], media_race_stats['THP'])[1]),
                html.Div(id="form", children=[
                    html.Div('Form Guide(Any->THP):', className='form_guide_header'),
                    html.Ul(id='form_guide_ul',
                            className='form_guide_ul_first',
                            children=[generate_li_for_media('allthp', i, j, n) for n, (i, j) in
                                      enumerate(zip(media_race_stats['THP'][::-1],
                                                    media_race_stats['THP'][::-1]))])]),
                0,
                0,
                0,
                'form_guide_ul_third',
                [html.Div("Streaks:", className='streak_header'),
                 "{}".format(streak_media('THP', media_race_stats['THP'])[0])]
                ]
    dff = pd.read_json(frame[0], orient='split')
    if len(searchterm) == 1:
        return ["LWS",
                "LLS",
                "{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [html.Div('Form Guide(THP):', className='form_guide_header'),
                html.Ul(id='form_guide_ul', className='standings-table__form', children=[generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))])],
                0,
                0,
                0,
                'standings-table__form',
                dash.no_update
                ]
    else:
        return ["LWS",
                "LLS",
                "{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                'standings-table__form',
                dash.no_update
                ]


@app.callback(
    [Output(component_id='longest_winning_streak_header', component_property='children'),
     Output(component_id='longest_losing_streak_header', component_property='children'),
     Output(component_id='longest_winning_streak_data', component_property='children'),
     Output(component_id='longest_losing_streak_data', component_property='children'),
     Output(component_id='form', component_property='children'),
     Output(component_id='first_button', component_property='n_clicks'),
     Output(component_id='second_button', component_property='n_clicks'),
     Output(component_id='third_button', component_property='n_clicks'),
     Output(component_id='form_guide_ul', component_property='className'),
     Output('streak_display','children')],
    [Input(component_id='place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State('store_media_race_stats', 'children'),
     State(component_id='first_button', component_property='n_clicks'),
     State(component_id='second_button', component_property='n_clicks'),
     State(component_id='third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')]
)
def update_form_place(clicks, frame, media_race_frame, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    if searchterm in media_list:
        print("It is part of the media list place")
        media_race_stats = pd.read_json(media_race_frame[0], orient='index')
        return ["LWS",
                "LLS",
                "{}".format(streak_media('plc', media_race_stats['Plc'])[1]),
                "{}".format(streak_media('plc', media_race_stats['Plc'])[2]),
                html.Div(id="form", children=[
                    html.Div('Form Guide(Any->Plc):', className='form_guide_header'),
                    html.Ul(id='form_guide_ul',
                            className='form_guide_ul_first',
                            children=[generate_li_for_media('allplc', i, j, n) for n, (i, j) in
                                      enumerate(zip(media_race_stats['Plc'][::-1],
                                                    media_race_stats['Plc'][::-1]))])]),
                0,
                0,
                0,
                'form_guide_ul_place',
                [html.Div("Streaks:", className='streak_header'),
                 "{}".format(streak_media('plc', media_race_stats['Plc'])[0])]
                ]
    dff = pd.read_json(frame[0], orient='split')
    if len(searchterm) == 1:
        return ["LWS",
                "LLS",
                "{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [html.Div('Form Guide(Place):', className='form_guide_header'),
                html.Ul(id='form_guide_ul', className='standings-table__form', children=[generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))])],
                0,
                0,
                0,
                'standings-table__form',
                dash.no_update
                ]
    else:
        return ["LWS",
                "LLS",
                "{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                'standings-table__form',
                dash.no_update
                ]


@app.callback(
    [Output('form_guide_ul_f', 'className'),
     Output('first_longest_winning_streak_data', 'children'),
     Output('first_longest_losing_streak_data', 'children'),
     Output('first_second_button', 'n_clicks'),
     Output('first_third_button', 'n_clicks'),
     Output('first_place_button', 'n_clicks')],
    [Input('first_first_button', 'n_clicks')],
    [State('store', 'children'),
     State('store_media_race_stats', 'children'),
     State('first_second_button', 'n_clicks'),
     State('first_third_button', 'n_clicks'),
     State('first_place_button', 'n_clicks')]
)
def update_form_first_win(clicks, frame, media_race_frame, sclicks, tclicks, pclicks):
    if clicks == 0:
        raise PreventUpdate
    media_race_stats = pd.read_json(media_race_frame[0], orient='index')
    firstdf = media_race_stats[media_race_stats['one']==1]
    return ['form_guide_ul_win',
            "{}".format(maxminstreaks_media(['L', 'SHP', 'THP'], firstdf['firstplc'])[0]),
            "{}".format(maxminstreaks_media(['L', 'SHP', 'THP'], firstdf['firstplc'])[1]),
            0,
            0,
            0
            ]


@app.callback(
    [Output('form_guide_ul_f', 'className'),
     Output('first_longest_winning_streak_data', 'children'),
     Output('first_longest_losing_streak_data', 'children'),
     Output('first_first_button', 'n_clicks'),
     Output('first_third_button', 'n_clicks'),
     Output('first_place_button', 'n_clicks')],
    [Input('first_second_button', 'n_clicks')],
    [State('store', 'children'),
     State('store_media_race_stats', 'children'),
     State('first_first_button', 'n_clicks'),
     State('first_third_button', 'n_clicks'),
     State('first_place_button', 'n_clicks')]
)
def update_form_first_shp(clicks, frame, media_race_frame, fclicks, tclicks, pclicks):
    if clicks == 0:
        raise PreventUpdate
    media_race_stats = pd.read_json(media_race_frame[0], orient='index')
    firstdf = media_race_stats[media_race_stats['one'] == 1]
    return ['form_guide_ul_shp',
            "{}".format(maxminstreaks_media(['L', 'W', 'THP'], firstdf['firstplc'])[0]),
            "{}".format(maxminstreaks_media(['L', 'W', 'THP'], firstdf['firstplc'])[1]),
            0,
            0,
            0
            ]


@app.callback(
    [Output('form_guide_ul_f', 'className'),
     Output('first_longest_winning_streak_data', 'children'),
     Output('first_longest_losing_streak_data', 'children'),
     Output('first_second_button', 'n_clicks'),
     Output('first_first_button', 'n_clicks'),
     Output('first_place_button', 'n_clicks')],
    [Input('first_third_button', 'n_clicks')],
    [State('store', 'children'),
     State('store_media_race_stats', 'children'),
     State('first_second_button', 'n_clicks'),
     State('first_first_button', 'n_clicks'),
     State('first_place_button', 'n_clicks')]
)
def update_form_first_thp(clicks, frame, media_race_frame, sclicks, fclicks, pclicks):
    if clicks == 0:
        raise PreventUpdate
    media_race_stats = pd.read_json(media_race_frame[0], orient='index')
    firstdf = media_race_stats[media_race_stats['one'] == 1]
    return ['form_guide_ul_thp',
            "{}".format(maxminstreaks_media(['L', 'SHP', 'W'], firstdf['firstplc'])[0]),
            "{}".format(maxminstreaks_media(['L', 'SHP', 'W'], firstdf['firstplc'])[1]),
            0,
            0,
            0
            ]


@app.callback(
    [Output('form_guide_ul_f', 'className'),
     Output('first_longest_winning_streak_data', 'children'),
     Output('first_longest_losing_streak_data', 'children'),
     Output('first_second_button', 'n_clicks'),
     Output('first_third_button', 'n_clicks'),
     Output('first_first_button', 'n_clicks')],
    [Input('first_place_button', 'n_clicks')],
    [State('store', 'children'),
     State('store_media_race_stats', 'children'),
     State('first_second_button', 'n_clicks'),
     State('first_third_button', 'n_clicks'),
     State('first_first_button', 'n_clicks')]
)
def update_form_first_plc(clicks, frame, media_race_frame, sclicks, tclicks, fclicks):
    if clicks == 0:
        raise PreventUpdate
    media_race_stats = pd.read_json(media_race_frame[0], orient='index')
    firstdf = media_race_stats[media_race_stats['one'] == 1]
    return ['form_guide_ul_place',
            "{}".format(maxminstreaks_media(['L'], firstdf['firstplc'])[0]),
            "{}".format(maxminstreaks_media(['L'], firstdf['firstplc'])[1]),
            0,
            0,
            0
            ]



@app.callback(
    [Output('form_guide_ul_s', 'className'),
     Output('second_longest_winning_streak_data', 'children'),
     Output('second_longest_losing_streak_data', 'children'),
     Output('second_second_button', 'n_clicks'),
     Output('second_third_button', 'n_clicks'),
     Output('second_place_button', 'n_clicks')],
    [Input('second_first_button', 'n_clicks')],
    [State('store', 'children'),
     State('store_media_race_stats', 'children'),
     State('second_second_button', 'n_clicks'),
     State('second_third_button', 'n_clicks'),
     State('second_place_button', 'n_clicks')]
)
def update_form_second_win(clicks, frame, media_race_frame, sclicks, tclicks, pclicks):
    if clicks == 0:
        raise PreventUpdate
    media_race_stats = pd.read_json(media_race_frame[0], orient='index')
    seconddf = media_race_stats[media_race_stats['two'] == 1]
    return ['form_guide_ul_win',
            "{}".format(maxminstreaks_media(['L', 'SHP', 'THP'], seconddf['secondplc'])[0]),
            "{}".format(maxminstreaks_media(['L', 'SHP', 'THP'], seconddf['secondplc'])[1]),
            0,
            0,
            0
            ]


@app.callback(
    [Output('form_guide_ul_s', 'className'),
     Output('second_longest_winning_streak_data', 'children'),
     Output('second_longest_losing_streak_data', 'children'),
     Output('second_first_button', 'n_clicks'),
     Output('second_third_button', 'n_clicks'),
     Output('second_place_button', 'n_clicks')],
    [Input('second_second_button', 'n_clicks')],
    [State('store', 'children'),
     State('store_media_race_stats', 'children'),
     State('second_first_button', 'n_clicks'),
     State('second_third_button', 'n_clicks'),
     State('second_place_button', 'n_clicks')]
)
def update_form_second_shp(clicks, frame, media_race_frame, fclicks, tclicks, pclicks):
    if clicks == 0:
        raise PreventUpdate
    media_race_stats = pd.read_json(media_race_frame[0], orient='index')
    seconddf = media_race_stats[media_race_stats['two'] == 1]
    return ['form_guide_ul_shp',
            "{}".format(maxminstreaks_media(['L', 'W', 'THP'], seconddf['secondplc'])[0]),
            "{}".format(maxminstreaks_media(['L', 'W', 'THP'], seconddf['secondplc'])[1]),
            0,
            0,
            0
            ]


@app.callback(
    [Output('form_guide_ul_s', 'className'),
     Output('second_longest_winning_streak_data', 'children'),
     Output('second_longest_losing_streak_data', 'children'),
     Output('second_second_button', 'n_clicks'),
     Output('second_first_button', 'n_clicks'),
     Output('second_place_button', 'n_clicks')],
    [Input('second_third_button', 'n_clicks')],
    [State('store', 'children'),
     State('store_media_race_stats', 'children'),
     State('second_second_button', 'n_clicks'),
     State('second_first_button', 'n_clicks'),
     State('second_place_button', 'n_clicks')]
)
def update_form_second_thp(clicks, frame, media_race_frame, sclicks, fclicks, pclicks):
    if clicks == 0:
        raise PreventUpdate
    media_race_stats = pd.read_json(media_race_frame[0], orient='index')
    seconddf = media_race_stats[media_race_stats['two'] == 1]
    return ['form_guide_ul_thp',
            "{}".format(maxminstreaks_media(['L', 'SHP', 'W'], seconddf['secondplc'])[0]),
            "{}".format(maxminstreaks_media(['L', 'SHP', 'W'], seconddf['secondplc'])[1]),
            0,
            0,
            0
            ]


@app.callback(
    [Output('form_guide_ul_s', 'className'),
     Output('second_longest_winning_streak_data', 'children'),
     Output('second_longest_losing_streak_data', 'children'),
     Output('second_second_button', 'n_clicks'),
     Output('second_third_button', 'n_clicks'),
     Output('second_first_button', 'n_clicks')],
    [Input('second_place_button', 'n_clicks')],
    [State('store', 'children'),
     State('store_media_race_stats', 'children'),
     State('second_second_button', 'n_clicks'),
     State('second_third_button', 'n_clicks'),
     State('second_first_button', 'n_clicks')]
)
def update_form_second_plc(clicks, frame, media_race_frame, sclicks, tclicks, fclicks):
    if clicks == 0:
        raise PreventUpdate
    media_race_stats = pd.read_json(media_race_frame[0], orient='index')
    seconddf = media_race_stats[media_race_stats['two'] == 1]
    return ['form_guide_ul_place',
            "{}".format(maxminstreaks_media(['L'], seconddf['secondplc'])[0]),
            "{}".format(maxminstreaks_media(['L'], seconddf['secondplc'])[1]),
            0,
            0,
            0
            ]


@app.callback(
    [Output('form_guide_ul_t', 'className'),
     Output('third_longest_winning_streak_data', 'children'),
     Output('third_longest_losing_streak_data', 'children'),
     Output('third_second_button', 'n_clicks'),
     Output('third_third_button', 'n_clicks'),
     Output('third_place_button', 'n_clicks')],
    [Input('third_first_button', 'n_clicks')],
    [State('store', 'children'),
     State('store_media_race_stats', 'children'),
     State('third_second_button', 'n_clicks'),
     State('third_third_button', 'n_clicks'),
     State('third_place_button', 'n_clicks')]
)
def update_form_third_win(clicks, frame, media_race_frame, sclicks, tclicks, pclicks):
    if clicks == 0:
        raise PreventUpdate
    media_race_stats = pd.read_json(media_race_frame[0], orient='index')
    thirddf = media_race_stats[media_race_stats['three'] == 1]
    return ['form_guide_ul_win',
            "{}".format(maxminstreaks_media(['L', 'SHP', 'THP'], thirddf['thirdplc'])[0]),
            "{}".format(maxminstreaks_media(['L', 'SHP', 'THP'], thirddf['thirdplc'])[1]),
            0,
            0,
            0
            ]


@app.callback(
    [Output('form_guide_ul_t', 'className'),
     Output('third_longest_winning_streak_data', 'children'),
     Output('third_longest_losing_streak_data', 'children'),
     Output('third_first_button', 'n_clicks'),
     Output('third_third_button', 'n_clicks'),
     Output('third_place_button', 'n_clicks')],
    [Input('third_second_button', 'n_clicks')],
    [State('store', 'children'),
     State('store_media_race_stats', 'children'),
     State('third_first_button', 'n_clicks'),
     State('third_third_button', 'n_clicks'),
     State('third_place_button', 'n_clicks')]
)
def update_form_third_shp(clicks, frame, media_race_frame, fclicks, tclicks, pclicks):
    if clicks == 0:
        raise PreventUpdate
    media_race_stats = pd.read_json(media_race_frame[0], orient='index')
    thirddf = media_race_stats[media_race_stats['three'] == 1]
    return ['form_guide_ul_shp',
            "{}".format(maxminstreaks_media(['L', 'W', 'THP'], thirddf['thirdplc'])[0]),
            "{}".format(maxminstreaks_media(['L', 'W', 'THP'], thirddf['thirdplc'])[1]),
            0,
            0,
            0
            ]


@app.callback(
    [Output('form_guide_ul_t', 'className'),
     Output('third_longest_winning_streak_data', 'children'),
     Output('third_longest_losing_streak_data', 'children'),
     Output('third_second_button', 'n_clicks'),
     Output('third_first_button', 'n_clicks'),
     Output('third_place_button', 'n_clicks')],
    [Input('third_third_button', 'n_clicks')],
    [State('store', 'children'),
     State('store_media_race_stats', 'children'),
     State('third_second_button', 'n_clicks'),
     State('third_first_button', 'n_clicks'),
     State('third_place_button', 'n_clicks')]
)
def update_form_third_thp(clicks, frame, media_race_frame, sclicks, fclicks, pclicks):
    if clicks == 0:
        raise PreventUpdate
    media_race_stats = pd.read_json(media_race_frame[0], orient='index')
    thirddf = media_race_stats[media_race_stats['three'] == 1]
    return ['form_guide_ul_thp',
            "{}".format(maxminstreaks_media(['L', 'SHP', 'W'], thirddf['thirdplc'])[0]),
            "{}".format(maxminstreaks_media(['L', 'SHP', 'W'], thirddf['thirdplc'])[1]),
            0,
            0,
            0
            ]


@app.callback(
    [Output('form_guide_ul_t', 'className'),
     Output('third_longest_winning_streak_data', 'children'),
     Output('third_longest_losing_streak_data', 'children'),
     Output('third_second_button', 'n_clicks'),
     Output('third_third_button', 'n_clicks'),
     Output('third_first_button', 'n_clicks')],
    [Input('third_place_button', 'n_clicks')],
    [State('store', 'children'),
     State('store_media_race_stats', 'children'),
     State('third_second_button', 'n_clicks'),
     State('third_third_button', 'n_clicks'),
     State('third_first_button', 'n_clicks')]
)
def update_form_third_plc(clicks, frame, media_race_frame, sclicks, tclicks, fclicks):
    if clicks == 0:
        raise PreventUpdate
    media_race_stats = pd.read_json(media_race_frame[0], orient='index')
    thirddf = media_race_stats[media_race_stats['three'] == 1]
    return ['form_guide_ul_place',
            "{}".format(maxminstreaks_media(['L'], thirddf['thirdplc'])[0]),
            "{}".format(maxminstreaks_media(['L'], thirddf['thirdplc'])[1]),
            0,
            0,
            0
            ]


# @app.callback(
#     [Output(component_id='longest_winning_streak_header', component_property='children'),
#      Output(component_id='longest_losing_streak_header', component_property='children'),
#      Output(component_id='longest_winning_streak_data', component_property='children'),
#      Output(component_id='longest_losing_streak_data', component_property='children'),
#      Output(component_id='form', component_property='children'),
#      Output(component_id='second_button', component_property='n_clicks'),
#      Output(component_id='third_button', component_property='n_clicks'),
#      Output(component_id='place_button', component_property='n_clicks')],
#     [Input(component_id='first_button', component_property='n_clicks')],
#     [State(component_id='store', component_property='children'),
#      State(component_id='second_button', component_property='n_clicks'),
#      State(component_id='third_button', component_property='n_clicks'),
#      State(component_id='place_button', component_property='n_clicks'),
#      State(component_id='identity', component_property='value')]
# )
# def update_form_win(clicks, frame, sclicks, tclicks, pclicks, searchterm):
#     if clicks == 0:
#         raise PreventUpdate
#     dff = pd.read_json(frame[0], orient='split')
#     if len(searchterm)==1:
#         return ["LWS(WIN)",
#                 "LLS(WIN)",
#                 "{}".format(maxminstreaks(1, dff)[0]),
#                 "{}".format(maxminstreaks(1, dff)[1]),
#                 [html.Div('Form Guide(W):', className='form_guide_header'),
#                 html.Ul(className='standings-table__form', children=[generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))])],
#                 0,
#                 0,
#                 0
#                 ]
#     else:
#         return ["LWS(WIN)",
#                 "LLS(WIN)",
#                 "{}".format(maxminstreaks(1, dff)[0]),
#                 "{}".format(maxminstreaks(1, dff)[1]),
#                 [],
#                 0,
#                 0,
#                 0
#                 ]
#
#
# @app.callback(
#     [Output(component_id='longest_winning_streak_header', component_property='children'),
#      Output(component_id='longest_losing_streak_header', component_property='children'),
#      Output(component_id='longest_winning_streak_data', component_property='children'),
#      Output(component_id='longest_losing_streak_data', component_property='children'),
#      Output(component_id='form', component_property='children'),
#      Output(component_id='first_button', component_property='n_clicks'),
#      Output(component_id='third_button', component_property='n_clicks'),
#      Output(component_id='place_button', component_property='n_clicks')],
#     [Input(component_id='second_button', component_property='n_clicks')],
#     [State(component_id='store', component_property='children'),
#      State(component_id='first_button', component_property='n_clicks'),
#      State(component_id='third_button', component_property='n_clicks'),
#      State(component_id='place_button', component_property='n_clicks'),
#      State(component_id='identity', component_property='value')]
# )
# def update_form_shp(clicks, frame, fclicks, tclicks, pclicks, searchterm):
#     if clicks == 0:
#         raise PreventUpdate
#     dff = pd.read_json(frame[0], orient='split')
#     if len(searchterm) == 1:
#         return ["LWS(SHP)",
#                 "LLS(SHP)",
#                 "{}".format(maxminstreaks(2, dff)[0]),
#                 "{}".format(maxminstreaks(2, dff)[1]),
#                 [html.Div('Form Guide(SHP):', className='form_guide_header'),
#                 html.Ul(className='standings-table__form', children=[generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))])],
#                 0,
#                 0,
#                 0
#                 ]
#     else:
#         return ["LWS(SHP)",
#                 "LLS(SHP)",
#                 "{}".format(maxminstreaks(2, dff)[0]),
#                 "{}".format(maxminstreaks(2, dff)[1]),
#                 [],
#                 0,
#                 0,
#                 0
#                 ]
#
#
# @app.callback(
#     [Output(component_id='longest_winning_streak_header', component_property='children'),
#      Output(component_id='longest_losing_streak_header', component_property='children'),
#      Output(component_id='longest_winning_streak_data', component_property='children'),
#      Output(component_id='longest_losing_streak_data', component_property='children'),
#      Output(component_id='form', component_property='children'),
#      Output(component_id='first_button', component_property='n_clicks'),
#      Output(component_id='second_button', component_property='n_clicks'),
#      Output(component_id='place_button', component_property='n_clicks')],
#     [Input(component_id='third_button', component_property='n_clicks')],
#     [State(component_id='store', component_property='children'),
#      State(component_id='first_button', component_property='n_clicks'),
#      State(component_id='second_button', component_property='n_clicks'),
#      State(component_id='place_button', component_property='n_clicks'),
#      State(component_id='identity', component_property='value')]
# )
# def update_form_thp(clicks, frame, fclicks, sclicks, pclicks, searchterm):
#     if clicks == 0:
#         raise PreventUpdate
#     dff = pd.read_json(frame[0], orient='split')
#     if len(searchterm) == 1:
#         return ["LWS(THP)",
#                 "LLS(THP)",
#                 "{}".format(maxminstreaks(3, dff)[0]),
#                 "{}".format(maxminstreaks(3, dff)[1]),
#                 [html.Div('Form Guide(THP):', className='form_guide_header'),
#                 html.Ul(className='standings-table__form', children=[generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))])],
#                 0,
#                 0,
#                 0
#                 ]
#     else:
#         return ["LWS(THP)",
#                 "LLS(THP)",
#                 "{}".format(maxminstreaks(3, dff)[0]),
#                 "{}".format(maxminstreaks(3, dff)[1]),
#                 [],
#                 0,
#                 0,
#                 0
#                 ]
#
#
# @app.callback(
#     [Output(component_id='longest_winning_streak_header', component_property='children'),
#      Output(component_id='longest_losing_streak_header', component_property='children'),
#      Output(component_id='longest_winning_streak_data', component_property='children'),
#      Output(component_id='longest_losing_streak_data', component_property='children'),
#      Output(component_id='form', component_property='children'),
#      Output(component_id='first_button', component_property='n_clicks'),
#      Output(component_id='second_button', component_property='n_clicks'),
#      Output(component_id='third_button', component_property='n_clicks')],
#     [Input(component_id='place_button', component_property='n_clicks')],
#     [State(component_id='store', component_property='children'),
#      State(component_id='first_button', component_property='n_clicks'),
#      State(component_id='second_button', component_property='n_clicks'),
#      State(component_id='third_button', component_property='n_clicks'),
#      State(component_id='identity', component_property='value')]
# )
# def update_form_place(clicks, frame, fclicks, sclicks, tclicks, searchterm):
#     if clicks == 0:
#         raise PreventUpdate
#     dff = pd.read_json(frame[0], orient='split')
#     if len(searchterm) == 1:
#         return ["LWS(Plc)",
#                 "LLS(Plc)",
#                 "{}".format(streak(dff)[1]),
#                 "{}".format(streak(dff)[2]),
#                 [html.Div('Form Guide(Place):', className='form_guide_header'),
#                 html.Ul(className='standings-table__form', children=[generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))])],
#                 0,
#                 0,
#                 0
#                 ]
#     else:
#         return ["LWS(Plc)",
#                 "LLS(Plc)",
#                 "{}".format(streak(dff)[1]),
#                 "{}".format(streak(dff)[2]),
#                 [],
#                 0,
#                 0,
#                 0
#                 ]


######################################################################################################################

@app.callback(
    [Output(component_id='F0_lws', component_property='children'),
     Output(component_id='F0_lls', component_property='children'),
     Output(component_id='F0_form', component_property='children'),
     Output(component_id='F0_second_button', component_property='n_clicks'),
     Output(component_id='F0_third_button', component_property='n_clicks'),
     Output(component_id='F0_place_button', component_property='n_clicks'),
     Output(component_id='F0_split', component_property='children')],
    [Input(component_id='F0_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F0_second_button', component_property='n_clicks'),
     State(component_id='F0_third_button', component_property='n_clicks'),
     State(component_id='F0_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F0_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type='Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favweight, favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F0']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F0'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F0_lws', component_property='children'),
     Output(component_id='F0_lls', component_property='children'),
     Output(component_id='F0_form', component_property='children'),
     Output(component_id='F0_first_button', component_property='n_clicks'),
     Output(component_id='F0_third_button', component_property='n_clicks'),
     Output(component_id='F0_place_button', component_property='n_clicks'),
     Output(component_id='F0_split', component_property='children')],
    [Input(component_id='F0_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F0_first_button', component_property='n_clicks'),
     State(component_id='F0_third_button', component_property='n_clicks'),
     State(component_id='F0_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F0_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F0']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F0'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F0_lws', component_property='children'),
     Output(component_id='F0_lls', component_property='children'),
     Output(component_id='F0_form', component_property='children'),
     Output(component_id='F0_first_button', component_property='n_clicks'),
     Output(component_id='F0_second_button', component_property='n_clicks'),
     Output(component_id='F0_place_button', component_property='n_clicks'),
     Output(component_id='F0_split', component_property='children')],
    [Input(component_id='F0_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F0_first_button', component_property='n_clicks'),
     State(component_id='F0_second_button', component_property='n_clicks'),
     State(component_id='F0_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F0_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F0']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F0'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F0_lws', component_property='children'),
     Output(component_id='F0_lls', component_property='children'),
     Output(component_id='F0_form', component_property='children'),
     Output(component_id='F0_first_button', component_property='n_clicks'),
     Output(component_id='F0_second_button', component_property='n_clicks'),
     Output(component_id='F0_third_button', component_property='n_clicks'),
     Output(component_id='F0_split', component_property='children')],
    [Input(component_id='F0_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F0_first_button', component_property='n_clicks'),
     State(component_id='F0_second_button', component_property='n_clicks'),
     State(component_id='F0_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F0_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F0']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F0'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

##############################################################################################################

@app.callback(
    [Output(component_id='F1_lws', component_property='children'),
     Output(component_id='F1_lls', component_property='children'),
     Output(component_id='F1_form', component_property='children'),
     Output(component_id='F1_second_button', component_property='n_clicks'),
     Output(component_id='F1_third_button', component_property='n_clicks'),
     Output(component_id='F1_place_button', component_property='n_clicks'),
     Output(component_id='F1_split', component_property='children')],
    [Input(component_id='F1_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F1_second_button', component_property='n_clicks'),
     State(component_id='F1_third_button', component_property='n_clicks'),
     State(component_id='F1_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F1_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F1']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F1'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F1_lws', component_property='children'),
     Output(component_id='F1_lls', component_property='children'),
     Output(component_id='F1_form', component_property='children'),
     Output(component_id='F1_first_button', component_property='n_clicks'),
     Output(component_id='F1_third_button', component_property='n_clicks'),
     Output(component_id='F1_place_button', component_property='n_clicks'),
     Output(component_id='F1_split', component_property='children')],
    [Input(component_id='F1_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F1_first_button', component_property='n_clicks'),
     State(component_id='F1_third_button', component_property='n_clicks'),
     State(component_id='F1_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F1_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F1']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F1'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F1_lws', component_property='children'),
     Output(component_id='F1_lls', component_property='children'),
     Output(component_id='F1_form', component_property='children'),
     Output(component_id='F1_first_button', component_property='n_clicks'),
     Output(component_id='F1_second_button', component_property='n_clicks'),
     Output(component_id='F1_place_button', component_property='n_clicks'),
     Output(component_id='F1_split', component_property='children')],
    [Input(component_id='F1_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F1_first_button', component_property='n_clicks'),
     State(component_id='F1_second_button', component_property='n_clicks'),
     State(component_id='F1_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F1_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F1']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F1'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F1_lws', component_property='children'),
     Output(component_id='F1_lls', component_property='children'),
     Output(component_id='F1_form', component_property='children'),
     Output(component_id='F1_first_button', component_property='n_clicks'),
     Output(component_id='F1_second_button', component_property='n_clicks'),
     Output(component_id='F1_third_button', component_property='n_clicks'),
     Output(component_id='F1_split', component_property='children')],
    [Input(component_id='F1_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F1_first_button', component_property='n_clicks'),
     State(component_id='F1_second_button', component_property='n_clicks'),
     State(component_id='F1_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F1_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F1']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F1'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F2_lws', component_property='children'),
     Output(component_id='F2_lls', component_property='children'),
     Output(component_id='F2_form', component_property='children'),
     Output(component_id='F2_second_button', component_property='n_clicks'),
     Output(component_id='F2_third_button', component_property='n_clicks'),
     Output(component_id='F2_place_button', component_property='n_clicks'),
     Output(component_id='F2_split', component_property='children')],
    [Input(component_id='F2_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F2_second_button', component_property='n_clicks'),
     State(component_id='F2_third_button', component_property='n_clicks'),
     State(component_id='F2_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F2_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F2']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F2'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F2_lws', component_property='children'),
     Output(component_id='F2_lls', component_property='children'),
     Output(component_id='F2_form', component_property='children'),
     Output(component_id='F2_first_button', component_property='n_clicks'),
     Output(component_id='F2_third_button', component_property='n_clicks'),
     Output(component_id='F2_place_button', component_property='n_clicks'),
     Output(component_id='F2_split', component_property='children')],
    [Input(component_id='F2_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F2_first_button', component_property='n_clicks'),
     State(component_id='F2_third_button', component_property='n_clicks'),
     State(component_id='F2_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F2_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F2']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F2'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F2_lws', component_property='children'),
     Output(component_id='F2_lls', component_property='children'),
     Output(component_id='F2_form', component_property='children'),
     Output(component_id='F2_first_button', component_property='n_clicks'),
     Output(component_id='F2_second_button', component_property='n_clicks'),
     Output(component_id='F2_place_button', component_property='n_clicks'),
     Output(component_id='F2_split', component_property='children')],
    [Input(component_id='F2_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F2_first_button', component_property='n_clicks'),
     State(component_id='F2_second_button', component_property='n_clicks'),
     State(component_id='F2_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F2_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F2']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F2'], split_by_type, 3)
                ]
    else:
        return [[],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F2_lws', component_property='children'),
     Output(component_id='F2_lls', component_property='children'),
     Output(component_id='F2_form', component_property='children'),
     Output(component_id='F2_first_button', component_property='n_clicks'),
     Output(component_id='F2_second_button', component_property='n_clicks'),
     Output(component_id='F2_third_button', component_property='n_clicks'),
     Output(component_id='F2_split', component_property='children')],
    [Input(component_id='F2_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F2_first_button', component_property='n_clicks'),
     State(component_id='F2_second_button', component_property='n_clicks'),
     State(component_id='F2_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F2_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F2']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F2'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F3_lws', component_property='children'),
     Output(component_id='F3_lls', component_property='children'),
     Output(component_id='F3_form', component_property='children'),
     Output(component_id='F3_second_button', component_property='n_clicks'),
     Output(component_id='F3_third_button', component_property='n_clicks'),
     Output(component_id='F3_place_button', component_property='n_clicks'),
     Output(component_id='F3_split', component_property='children')],
    [Input(component_id='F3_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F3_second_button', component_property='n_clicks'),
     State(component_id='F3_third_button', component_property='n_clicks'),
     State(component_id='F3_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F3_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F3']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F3'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F3_lws', component_property='children'),
     Output(component_id='F3_lls', component_property='children'),
     Output(component_id='F3_form', component_property='children'),
     Output(component_id='F3_first_button', component_property='n_clicks'),
     Output(component_id='F3_third_button', component_property='n_clicks'),
     Output(component_id='F3_place_button', component_property='n_clicks'),
     Output(component_id='F3_split', component_property='children')],
    [Input(component_id='F3_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F3_first_button', component_property='n_clicks'),
     State(component_id='F3_third_button', component_property='n_clicks'),
     State(component_id='F3_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F3_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F3']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F3'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F3_lws', component_property='children'),
     Output(component_id='F3_lls', component_property='children'),
     Output(component_id='F3_form', component_property='children'),
     Output(component_id='F3_first_button', component_property='n_clicks'),
     Output(component_id='F3_second_button', component_property='n_clicks'),
     Output(component_id='F3_place_button', component_property='n_clicks'),
     Output(component_id='F3_split', component_property='children')],
    [Input(component_id='F3_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F3_first_button', component_property='n_clicks'),
     State(component_id='F3_second_button', component_property='n_clicks'),
     State(component_id='F3_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F3_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F3']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F3'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F3_lws', component_property='children'),
     Output(component_id='F3_lls', component_property='children'),
     Output(component_id='F3_form', component_property='children'),
     Output(component_id='F3_first_button', component_property='n_clicks'),
     Output(component_id='F3_second_button', component_property='n_clicks'),
     Output(component_id='F3_third_button', component_property='n_clicks'),
     Output(component_id='F3_split', component_property='children')],
    [Input(component_id='F3_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F3_first_button', component_property='n_clicks'),
     State(component_id='F3_second_button', component_property='n_clicks'),
     State(component_id='F3_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F3_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F3']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F3'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F4_lws', component_property='children'),
     Output(component_id='F4_lls', component_property='children'),
     Output(component_id='F4_form', component_property='children'),
     Output(component_id='F4_second_button', component_property='n_clicks'),
     Output(component_id='F4_third_button', component_property='n_clicks'),
     Output(component_id='F4_place_button', component_property='n_clicks'),
     Output(component_id='F4_split', component_property='children')],
    [Input(component_id='F4_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F4_second_button', component_property='n_clicks'),
     State(component_id='F4_third_button', component_property='n_clicks'),
     State(component_id='F4_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F4_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F4']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F4'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F4_lws', component_property='children'),
     Output(component_id='F4_lls', component_property='children'),
     Output(component_id='F4_form', component_property='children'),
     Output(component_id='F4_first_button', component_property='n_clicks'),
     Output(component_id='F4_third_button', component_property='n_clicks'),
     Output(component_id='F4_place_button', component_property='n_clicks'),
     Output(component_id='F4_split', component_property='children')],
    [Input(component_id='F4_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F4_first_button', component_property='n_clicks'),
     State(component_id='F4_third_button', component_property='n_clicks'),
     State(component_id='F4_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F4_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F4']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F4'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F4_lws', component_property='children'),
     Output(component_id='F4_lls', component_property='children'),
     Output(component_id='F4_form', component_property='children'),
     Output(component_id='F4_first_button', component_property='n_clicks'),
     Output(component_id='F4_second_button', component_property='n_clicks'),
     Output(component_id='F4_place_button', component_property='n_clicks'),
     Output(component_id='F4_split', component_property='children')],
    [Input(component_id='F4_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F4_first_button', component_property='n_clicks'),
     State(component_id='F4_second_button', component_property='n_clicks'),
     State(component_id='F4_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F4_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F4']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F4'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F4_lws', component_property='children'),
     Output(component_id='F4_lls', component_property='children'),
     Output(component_id='F4_form', component_property='children'),
     Output(component_id='F4_first_button', component_property='n_clicks'),
     Output(component_id='F4_second_button', component_property='n_clicks'),
     Output(component_id='F4_third_button', component_property='n_clicks'),
     Output(component_id='F4_split', component_property='children')],
    [Input(component_id='F4_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F4_first_button', component_property='n_clicks'),
     State(component_id='F4_second_button', component_property='n_clicks'),
     State(component_id='F4_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F4_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F4']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F4'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F5_lws', component_property='children'),
     Output(component_id='F5_lls', component_property='children'),
     Output(component_id='F5_form', component_property='children'),
     Output(component_id='F5_second_button', component_property='n_clicks'),
     Output(component_id='F5_third_button', component_property='n_clicks'),
     Output(component_id='F5_place_button', component_property='n_clicks'),
     Output(component_id='F5_split', component_property='children')],
    [Input(component_id='F5_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F5_second_button', component_property='n_clicks'),
     State(component_id='F5_third_button', component_property='n_clicks'),
     State(component_id='F5_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F5_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F5']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F5'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F5_lws', component_property='children'),
     Output(component_id='F5_lls', component_property='children'),
     Output(component_id='F5_form', component_property='children'),
     Output(component_id='F5_first_button', component_property='n_clicks'),
     Output(component_id='F5_third_button', component_property='n_clicks'),
     Output(component_id='F5_place_button', component_property='n_clicks'),
     Output(component_id='F5_split', component_property='children')],
    [Input(component_id='F5_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F5_first_button', component_property='n_clicks'),
     State(component_id='F5_third_button', component_property='n_clicks'),
     State(component_id='F5_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F5_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F5']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F5'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F5_lws', component_property='children'),
     Output(component_id='F5_lls', component_property='children'),
     Output(component_id='F5_form', component_property='children'),
     Output(component_id='F5_first_button', component_property='n_clicks'),
     Output(component_id='F5_second_button', component_property='n_clicks'),
     Output(component_id='F5_place_button', component_property='n_clicks'),
     Output(component_id='F5_split', component_property='children')],
    [Input(component_id='F5_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F5_first_button', component_property='n_clicks'),
     State(component_id='F5_second_button', component_property='n_clicks'),
     State(component_id='F5_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F5_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F5']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F5'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F5_lws', component_property='children'),
     Output(component_id='F5_lls', component_property='children'),
     Output(component_id='F5_form', component_property='children'),
     Output(component_id='F5_first_button', component_property='n_clicks'),
     Output(component_id='F5_second_button', component_property='n_clicks'),
     Output(component_id='F5_third_button', component_property='n_clicks'),
     Output(component_id='F5_split', component_property='children')],
    [Input(component_id='F5_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F5_first_button', component_property='n_clicks'),
     State(component_id='F5_second_button', component_property='n_clicks'),
     State(component_id='F5_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F5_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F5']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F5'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F6_lws', component_property='children'),
     Output(component_id='F6_lls', component_property='children'),
     Output(component_id='F6_form', component_property='children'),
     Output(component_id='F6_second_button', component_property='n_clicks'),
     Output(component_id='F6_third_button', component_property='n_clicks'),
     Output(component_id='F6_place_button', component_property='n_clicks'),
     Output(component_id='F6_split', component_property='children')],
    [Input(component_id='F6_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F6_second_button', component_property='n_clicks'),
     State(component_id='F6_third_button', component_property='n_clicks'),
     State(component_id='F6_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F6_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F6']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F6'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F6_lws', component_property='children'),
     Output(component_id='F6_lls', component_property='children'),
     Output(component_id='F6_form', component_property='children'),
     Output(component_id='F6_first_button', component_property='n_clicks'),
     Output(component_id='F6_third_button', component_property='n_clicks'),
     Output(component_id='F6_place_button', component_property='n_clicks'),
     Output(component_id='F6_split', component_property='children')],
    [Input(component_id='F6_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F6_first_button', component_property='n_clicks'),
     State(component_id='F6_third_button', component_property='n_clicks'),
     State(component_id='F6_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F6_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F6']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F6'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F6_lws', component_property='children'),
     Output(component_id='F6_lls', component_property='children'),
     Output(component_id='F6_form', component_property='children'),
     Output(component_id='F6_first_button', component_property='n_clicks'),
     Output(component_id='F6_second_button', component_property='n_clicks'),
     Output(component_id='F6_place_button', component_property='n_clicks'),
     Output(component_id='F6_split', component_property='children')],
    [Input(component_id='F6_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F6_first_button', component_property='n_clicks'),
     State(component_id='F6_second_button', component_property='n_clicks'),
     State(component_id='F6_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F6_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F6']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F6'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F6_lws', component_property='children'),
     Output(component_id='F6_lls', component_property='children'),
     Output(component_id='F6_form', component_property='children'),
     Output(component_id='F6_first_button', component_property='n_clicks'),
     Output(component_id='F6_second_button', component_property='n_clicks'),
     Output(component_id='F6_third_button', component_property='n_clicks'),
     Output(component_id='F6_split', component_property='children')],
    [Input(component_id='F6_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F6_first_button', component_property='n_clicks'),
     State(component_id='F6_second_button', component_property='n_clicks'),
     State(component_id='F6_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F6_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F6']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F6'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F7_lws', component_property='children'),
     Output(component_id='F7_lls', component_property='children'),
     Output(component_id='F7_form', component_property='children'),
     Output(component_id='F7_second_button', component_property='n_clicks'),
     Output(component_id='F7_third_button', component_property='n_clicks'),
     Output(component_id='F7_place_button', component_property='n_clicks'),
     Output(component_id='F7_split', component_property='children')],
    [Input(component_id='F7_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F7_second_button', component_property='n_clicks'),
     State(component_id='F7_third_button', component_property='n_clicks'),
     State(component_id='F7_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F7_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F7']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F7'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F7_lws', component_property='children'),
     Output(component_id='F7_lls', component_property='children'),
     Output(component_id='F7_form', component_property='children'),
     Output(component_id='F7_first_button', component_property='n_clicks'),
     Output(component_id='F7_third_button', component_property='n_clicks'),
     Output(component_id='F7_place_button', component_property='n_clicks'),
     Output(component_id='F7_split', component_property='children')],
    [Input(component_id='F7_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F7_first_button', component_property='n_clicks'),
     State(component_id='F7_third_button', component_property='n_clicks'),
     State(component_id='F7_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F7_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F7']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F7'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F7_lws', component_property='children'),
     Output(component_id='F7_lls', component_property='children'),
     Output(component_id='F7_form', component_property='children'),
     Output(component_id='F7_first_button', component_property='n_clicks'),
     Output(component_id='F7_second_button', component_property='n_clicks'),
     Output(component_id='F7_place_button', component_property='n_clicks'),
     Output(component_id='F7_split', component_property='children')],
    [Input(component_id='F7_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F7_first_button', component_property='n_clicks'),
     State(component_id='F7_second_button', component_property='n_clicks'),
     State(component_id='F7_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F7_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F7']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F7'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F7_lws', component_property='children'),
     Output(component_id='F7_lls', component_property='children'),
     Output(component_id='F7_form', component_property='children'),
     Output(component_id='F7_first_button', component_property='n_clicks'),
     Output(component_id='F7_second_button', component_property='n_clicks'),
     Output(component_id='F7_third_button', component_property='n_clicks'),
     Output(component_id='F7_split', component_property='children')],
    [Input(component_id='F7_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F7_first_button', component_property='n_clicks'),
     State(component_id='F7_second_button', component_property='n_clicks'),
     State(component_id='F7_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F7_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F7']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F7'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F8_lws', component_property='children'),
     Output(component_id='F8_lls', component_property='children'),
     Output(component_id='F8_form', component_property='children'),
     Output(component_id='F8_second_button', component_property='n_clicks'),
     Output(component_id='F8_third_button', component_property='n_clicks'),
     Output(component_id='F8_place_button', component_property='n_clicks'),
     Output(component_id='F8_split', component_property='children')],
    [Input(component_id='F8_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F8_second_button', component_property='n_clicks'),
     State(component_id='F8_third_button', component_property='n_clicks'),
     State(component_id='F8_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F8_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F8']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F8'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F8_lws', component_property='children'),
     Output(component_id='F8_lls', component_property='children'),
     Output(component_id='F8_form', component_property='children'),
     Output(component_id='F8_first_button', component_property='n_clicks'),
     Output(component_id='F8_third_button', component_property='n_clicks'),
     Output(component_id='F8_place_button', component_property='n_clicks'),
     Output(component_id='F8_split', component_property='children')],
    [Input(component_id='F8_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F8_first_button', component_property='n_clicks'),
     State(component_id='F8_third_button', component_property='n_clicks'),
     State(component_id='F8_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F8_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F8']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F8'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F8_lws', component_property='children'),
     Output(component_id='F8_lls', component_property='children'),
     Output(component_id='F8_form', component_property='children'),
     Output(component_id='F8_first_button', component_property='n_clicks'),
     Output(component_id='F8_second_button', component_property='n_clicks'),
     Output(component_id='F8_place_button', component_property='n_clicks'),
     Output(component_id='F8_split', component_property='children')],
    [Input(component_id='F8_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F8_first_button', component_property='n_clicks'),
     State(component_id='F8_second_button', component_property='n_clicks'),
     State(component_id='F8_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F8_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F8']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F8'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F8_lws', component_property='children'),
     Output(component_id='F8_lls', component_property='children'),
     Output(component_id='F8_form', component_property='children'),
     Output(component_id='F8_first_button', component_property='n_clicks'),
     Output(component_id='F8_second_button', component_property='n_clicks'),
     Output(component_id='F8_third_button', component_property='n_clicks'),
     Output(component_id='F8_split', component_property='children')],
    [Input(component_id='F8_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F8_first_button', component_property='n_clicks'),
     State(component_id='F8_second_button', component_property='n_clicks'),
     State(component_id='F8_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F8_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F8']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F8'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F9_lws', component_property='children'),
     Output(component_id='F9_lls', component_property='children'),
     Output(component_id='F9_form', component_property='children'),
     Output(component_id='F9_second_button', component_property='n_clicks'),
     Output(component_id='F9_third_button', component_property='n_clicks'),
     Output(component_id='F9_place_button', component_property='n_clicks'),
     Output(component_id='F9_split', component_property='children')],
    [Input(component_id='F9_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F9_second_button', component_property='n_clicks'),
     State(component_id='F9_third_button', component_property='n_clicks'),
     State(component_id='F9_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F9_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F9']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F9'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F9_lws', component_property='children'),
     Output(component_id='F9_lls', component_property='children'),
     Output(component_id='F9_form', component_property='children'),
     Output(component_id='F9_first_button', component_property='n_clicks'),
     Output(component_id='F9_third_button', component_property='n_clicks'),
     Output(component_id='F9_place_button', component_property='n_clicks'),
     Output(component_id='F9_split', component_property='children')],
    [Input(component_id='F9_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F9_first_button', component_property='n_clicks'),
     State(component_id='F9_third_button', component_property='n_clicks'),
     State(component_id='F9_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F9_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F9']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F9'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F9_lws', component_property='children'),
     Output(component_id='F9_lls', component_property='children'),
     Output(component_id='F9_form', component_property='children'),
     Output(component_id='F9_first_button', component_property='n_clicks'),
     Output(component_id='F9_second_button', component_property='n_clicks'),
     Output(component_id='F9_place_button', component_property='n_clicks'),
     Output(component_id='F9_split', component_property='children')],
    [Input(component_id='F9_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F9_first_button', component_property='n_clicks'),
     State(component_id='F9_second_button', component_property='n_clicks'),
     State(component_id='F9_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F9_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F9']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F9'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F9_lws', component_property='children'),
     Output(component_id='F9_lls', component_property='children'),
     Output(component_id='F9_form', component_property='children'),
     Output(component_id='F9_first_button', component_property='n_clicks'),
     Output(component_id='F9_second_button', component_property='n_clicks'),
     Output(component_id='F9_third_button', component_property='n_clicks'),
     Output(component_id='F9_split', component_property='children')],
    [Input(component_id='F9_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F9_first_button', component_property='n_clicks'),
     State(component_id='F9_second_button', component_property='n_clicks'),
     State(component_id='F9_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F9_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F9']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F9'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F10_lws', component_property='children'),
     Output(component_id='F10_lls', component_property='children'),
     Output(component_id='F10_form', component_property='children'),
     Output(component_id='F10_second_button', component_property='n_clicks'),
     Output(component_id='F10_third_button', component_property='n_clicks'),
     Output(component_id='F10_place_button', component_property='n_clicks'),
     Output(component_id='F10_split', component_property='children')],
    [Input(component_id='F10_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F10_second_button', component_property='n_clicks'),
     State(component_id='F10_third_button', component_property='n_clicks'),
     State(component_id='F10_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F10_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F10']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F10'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F10_lws', component_property='children'),
     Output(component_id='F10_lls', component_property='children'),
     Output(component_id='F10_form', component_property='children'),
     Output(component_id='F10_first_button', component_property='n_clicks'),
     Output(component_id='F10_third_button', component_property='n_clicks'),
     Output(component_id='F10_place_button', component_property='n_clicks'),
     Output(component_id='F10_split', component_property='children')],
    [Input(component_id='F10_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F10_first_button', component_property='n_clicks'),
     State(component_id='F10_third_button', component_property='n_clicks'),
     State(component_id='F10_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F10_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F10']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F10'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F10_lws', component_property='children'),
     Output(component_id='F10_lls', component_property='children'),
     Output(component_id='F10_form', component_property='children'),
     Output(component_id='F10_first_button', component_property='n_clicks'),
     Output(component_id='F10_second_button', component_property='n_clicks'),
     Output(component_id='F10_place_button', component_property='n_clicks'),
     Output(component_id='F10_split', component_property='children')],
    [Input(component_id='F10_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F10_first_button', component_property='n_clicks'),
     State(component_id='F10_second_button', component_property='n_clicks'),
     State(component_id='F10_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F10_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F10']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F10'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F10_lws', component_property='children'),
     Output(component_id='F10_lls', component_property='children'),
     Output(component_id='F10_form', component_property='children'),
     Output(component_id='F10_first_button', component_property='n_clicks'),
     Output(component_id='F10_second_button', component_property='n_clicks'),
     Output(component_id='F10_third_button', component_property='n_clicks'),
     Output(component_id='F10_split', component_property='children')],
    [Input(component_id='F10_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F10_first_button', component_property='n_clicks'),
     State(component_id='F10_second_button', component_property='n_clicks'),
     State(component_id='F10_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F10_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F10']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F10'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F11_lws', component_property='children'),
     Output(component_id='F11_lls', component_property='children'),
     Output(component_id='F11_form', component_property='children'),
     Output(component_id='F11_second_button', component_property='n_clicks'),
     Output(component_id='F11_third_button', component_property='n_clicks'),
     Output(component_id='F11_place_button', component_property='n_clicks'),
     Output(component_id='F11_split', component_property='children')],
    [Input(component_id='F11_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F11_second_button', component_property='n_clicks'),
     State(component_id='F11_third_button', component_property='n_clicks'),
     State(component_id='F11_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F11_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F11']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F11'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F11_lws', component_property='children'),
     Output(component_id='F11_lls', component_property='children'),
     Output(component_id='F11_form', component_property='children'),
     Output(component_id='F11_first_button', component_property='n_clicks'),
     Output(component_id='F11_third_button', component_property='n_clicks'),
     Output(component_id='F11_place_button', component_property='n_clicks'),
     Output(component_id='F11_split', component_property='children')],
    [Input(component_id='F11_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F11_first_button', component_property='n_clicks'),
     State(component_id='F11_third_button', component_property='n_clicks'),
     State(component_id='F11_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F11_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F11']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F11'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F11_lws', component_property='children'),
     Output(component_id='F11_lls', component_property='children'),
     Output(component_id='F11_form', component_property='children'),
     Output(component_id='F11_first_button', component_property='n_clicks'),
     Output(component_id='F11_second_button', component_property='n_clicks'),
     Output(component_id='F11_place_button', component_property='n_clicks'),
     Output(component_id='F11_split', component_property='children')],
    [Input(component_id='F11_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F11_first_button', component_property='n_clicks'),
     State(component_id='F11_second_button', component_property='n_clicks'),
     State(component_id='F11_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F11_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F11']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F11'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F11_lws', component_property='children'),
     Output(component_id='F11_lls', component_property='children'),
     Output(component_id='F11_form', component_property='children'),
     Output(component_id='F11_first_button', component_property='n_clicks'),
     Output(component_id='F11_second_button', component_property='n_clicks'),
     Output(component_id='F11_third_button', component_property='n_clicks'),
     Output(component_id='F11_split', component_property='children')],
    [Input(component_id='F11_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F11_first_button', component_property='n_clicks'),
     State(component_id='F11_second_button', component_property='n_clicks'),
     State(component_id='F11_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F11_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F11']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F11'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F12_lws', component_property='children'),
     Output(component_id='F12_lls', component_property='children'),
     Output(component_id='F12_form', component_property='children'),
     Output(component_id='F12_second_button', component_property='n_clicks'),
     Output(component_id='F12_third_button', component_property='n_clicks'),
     Output(component_id='F12_place_button', component_property='n_clicks'),
     Output(component_id='F12_split', component_property='children')],
    [Input(component_id='F12_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F12_second_button', component_property='n_clicks'),
     State(component_id='F12_third_button', component_property='n_clicks'),
     State(component_id='F12_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F12_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F12']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F12'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F12_lws', component_property='children'),
     Output(component_id='F12_lls', component_property='children'),
     Output(component_id='F12_form', component_property='children'),
     Output(component_id='F12_first_button', component_property='n_clicks'),
     Output(component_id='F12_third_button', component_property='n_clicks'),
     Output(component_id='F12_place_button', component_property='n_clicks'),
     Output(component_id='F12_split', component_property='children')],
    [Input(component_id='F12_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F12_first_button', component_property='n_clicks'),
     State(component_id='F12_third_button', component_property='n_clicks'),
     State(component_id='F12_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F12_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F12']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F12'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F12_lws', component_property='children'),
     Output(component_id='F12_lls', component_property='children'),
     Output(component_id='F12_form', component_property='children'),
     Output(component_id='F12_first_button', component_property='n_clicks'),
     Output(component_id='F12_second_button', component_property='n_clicks'),
     Output(component_id='F12_place_button', component_property='n_clicks'),
     Output(component_id='F12_split', component_property='children')],
    [Input(component_id='F12_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F12_first_button', component_property='n_clicks'),
     State(component_id='F12_second_button', component_property='n_clicks'),
     State(component_id='F12_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F12_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F12']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F12'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F12_lws', component_property='children'),
     Output(component_id='F12_lls', component_property='children'),
     Output(component_id='F12_form', component_property='children'),
     Output(component_id='F12_first_button', component_property='n_clicks'),
     Output(component_id='F12_second_button', component_property='n_clicks'),
     Output(component_id='F12_third_button', component_property='n_clicks'),
     Output(component_id='F12_split', component_property='children')],
    [Input(component_id='F12_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F12_first_button', component_property='n_clicks'),
     State(component_id='F12_second_button', component_property='n_clicks'),
     State(component_id='F12_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F12_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F12']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F12'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F13_lws', component_property='children'),
     Output(component_id='F13_lls', component_property='children'),
     Output(component_id='F13_form', component_property='children'),
     Output(component_id='F13_second_button', component_property='n_clicks'),
     Output(component_id='F13_third_button', component_property='n_clicks'),
     Output(component_id='F13_place_button', component_property='n_clicks'),
     Output(component_id='F13_split', component_property='children')],
    [Input(component_id='F13_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F13_second_button', component_property='n_clicks'),
     State(component_id='F13_third_button', component_property='n_clicks'),
     State(component_id='F13_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F13_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F13']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F13'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F13_lws', component_property='children'),
     Output(component_id='F13_lls', component_property='children'),
     Output(component_id='F13_form', component_property='children'),
     Output(component_id='F13_first_button', component_property='n_clicks'),
     Output(component_id='F13_third_button', component_property='n_clicks'),
     Output(component_id='F13_place_button', component_property='n_clicks'),
     Output(component_id='F13_split', component_property='children')],
    [Input(component_id='F13_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F13_first_button', component_property='n_clicks'),
     State(component_id='F13_third_button', component_property='n_clicks'),
     State(component_id='F13_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F13_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F13']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F13'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F13_lws', component_property='children'),
     Output(component_id='F13_lls', component_property='children'),
     Output(component_id='F13_form', component_property='children'),
     Output(component_id='F13_first_button', component_property='n_clicks'),
     Output(component_id='F13_second_button', component_property='n_clicks'),
     Output(component_id='F13_place_button', component_property='n_clicks'),
     Output(component_id='F13_split', component_property='children')],
    [Input(component_id='F13_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F13_first_button', component_property='n_clicks'),
     State(component_id='F13_second_button', component_property='n_clicks'),
     State(component_id='F13_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F13_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F13']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F13'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F13_lws', component_property='children'),
     Output(component_id='F13_lls', component_property='children'),
     Output(component_id='F13_form', component_property='children'),
     Output(component_id='F13_first_button', component_property='n_clicks'),
     Output(component_id='F13_second_button', component_property='n_clicks'),
     Output(component_id='F13_third_button', component_property='n_clicks'),
     Output(component_id='F13_split', component_property='children')],
    [Input(component_id='F13_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F13_first_button', component_property='n_clicks'),
     State(component_id='F13_second_button', component_property='n_clicks'),
     State(component_id='F13_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F13_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F13']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F13'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F14_lws', component_property='children'),
     Output(component_id='F14_lls', component_property='children'),
     Output(component_id='F14_form', component_property='children'),
     Output(component_id='F14_second_button', component_property='n_clicks'),
     Output(component_id='F14_third_button', component_property='n_clicks'),
     Output(component_id='F14_place_button', component_property='n_clicks'),
     Output(component_id='F14_split', component_property='children')],
    [Input(component_id='F14_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F14_second_button', component_property='n_clicks'),
     State(component_id='F14_third_button', component_property='n_clicks'),
     State(component_id='F14_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F14_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F14']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F14'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F14_lws', component_property='children'),
     Output(component_id='F14_lls', component_property='children'),
     Output(component_id='F14_form', component_property='children'),
     Output(component_id='F14_first_button', component_property='n_clicks'),
     Output(component_id='F14_third_button', component_property='n_clicks'),
     Output(component_id='F14_place_button', component_property='n_clicks'),
     Output(component_id='F14_split', component_property='children')],
    [Input(component_id='F14_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F14_first_button', component_property='n_clicks'),
     State(component_id='F14_third_button', component_property='n_clicks'),
     State(component_id='F14_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F14_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F14']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F14'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F14_lws', component_property='children'),
     Output(component_id='F14_lls', component_property='children'),
     Output(component_id='F14_form', component_property='children'),
     Output(component_id='F14_first_button', component_property='n_clicks'),
     Output(component_id='F14_second_button', component_property='n_clicks'),
     Output(component_id='F14_place_button', component_property='n_clicks'),
     Output(component_id='F14_split', component_property='children')],
    [Input(component_id='F14_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F14_first_button', component_property='n_clicks'),
     State(component_id='F14_second_button', component_property='n_clicks'),
     State(component_id='F14_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F14_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F14']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F14'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F14_lws', component_property='children'),
     Output(component_id='F14_lls', component_property='children'),
     Output(component_id='F14_form', component_property='children'),
     Output(component_id='F14_first_button', component_property='n_clicks'),
     Output(component_id='F14_second_button', component_property='n_clicks'),
     Output(component_id='F14_third_button', component_property='n_clicks'),
     Output(component_id='F14_split', component_property='children')],
    [Input(component_id='F14_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F14_first_button', component_property='n_clicks'),
     State(component_id='F14_second_button', component_property='n_clicks'),
     State(component_id='F14_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F14_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F14']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F14'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F15_lws', component_property='children'),
     Output(component_id='F15_lls', component_property='children'),
     Output(component_id='F15_form', component_property='children'),
     Output(component_id='F15_second_button', component_property='n_clicks'),
     Output(component_id='F15_third_button', component_property='n_clicks'),
     Output(component_id='F15_place_button', component_property='n_clicks'),
     Output(component_id='F15_split', component_property='children')],
    [Input(component_id='F15_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F15_second_button', component_property='n_clicks'),
     State(component_id='F15_third_button', component_property='n_clicks'),
     State(component_id='F15_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F15_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F15']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F15'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F15_lws', component_property='children'),
     Output(component_id='F15_lls', component_property='children'),
     Output(component_id='F15_form', component_property='children'),
     Output(component_id='F15_first_button', component_property='n_clicks'),
     Output(component_id='F15_third_button', component_property='n_clicks'),
     Output(component_id='F15_place_button', component_property='n_clicks'),
     Output(component_id='F15_split', component_property='children')],
    [Input(component_id='F15_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F15_first_button', component_property='n_clicks'),
     State(component_id='F15_third_button', component_property='n_clicks'),
     State(component_id='F15_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F15_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F15']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F15'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F15_lws', component_property='children'),
     Output(component_id='F15_lls', component_property='children'),
     Output(component_id='F15_form', component_property='children'),
     Output(component_id='F15_first_button', component_property='n_clicks'),
     Output(component_id='F15_second_button', component_property='n_clicks'),
     Output(component_id='F15_place_button', component_property='n_clicks'),
     Output(component_id='F15_split', component_property='children')],
    [Input(component_id='F15_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F15_first_button', component_property='n_clicks'),
     State(component_id='F15_second_button', component_property='n_clicks'),
     State(component_id='F15_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F15_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F15']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F15'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F15_lws', component_property='children'),
     Output(component_id='F15_lls', component_property='children'),
     Output(component_id='F15_form', component_property='children'),
     Output(component_id='F15_first_button', component_property='n_clicks'),
     Output(component_id='F15_second_button', component_property='n_clicks'),
     Output(component_id='F15_third_button', component_property='n_clicks'),
     Output(component_id='F15_split', component_property='children')],
    [Input(component_id='F15_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F15_first_button', component_property='n_clicks'),
     State(component_id='F15_second_button', component_property='n_clicks'),
     State(component_id='F15_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F15_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F15']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F15'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]

###############################################################################################################

@app.callback(
    [Output(component_id='F16_lws', component_property='children'),
     Output(component_id='F16_lls', component_property='children'),
     Output(component_id='F16_form', component_property='children'),
     Output(component_id='F16_second_button', component_property='n_clicks'),
     Output(component_id='F16_third_button', component_property='n_clicks'),
     Output(component_id='F16_place_button', component_property='n_clicks'),
     Output(component_id='F16_split', component_property='children')],
    [Input(component_id='F16_first_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F16_second_button', component_property='n_clicks'),
     State(component_id='F16_third_button', component_property='n_clicks'),
     State(component_id='F16_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F16_form_win(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, sclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F16']
    if len(searchterm)==1:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [generate_li_for_fav_sb(1, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F16'], split_by_type, 1)
                ]
    else:
        return ["{}".format(maxminstreaks(1, dff)[0]),
                "{}".format(maxminstreaks(1, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F16_lws', component_property='children'),
     Output(component_id='F16_lls', component_property='children'),
     Output(component_id='F16_form', component_property='children'),
     Output(component_id='F16_first_button', component_property='n_clicks'),
     Output(component_id='F16_third_button', component_property='n_clicks'),
     Output(component_id='F16_place_button', component_property='n_clicks'),
     Output(component_id='F16_split', component_property='children')],
    [Input(component_id='F16_second_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F16_first_button', component_property='n_clicks'),
     State(component_id='F16_third_button', component_property='n_clicks'),
     State(component_id='F16_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F16_form_shp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, tclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F16']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [generate_li_for_fav_sb(2, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F16'], split_by_type, 2)
                ]
    else:
        return ["{}".format(maxminstreaks(2, dff)[0]),
                "{}".format(maxminstreaks(2, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F16_lws', component_property='children'),
     Output(component_id='F16_lls', component_property='children'),
     Output(component_id='F16_form', component_property='children'),
     Output(component_id='F16_first_button', component_property='n_clicks'),
     Output(component_id='F16_second_button', component_property='n_clicks'),
     Output(component_id='F16_place_button', component_property='n_clicks'),
     Output(component_id='F16_split', component_property='children')],
    [Input(component_id='F16_third_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F16_first_button', component_property='n_clicks'),
     State(component_id='F16_second_button', component_property='n_clicks'),
     State(component_id='F16_place_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F16_form_thp(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, pclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F16']
    if len(searchterm) == 1:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [generate_li_for_fav_sb(3, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F16'], split_by_type, 3)
                ]
    else:
        return ["{}".format(maxminstreaks(3, dff)[0]),
                "{}".format(maxminstreaks(3, dff)[1]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output(component_id='F16_lws', component_property='children'),
     Output(component_id='F16_lls', component_property='children'),
     Output(component_id='F16_form', component_property='children'),
     Output(component_id='F16_first_button', component_property='n_clicks'),
     Output(component_id='F16_second_button', component_property='n_clicks'),
     Output(component_id='F16_third_button', component_property='n_clicks'),
     Output(component_id='F16_split', component_property='children')],
    [Input(component_id='F16_place_button', component_property='n_clicks')],
    [State(component_id='store', component_property='children'),
     State(component_id='fav_season', component_property='value'),
     State(component_id='fav_jockey', component_property='value'),
     State(component_id='fav_trainer', component_property='value'),
     State(component_id='fav_distance', component_property='value'),
     State(component_id='fav_raceno', component_property='value'),
     State(component_id='fav_dr', component_property='value'),
     State(component_id='fav_age', component_property='value'),
     State(component_id='fav_weight', component_property='value'),
     State(component_id='fav_class', component_property='value'),
     State(component_id='F16_first_button', component_property='n_clicks'),
     State(component_id='F16_second_button', component_property='n_clicks'),
     State(component_id='F16_third_button', component_property='n_clicks'),
     State(component_id='identity', component_property='value')], prevent_initial_call=True
)
def update_F16_form_place(clicks, frame, favseason, favjockey, favtrainer, favdistance, faveraceno, favdr, favage, favweight, favclass, fclicks, sclicks, tclicks, searchterm):
    if clicks == 0:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    split_by_type = 'Jockey'
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        # print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            cond.append(dff[key] == value)
            condition.append(conjunction(1, *cond))
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    dff = dff.loc[dff['FO'] == 'F16']
    if len(searchterm) == 1:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [generate_li_for_fav_sb(4, i, j, n) for n,(i,j) in enumerate(zip(dff['Pl'][::-1],dff['FINAL'][::-1]))],
                0,
                0,
                0,
                generate_fav_table_split(dff.loc[dff['FO'] == 'F16'], split_by_type, 4)
                ]
    else:
        return ["{}".format(streak(dff)[1]),
                "{}".format(streak(dff)[2]),
                [],
                0,
                0,
                0,
                []
                ]


@app.callback(
    [Output('firstgraph', 'children'),
     Output('twopointfivegraph', 'children'),
     Output('twopointeightfivegraph', 'children'),
     Output('twopointninefivegraph', 'children'),
     Output('twopointnineninefivegraph', 'children'),
     Output('thirdgraph', 'children'),
     Output('thirdfourthgraph', 'children'),
     Output('fifthgraph', 'children'),
     Output('fav_season','options'),
     Output('fav_jockey', 'options'),
     Output('fav_trainer', 'options'),
     Output('fav_distance', 'options'),
     Output('fav_raceno', 'options'),
     Output('fav_dr', 'options'),
     Output('fav_age', 'options'),
     Output('fav_weight', 'options'),
     Output('fav_class', 'options'),
     Output('twopointsevenfivegraph', 'children')],
    [Input('store', 'children')],
    [State('identity', 'value'),
     State('type', 'value')]
)
def generate_graphs(frame, searchterm, searchtype):
    if frame is None or frame==[]:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    distance_stats = generate_distance_stats(dff)
    dr_stats = generate_dr_stats(dff)
    age_stats = generate_age_stats(dff)
    fo_stats = generate_FO_stats(dff)
    if searchtype == 'Media':
        raise PreventUpdate
    if len(searchterm)==1:
        try:
            return [
                generate_distance_boxes(dff, distance_stats),
                generate_favourite_scoreboard(fo_stats),
                generate_additional_scoreboards(distance_stats, "Distance"),
                generate_additional_scoreboards(dr_stats, "Dr"),
                generate_additional_scoreboards(age_stats, "Age"),
                [html.Div(), dcc.Graph(figure=generatescatter(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                # [html.Div(), dcc.Graph(figure=generatebarfavorite(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                [html.Div(), dcc.Graph(figure=generatebar(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                [html.Div(), dcc.Graph(figure=generatejockeystack(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                [{'label':'All seasons', 'value':'All'}]+[{'label':i,'value':i} for i in dff['Season'].unique()],
                [{'label': 'All jockeys', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Jockey'].unique()) if str(x) != 'nan'])],
                [{'label': 'All trainers', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Trainer'].unique()) if str(x) != 'nan'])],
                [{'label': 'All distances', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Distance'].unique()) if str(x) != 'nan'])],
                [{'label': 'All race no.s', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Race No'].unique()) if str(x) != 'nan'])],
                [{'label': 'All Dr', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Dr'].unique()) if str(x) != 'nan'])],
                [{'label': 'All Age', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Age'].unique()) if str(x) != 'nan'])],
                [{'label': 'All Weights', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Wt'].unique()) if str(x) != 'nan'])],
                [{'label': 'All Classifications', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Classification'].unique()],
                generate_favourite_detailed_table(dff, fo_stats)
            ]
        except:
            return [
                generate_distance_boxes(dff, distance_stats),
                generate_favourite_scoreboard(fo_stats),
                generate_additional_scoreboards(distance_stats, "Distance"),
                generate_additional_scoreboards(dr_stats, "Dr"),
                generate_additional_scoreboards(age_stats, "Age"),
                [html.Div(),
                 dcc.Graph(figure=generatescatter(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                # [html.Div(), dcc.Graph(figure=generatebarfavorite(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                [html.Div(), dcc.Graph(figure=generatebar(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                [html.Div(),
                 dcc.Graph(figure=generatejockeystack(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                [{'label': 'All seasons', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Season'].unique()],
                [{'label': 'All jockeys', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Jockey'].unique()],
                [{'label': 'All trainers', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Trainer'].unique()],
                [{'label': 'All distances', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Distance'].unique()],
                [{'label': 'All race no.s', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Race No'].unique()],
                [{'label': 'All Dr', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Dr'].unique()],
                [{'label': 'All Age', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Age'].unique()],
                [{'label': 'All Weights', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Wt'].unique()],
                [{'label': 'All Classifications', 'value': 'All'}] + [{'label': i, 'value': i} for i in
                                                                      dff['Classification'].unique()],
                generate_favourite_detailed_table(dff, fo_stats)
            ]
    else:
        try:
            return [
                generate_distance_boxes(dff, distance_stats),
                generate_favourite_scoreboard(fo_stats),
                generate_additional_scoreboards(distance_stats, "Distance"),
                generate_additional_scoreboards(dr_stats, "Dr"),
                generate_additional_scoreboards(age_stats, "Age"),
                [html.Div(), dcc.Graph(figure=generatescatter(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                # [html.Div(), dcc.Graph(figure=generatebarfavorite(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                [html.Div(), dcc.Graph(figure=generatebar(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                [],
                [{'label':'All seasons', 'value':'All'}]+ [{'label':i,'value':i} for i in dff['Season'].unique()],
                [{'label': 'All jockeys', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Jockey'].unique()) if str(x) != 'nan'])],
                [{'label': 'All trainers', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Trainer'].unique()) if str(x) != 'nan'])],
                [{'label': 'All distances', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Distance'].unique()) if str(x) != 'nan'])],
                [{'label': 'All race no.s', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Race No'].unique()) if str(x) != 'nan'])],
                [{'label': 'All Dr', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Dr'].unique()) if str(x) != 'nan'])],
                [{'label': 'All Age', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Age'].unique()) if str(x) != 'nan'])],
                [{'label': 'All Weights', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted([x for x in list(dff['Wt'].unique()) if str(x) != 'nan'])],
                [{'label': 'All Classifications', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Classification'].unique()],
                generate_favourite_detailed_table(dff,fo_stats)
            ]
        except:
            return [
                generate_distance_boxes(dff, distance_stats),
                generate_favourite_scoreboard(fo_stats),
                generate_additional_scoreboards(distance_stats, "Distance"),
                generate_additional_scoreboards(dr_stats, "Dr"),
                generate_additional_scoreboards(age_stats, "Age"),
                [html.Div(),
                 dcc.Graph(figure=generatescatter(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                # [html.Div(), dcc.Graph(figure=generatebarfavorite(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                [html.Div(), dcc.Graph(figure=generatebar(dff, searchtype), style={'height': '100%', 'width': '100%'})],
                [],
                [{'label': 'All seasons', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Season'].unique()],
                [{'label': 'All jockeys', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Jockey'].unique()],
                [{'label': 'All trainers', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Trainer'].unique()],
                [{'label': 'All distances', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Distance'].unique()],
                [{'label': 'All race no.s', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Race No'].unique()],
                [{'label': 'All Dr', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Dr'].unique()],
                [{'label': 'All Age', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Age'].unique()],
                [{'label': 'All Weights', 'value': 'All'}] + [{'label': i, 'value': i} for i in dff['Wt'].unique()],
                [{'label': 'All Classifications', 'value': 'All'}] + [{'label': i, 'value': i} for i in
                                                                      dff['Classification'].unique()],
                generate_favourite_detailed_table(dff, fo_stats)
            ]

@app.callback(
    Output('twopointsevenfivegraph', 'children'),
    [Input('fav_season', 'value'),
     Input('fav_jockey', 'value'),
     Input('fav_trainer', 'value'),
     Input('fav_distance', 'value'),
     Input('fav_raceno', 'value'),
     Input('fav_dr', 'value'),
     Input('fav_age', 'value'),
     Input('fav_weight', 'value'),
     Input('fav_class', 'value')],
    [State('store', 'children')]
)
def update_fav_detailed_table_season(favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass,frame):
    if frame is None or frame==[]:
        raise PreventUpdate
    dff = pd.read_json(frame[0], orient='split')
    cond = []
    cols = ['Season', 'Jockey', 'Trainer', 'Distance', 'Race No', 'Dr', 'Age', 'Wt', 'Classification']
    searchparam = [favseason,favjockey,favtrainer,favdistance,faveraceno,favdr,favage,favweight,favclass]
    condition = []
    for (key, value) in zip(cols, searchparam):
        print(key,'=',value)
        if value != [] and value is not None and value != "" and value !='All':
            print(value)
            cond.append(dff[key] == value)
            print(cond)
            condition.append(conjunction(1, *cond))
            print(condition)
            cond = []
    if condition:
        dff = dff[conjunction(0, *condition)]
    fo_stats = generate_FO_stats(dff)
    return generate_favourite_detailed_table(dff, fo_stats)



@app.callback(
    Output('selected_filters', 'children'),
    [Input('store', 'children')],
    [State('identity', 'value'),
     State('type', 'value'),
     State('Odds', 'value'),
     State('Favourite', 'value'),
     State('Season', 'value'),
     State('Jockey', 'value'),
     State('Trainer', 'value'),
     State('Distance', 'value'),
     State('Race_no', 'value'),
     State('Dr', 'value'),
     State('Age', 'value'),
     State('Weight', 'value'),
     State('Classification', 'value')]
)
def update_selected_filters(frame, searchterm, searchtype, oddstype, fav, *filts):
    if frame is None or frame==[]:
        raise PreventUpdate
    selections = []
    print("HEYYYY", filts, "Heyyy", selections)
    if searchtype == 'Media':
        selections.append(html.Span("{}".format(searchterm), className='main_filter_span'))
    else:
        for i in searchterm:
            selections.append(html.Span("{}".format(i), className='main_filter_span'))
        # print("Hey", selections)
    for f in filts:
        if f != [] and f is not None:
            for i in f:
                selections.append(html.Span("{}".format(i), className='sub_filter_span'))
            selections.append("|")
    print("Hey", selections)
    if fav!= [] and fav is not None:
        for i in fav:
            selections.append(html.Span("{}".format(i), className='fav_filter_span'))
    print("Hey", selections)
    return selections



if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)