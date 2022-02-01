import dash
from dash.dependencies import Input, Output, State
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
# import dash_design_kit as ddk
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
import math

def point5round(a):
    return math.ceil(a * 2) / 2

'''********************************************************'''
# data = pd.read_excel(r'data\NEW METHOD 210822.xlsx')
data = pd.read_excel(r'data\ALL METHODS - MASTER FILE.xlsb', engine='pyxlsb')
data['Date']= pd.to_datetime(data['Date'], unit='D', origin='1899-12-30T00:00:00.000000000')
data.rename(columns = {'CENTER':'Centre', 'VENUE ID': 'Venue ID', 'R No #': 'Race No', 'RES': 'Result', 'CLASS': 'Class', 'DIST': 'Dist'}, inplace = True)
data.Dist.replace([np.nan],[0],inplace=True)
data.Class.replace([np.nan],['-'],inplace=True)
data.Jockey.replace([np.nan],['-'],inplace=True)
data.Trainer.replace([np.nan],['-'],inplace=True)
data['Horse Name'].replace([np.nan],['-'],inplace=True)
data['Sr #'] = [x+1 for x in data.index]
data['Rtg'] = [0]*len(data)
data['Penalty'] = [0]*len(data)
data['Time in S'] = [0]*len(data)
data['Time'] = [0]*len(data)
data['Odds'] = [0]*len(data)

data.rename(columns = {'Center':'Season', 'Horse Name': 'Horse', 'Dist': 'Distance', }, inplace = True)

data.Age.replace([np.nan, '3y', '5y', '4y', '6y', '7y', '8y', '9y', '2y', '0y', '14y', '10y', '11y', '12y', '1y'],[0, 3, 5, 4, 6, 7, 8, 9, 2, 0, 14, 10, 11, 12, 1],inplace=True)
data.Result.replace(['DNC', 'NC', 'DNF', 'DR', 'W'],[0,0,0,0,0], inplace=True)
data.Jockey.replace([np.nan], [""],inplace=True)
data.Trainer.replace([np.nan], [""],inplace=True)

data['combined_data'] = list(zip(data.Result, data.Age))

data['RunNo.'] = [-1]*len(data)

data['Dr'].replace([np.nan], [0], inplace=True)
data['LBW'].replace([np.nan, 'W'], [0,0], inplace=True)
data['Sh'].replace(['s', 'a', np.nan], ['S','A','-'], inplace=True)
# data['Time in S'].replace([np.nan, 'DNF'], [0,0], inplace=True)
data['Dr'] = data.Dr.astype(int)

data.drop(columns=['Time', 'Odds', 'Rtg'], inplace=True)
data.rename(columns = {'Time in S':'Time'}, inplace=True)
data['Date'] = data['Date'].dt.date

data['first_age']=[0]*len(data)
data.loc[data['Result']==1,'first_age']=data.loc[data['Result']==1,'Age']

new = data['Venue ID'].str.split("=", n = 1, expand = True)
data['Venue ID'] = new[1]
data['Venue ID'] = data['Venue ID'].astype(int)
data['Venue Name'] = data['Venue ID'].replace([1,2,3,4,7,8,9,10,11], ['KOL', 'MUM', 'BLR', 'CHE', 'DEL', 'MYS', 'OOT', 'PUN', 'HYD'])

result_valid = list(data['Result'].unique())
result_valid.remove(1)

data['LBWmod'] = data['LBW'].apply(lambda x: point5round(x) if not np.isnan(x) else x)
'''**************************************************************************'''

resultslide_marks = {}
keys = np.arange(2,21)
values = np.arange(2,21)
for i in keys:
        resultslide_marks[int(i)] = str(i)

lbwslide_marks = {}
keys = np.arange(0,21)
values = np.arange(0,21)
for i in keys:
        lbwslide_marks[int(i)] = str(i)

app = dash.Dash(__name__)

def generate_specific_stats(datas, types):
    pl = [1,2,3]
    m = datas.groupby([types], as_index=True).size().to_frame()
    m.rename(columns={0: 'TR'}, inplace=True)
    n = datas.groupby([types, 'Result'], as_index=True).size().unstack(fill_value=0)
    mis_col = [i for i in pl if i not in n.columns]
    for col in mis_col:
        n[col] = [0] * len(n)
    dis = pd.concat([m, n], axis=1)
    dis.rename(columns={1: "W", 2: "SHP", 3: "THP", 'maxwinstreaks': 'LWS', 'maxlosestreaks': 'LLS'}, inplace=True)
    dis["W%"] = ((dis["W"] / dis['TR']) * 100).round(2)
    dis["SHP%"] = ((dis["SHP"] / dis['TR']) * 100).round(2)
    dis["THP%"] = ((dis["THP"] / dis['TR']) * 100).round(2)
    dis["Plc%"] = (((dis["W"] + dis["SHP"] + dis["THP"]) / dis['TR']) * 100).round(2)
    dis["Plc"] = dis["W"] + dis["SHP"] + dis["THP"]
    dis["BO"] = dis["TR"] - dis["Plc"]
    return dis

def generate_specific_scoreboards(xdf, type):
    header = html.H3("{} stats".format(type),
            style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                   'font-family': 'Arial Black'})
    fav_sb =    dash_table.DataTable(
                   # id='table',
                   columns=[{"name": i, "id": i} for i in xdf[['TR','W','W%','SHP','SHP%','THP','THP%','Plc','Plc%','BO']].reset_index().columns],
                   data=xdf[['TR','W','W%','SHP','SHP%','THP','THP%','Plc','Plc%','BO']].reset_index().to_dict('records'),
                   page_size=15,
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

    return header, fav_sb, html.Div(style={'height':'50px'})



def generatebaroverall(RunFiltered_df, type):
    STR = len(RunFiltered_df)
    SW = int(len(RunFiltered_df.loc[RunFiltered_df['Result'] == 1]))
    SS = int(len(RunFiltered_df.loc[RunFiltered_df['Result'] == 2]))
    ST = int(len(RunFiltered_df.loc[RunFiltered_df['Result'] == 3]))
    SP = int(len(RunFiltered_df.loc[(RunFiltered_df['Result'] == 1) | (RunFiltered_df['Result'] == 2) | (RunFiltered_df['Result'] == 3)]))
    SB = STR - SP
    SWpct = round(100 * safe_division(SW, STR), 2)
    SSpct = round(100 * safe_division(SS, STR), 2)
    STpct = round(100 * safe_division(ST, STR), 2)
    SPpct = round(100 * safe_division(SP, STR), 2)
    SBpct = round(100 * safe_division(SB, STR), 2)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['W%', 'SHP%', 'THP%', 'BO%'],
                             y=[SWpct, SSpct, STpct, SBpct],
                             marker_color = ["#2ECC71", "#e8e40e", "#3498DB", "#ff2200"],
                             text=[round(SWpct,2), round(SSpct,2), round(STpct,2), round(SBpct,2)],
                             textposition='outside',
                             hoverinfo='none',
                             name = type
                         ))
    # fig.add_trace(go.Bar(x=['W%', 'SHP%', 'THP%', 'BO%'],
    #                          y=[SSpct, SWpct, SBpct, STpct],
    #                          marker_color = ["#2ECC71", "#e8e40e", "#3498DB", "#ff2200"],
    #                          text=[SWpct, SSpct, STpct, SBpct],
    #                          textposition='outside',
    #                          hoverinfo='none',
    #                          name = 'Run1'
    #                      ))
    fig.update_xaxes(title_text="Result",
                     ticktext=['W%', 'SHP%', 'THP%', 'BO%'],
                     showticklabels=True)
    fig.update_yaxes(title_text="Percentage",
                     showticklabels=True,
                     showgrid=True,
                     tickvals=np.linspace(0,100,11),
                     range=[0, 110])
    fig.update_layout(title_text='Result percentages - {}(filtered)'.format(type),
                      showlegend=True,
                      autosize=True,
                      # height=700,
                      xaxis={'showgrid': True, 'zeroline': True, 'gridcolor': '#082255', 'zerolinecolor': '#082255'},
                      yaxis={'showgrid': True, 'zeroline': True, 'gridcolor': '#082255', 'zerolinecolor': '#082255'},
                      # xaxis_tickangle=-45,
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

def generate_jockey_stats(df):
    n = df.groupby(["Venue ID", "Jockey", "Result"], as_index=True).size().unstack(fill_value=0)
    n.rename(columns={0: 'NA', 1: "W", 2: "SHP", 3: "THP"}, inplace=True)
    m = df.groupby(["Venue ID", "Jockey"], as_index=True).size().to_frame()
    Jockey_stats = pd.concat([n, m], axis=1)
    Jockey_stats.rename(columns={0: 'Total_Races', 1: "W", 2: "SHP", 3: "THP"}, inplace=True)
    Jockey_stats["W%"] = ((Jockey_stats["W"] / Jockey_stats['Total_Races']) * 100).round(2)
    Jockey_stats = Jockey_stats[['Total_Races', 'W', 'SHP', 'THP', 'W%']]
    return Jockey_stats

def generate_race_stats(df):
    # race_stats = df.groupby(["Date", "Venue ID", "Season", "Race No", "LBWmod"], as_index=True).size().unstack(
    #     fill_value=0)
    # race_stats.reset_index(inplace=True)
    # race_stats.index = np.arange(1, len(race_stats) + 1)
    m = df.groupby(["Date", "Venue ID", "Season", "Race No", "first_age"], as_index=True).size().unstack(
        fill_value=0)
    new = {key: str(key) + 'y' for key in m.columns}
    m.rename(columns=new, inplace=True)
    n = df.groupby(["Date", "Venue ID", "Season", "Race No", "LBWmod"], as_index=True).size().unstack(
        fill_value=0)
    race_stats = pd.concat([m, n], axis=1)
    race_stats.reset_index(inplace=True)
    race_stats.index = np.arange(1, len(race_stats) + 1)
    return race_stats


def generate_initial_qualifying_races(lbw_limit1, lbw_limit2, race_stats, df, first_age):
    # lbw_limit1 = 2.5
    # lbw_limit2 = 4.5
    # first_age_cols = [(1, x) for x in first_age]
    first_age_cols = [str(i) + 'y' for i in first_age]
    number = int((lbw_limit2 - lbw_limit1) / 0.5)
    a = np.linspace(lbw_limit1 + 0.5, lbw_limit2, number, endpoint=True)
    cond = []
    condition = []
    for col in a:
        cond.append(race_stats[col] == 0)
        condition.append(conjunction(0, *cond))
    if condition:
        initial_qualifying_races = race_stats[conjunction(0, *condition)]

    cond = []
    condition = []
    for col in first_age_cols:
        cond.append(race_stats[col] == 1)
        condition.append(conjunction(1, *cond))
    if condition:
        initial_qualifying_races = initial_qualifying_races[conjunction(1, *condition)]

    initial_qualifying_races = initial_qualifying_races[["Date", "Season", "Race No", "Venue ID"]]

    initial_qualifying_races_detailed = pd.merge(initial_qualifying_races, df,
                                                 on=["Date", "Venue ID", "Season", "Race No"])
    initial_qualifying_races_detailed.set_index(["Date", "Venue ID", "Season", "Race No"], inplace=True)

    return initial_qualifying_races, initial_qualifying_races_detailed


def generate_initial_qualified_horses(initial_qualifying_races_detailed, lbw_limit1, result_limit):
    # horseno_limit1 = 2
    # horseno_limit2 = 10
    if result_limit==[]:
        result_limit = result_valid
    initial_qualified_horses = initial_qualifying_races_detailed.loc[
        (initial_qualifying_races_detailed['LBW'] <= lbw_limit1) & (
            initial_qualifying_races_detailed['Result'].isin(result_limit))]
    initial_qualified_horses.reset_index(inplace=True)
    initial_qualified_horses.index = np.arange(1, len(initial_qualified_horses) + 1)
    initial_qualified_horses = initial_qualified_horses[
        ['Horse', 'Date', 'Venue ID', 'Centre', 'Season', 'Race No', 'Sr #', 'Trainer', 'Class', 'Distance', 'Age',
         'Wt', 'Jockey', 'Dr', 'Result', 'LBW', 'Penalty', 'Time', 'RunNo.', 'LBWmod', 'Sh']]
    return initial_qualified_horses


def generate_qualified_horses(class_list, age_list, venue_list, distance_list, initial_qualified_horses):
    a = ['Class', 'Age', 'Venue ID', 'Distance']
    b = [class_list, age_list, venue_list, distance_list]
    cond = []
    condition = []
    for col, filt in zip(a, b):
        cond.append(initial_qualified_horses[col].isin(filt))
        condition.append(conjunction(0, *cond))
    if condition:
        qualified_horses = initial_qualified_horses[conjunction(0, *condition)]
    return qualified_horses


def generate_qualifying_races(qualified_horses):
    qualifying_races = qualified_horses[["Date", "Season", "Race No", "Venue ID"]]
    qualifying_races.drop_duplicates(inplace=True, ignore_index=True)
    # qualifying_races['Date'] = qualifying_races['Date'].dt.date
    return qualifying_races


def generate_qualifying_races_detailed(qualifying_races, df):
    # print(df['Date'].dtype, qualifying_races['Date'].dtype)
    qualifying_races_detailed = pd.merge(qualifying_races, df, on=["Date", "Season", "Race No"])
    qualifying_races_detailed.set_index(["Date", "Season", "Race No"], inplace=True)
    qualifying_races_detailed['LBW'] = round(qualifying_races_detailed['LBW'],2)
    qualifying_races_detailed['Time'] = round(qualifying_races_detailed['Time'], 3)
    return qualifying_races_detailed



def generate_TrackedRuns(df, qualified_horses):
    for i in qualified_horses['Sr #']:
        df.loc[i - 1, 'RunNo.'] = 0
    races_Q_horses = pd.merge(qualified_horses[['Horse']], df, on=["Horse"])
    races_Q_horses.drop_duplicates(subset=['Sr #'], keep='first', inplace=True)
    # new.set_index(['Horse', 'Sr #'], inplace=True)
    zerothindex = races_Q_horses.index[races_Q_horses['RunNo.'] == 0].tolist()
    for i in zerothindex:
        if (i + 1 in races_Q_horses.index):
            if (races_Q_horses.loc[i + 1, 'Horse'] == races_Q_horses.loc[i, 'Horse']) and (
                    races_Q_horses.loc[i + 1, 'RunNo.'] < 0):
                races_Q_horses.loc[i + 1, 'RunNo.'] = 1
            elif (races_Q_horses.loc[i + 1, 'Horse'] == races_Q_horses.loc[i, 'Horse']) and (
                    races_Q_horses.loc[i + 1, 'RunNo.'] == 0):
                races_Q_horses.loc[i + 1, 'RunNo.'] = 10
            elif (races_Q_horses.loc[i + 1, 'Horse'] == races_Q_horses.loc[i, 'Horse']) and (
                    races_Q_horses.loc[i + 1, 'RunNo.'] > 0):
                races_Q_horses.loc[i + 1, 'RunNo.'] = 10 * races_Q_horses.loc[i + 1, 'RunNo.'] + 1
        if (i + 2 in races_Q_horses.index):
            if (races_Q_horses.loc[i + 2, 'Horse'] == races_Q_horses.loc[i, 'Horse']) and (
                    races_Q_horses.loc[i + 2, 'RunNo.'] < 0):
                races_Q_horses.loc[i + 2, 'RunNo.'] = 2
            elif (races_Q_horses.loc[i + 2, 'Horse'] == races_Q_horses.loc[i, 'Horse']) and (
                    races_Q_horses.loc[i + 2, 'RunNo.'] == 0):
                races_Q_horses.loc[i + 2, 'RunNo.'] = 20
    TrackedRuns = races_Q_horses.loc[races_Q_horses['RunNo.'] >= 0]
    TrackedRuns.reset_index(inplace=True)
    return TrackedRuns


def generate_qualified_horse_summary(TrackedRuns):
    runzero_list = [x for x in TrackedRuns['RunNo.'].unique() if '0' in str(x)]
    runone_list = [x for x in TrackedRuns['RunNo.'].unique() if '1' in str(x)]
    runtwo_list = [x for x in TrackedRuns['RunNo.'].unique() if '2' in str(x)]
    Run0_df = TrackedRuns.loc[TrackedRuns['RunNo.'].isin(runzero_list)]
    Run1_df = TrackedRuns.loc[TrackedRuns['RunNo.'].isin(runone_list)]
    Run2_df = TrackedRuns.loc[TrackedRuns['RunNo.'].isin(runtwo_list)]
    ran = [i - 1 for i in Run1_df.index]
    ytr = Run0_df.drop(ran)
    return Run0_df, Run1_df, Run2_df, ytr



def table_counts(df, odds, f):
    return df[odds].value_counts().get(f, 0)


def unique_non_null(s):
    return s.dropna().unique()

def safe_division(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0


def generate_complete_scoreboard(Run0_df, Run1_df, Run2_df, race_stats, qualifying_races):
    total_races = len(race_stats)
    total_qualified_races = len(qualifying_races)
    total_qualified_horses = len(Run0_df)
    qualified_pct = 100 * safe_division(total_qualified_races, total_races)
    ytr = len(Run0_df)- len(Run1_df)
    nr1 = len(Run1_df)
    nr2 = len(Run2_df)
    win = len(Run1_df.loc[Run1_df['Result'] == 1]) + len(Run2_df.loc[Run2_df['Result'] == 1])
    # winpct = 100 * safe_division(win, total_qualified_horses)
    # strike_rate = 100 * safe_division(win, total_races)
    edge = "-"
    n = "-"
    Overall_SB = [html.Div(children=[
        html.H6("Method Stats",
                style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                       'font-family': 'Arial Black'})
    ], id='complete_scoreboard'),
        html.Hr(className='othr'),
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Total Races", id='overall_TR_header'),
                    html.Th("Total Qualified Races", id='overall_TR_header'),
                    html.Th("Qualified%", id='overall_TR_header'),
                    html.Th("Total Qualified Horses", id='overall_TR_header'),
                    html.Th("Yet to run", id='overall_TR_header'),
                    html.Th("NR1", id='overall_TR_header'),
                    html.Th("NR2", id='overall_TR_header'),
                    html.Th('Wins', id='overall_firstpct_header'),
                    html.Th("PG Edge", className='overall_place_header'),
                    html.Th("NG formula", className='overall_place_header')
                ])
            ]),
            html.Tr([
                html.Td("{}".format(total_races), className='place_pos'),
                html.Td("{}".format(total_qualified_races),
                        className='place_pos'),
                html.Td("{0:.2f}".format(qualified_pct),
                        className='place_pos'),
                html.Td("{}".format(total_qualified_horses),
                        className='place_pos'),
                html.Td("{}".format(ytr),
                        className='place_pos'),
                html.Td("{}".format(nr1),
                        className='place_pos'),
                html.Td("{}".format(nr2),
                        className='place_pos'),
                html.Td("{}".format(win),
                        className='first_pos'),
                html.Td("{}".format(edge),
                        className='place_pos'),
                html.Td("{}".format(n),
                        className='place_pos')
            ])
        ], id='overall_table')]
    return Overall_SB

def generate_runs_summary_scoreboard(qualified_horse_summary):
    total = len(qualified_horse_summary)
    W1 = len(qualified_horse_summary.loc[qualified_horse_summary['W1'] == 'Yes'])
    P1 = len(qualified_horse_summary.loc[qualified_horse_summary['P1'] == 'Yes'])
    L1 = len(qualified_horse_summary.loc[qualified_horse_summary['L1'] == 'Yes'])
    W2 = len(qualified_horse_summary.loc[qualified_horse_summary['W2'] == 'Yes'])
    L2 = len(qualified_horse_summary.loc[qualified_horse_summary['L2'] == 'Yes'])
    W1pct = round(100 * safe_division(W1, total), 2)
    P1pct = round(100 * safe_division(P1, total), 2)
    L1pct = round(100 * safe_division(L1, total), 2)
    W2pct = round(100 * safe_division(W2, total), 2)
    L2pct = round(100 * safe_division(L2, total), 2)
    Wpct = W1pct + W2pct
    Lpct = L1pct + L2pct

    Overall_SB = [html.Div(children=[
        html.H6("Run Results for Qualified Horses",
                style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                       'font-family': 'Arial Black'})
    ],id='run_summary_scoreboard'),
                  html.Hr(className='othr'),
                  html.Table([
                      html.Thead([
                          html.Tr([
                              html.Th("Total Qualified", id='overall_TR_header'),
                              html.Th("Run1-W", id='overall_first_header'),
                              html.Th('Run1-W%', id='overall_firstpct_header'),
                              html.Th("Run1-P", id='overall_second_header'),
                              html.Th('Run1-P%', id='overall_secondpct_header'),
                              html.Th("Run1-L", className='overall_loss_header'),
                              html.Th('Run1-L%', className='overall_loss_header'),
                              html.Th('Run2-W', className='first_header'),
                              html.Th('Run2-W%', className='first_header'),
                              html.Th('Run2-L', className='overall_loss_header'),
                              html.Th('Run2-L%', className='overall_loss_header')
                          ])
                      ]),
                      html.Tr([
                          html.Td("{}".format(len(qualified_horse_summary))),
                          html.Td("{}".format(W1),
                                  className='first_pos'),
                          html.Td("{0:.2f}".format(W1pct),
                                  className='first_pos'),
                          html.Td("{}".format(P1),
                                  id='overall_second_pos'),
                          html.Td("{0:.2f}".format(P1pct),
                                  id='overall_second_pct'),
                          html.Td("{}".format(L1),
                                  className='loss'),
                          html.Td("{0:.2f}".format(L1pct),
                                  className='loss'),
                          html.Td("{}".format(W2),
                                  className='first_pos'),
                          html.Td("{0:.2f}".format(W2pct),
                                  className='first_pos'),
                          html.Td("{}".format(L2),
                                  className='loss'),
                          html.Td("{0:.2f}".format(L2pct),
                                  className='loss')
                      ])
                  ], id='overall_sb1_table')]
    return Overall_SB


def generate_overall_scoreboard_for_combo(Run1_df, Run2_df):
    FTR = len(Run1_df)
    FW = int(len(Run1_df.loc[Run1_df['Result'] == 1]))
    FS = int(len(Run1_df.loc[Run1_df['Result'] == 2]))
    FT = int(len(Run1_df.loc[Run1_df['Result'] == 3]))
    FP = int(len(Run1_df.loc[(Run1_df['Result'] == 1) | (Run1_df['Result'] == 2) | (Run1_df['Result'] == 3)]))
    FB = FTR - FP
    FWpct = round(100 * safe_division(FW, FTR), 2)
    FSpct = round(100 * safe_division(FS, FTR), 2)
    FTpct = round(100 * safe_division(FT, FTR), 2)
    FPpct = round(100 * safe_division(FP, FTR), 2)
    STR = len(Run2_df)
    SW = int(len(Run2_df.loc[Run2_df['Result'] == 1]))
    SS = int(len(Run2_df.loc[Run2_df['Result'] == 2]))
    ST = int(len(Run2_df.loc[Run2_df['Result'] == 3]))
    SP = int(len(Run2_df.loc[(Run2_df['Result'] == 1) | (Run2_df['Result'] == 2) | (Run2_df['Result'] == 3)]))
    SB = STR - SP
    SWpct = round(100 * safe_division(SW, STR), 2)
    SSpct = round(100 * safe_division(SS, STR), 2)
    STpct = round(100 * safe_division(ST, STR), 2)
    SPpct = round(100 * safe_division(SP, STR), 2)
    Overall_SB = [html.Div(children = [
        html.H6("Split by First and Second Run",
                style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                       'font-family': 'Arial Black'})
    ],id='split_run_stats_title'),
                  html.Hr(className='othr'),
                  html.Table([
                      html.Thead([
                          html.Tr([
                              html.Th("", id='overall_header'),
                              html.Th("TR", id='overall_TR_header'),
                              html.Th("WIN", id='overall_first_header'),
                              html.Th('W%', id='overall_firstpct_header'),
                              html.Th("SHP", id='overall_second_header'),
                              html.Th('2%', id='overall_secondpct_header'),
                              html.Th("THP", id='overall_third_header'),
                              html.Th('3%', id='overall_thirdpct_header'),
                              html.Th('Plc', id='overall_place_header'),
                              html.Th('P%', id='overall_placepct_header'),
                              html.Th('BO', id='overall_loss_header')
                          ])
                      ]),
                      html.Tr([
                          html.Td("First Run", className='type_header'),
                          html.Td("{}".format(FTR)),
                          html.Td("{}".format(FW), id='overall_first_pos'),
                          html.Td("{0:.2f}".format(FWpct), id='overall_first_pct'),
                          html.Td("{}".format(FS), id='overall_second_pos'),
                          html.Td("{0:.2f}".format(FSpct), id='overall_second_pct'),
                          html.Td("{}".format(FT), id='overall_third_pos'),
                          html.Td("{0:.2f}".format(FTpct), id='overall_third_pct'),
                          html.Td("{}".format(FP), id='overall_place_pos'),
                          html.Td("{0:.2f}".format(FPpct), id='overall_place_pct'),
                          html.Td("{}".format(FB), className='loss')
                      ]),
                      html.Tr([
                          html.Td("Second Run", className='type_header'),
                          html.Td("{}".format(STR)),
                          html.Td("{}".format(SW), id='first_pos'),
                          html.Td("{0:.2f}".format(SWpct), id='first_pct'),
                          html.Td("{}".format(SS), id='second_pos'),
                          html.Td("{0:.2f}".format(SSpct), id='second_pct'),
                          html.Td("{}".format(ST), id='third_pos'),
                          html.Td("{0:.2f}".format(STpct), id='third_pct'),
                          html.Td("{}".format(SP), id='place_pos'),
                          html.Td("{0:.2f}".format(SPpct), id='place_pct'),
                          html.Td("{}".format(SB), className='loss')
                      ])
                  ], id='overall_sb_table')]
    return Overall_SB


def generate_run1run2_scoreboard(RunOverall_df, RunFiltered_df, type):
    FTR = len(RunOverall_df)
    FW = int(len(RunOverall_df.loc[RunOverall_df['Result'] == 1]))
    FS = int(len(RunOverall_df.loc[RunOverall_df['Result'] == 2]))
    FT = int(len(RunOverall_df.loc[RunOverall_df['Result'] == 3]))
    FP = int(len(RunOverall_df.loc[(RunOverall_df['Result'] == 1) | (RunOverall_df['Result'] == 2) | (RunOverall_df['Result'] == 3)]))
    FB = FTR - FP
    FWpct = round(100 * safe_division(FW, FTR), 2)
    FSpct = round(100 * safe_division(FS, FTR), 2)
    FTpct = round(100 * safe_division(FT, FTR), 2)
    FPpct = round(100 * safe_division(FP, FTR), 2)
    STR = len(RunFiltered_df)
    SW = int(len(RunFiltered_df.loc[RunFiltered_df['Result'] == 1]))
    SS = int(len(RunFiltered_df.loc[RunFiltered_df['Result'] == 2]))
    ST = int(len(RunFiltered_df.loc[RunFiltered_df['Result'] == 3]))
    SP = int(len(RunFiltered_df.loc[(RunFiltered_df['Result'] == 1) | (RunFiltered_df['Result'] == 2) | (RunFiltered_df['Result'] == 3)]))
    SB = STR - SP
    SWpct = round(100 * safe_division(SW, STR), 2)
    SSpct = round(100 * safe_division(SS, STR), 2)
    STpct = round(100 * safe_division(ST, STR), 2)
    SPpct = round(100 * safe_division(SP, STR), 2)
    Overall_SB = [html.Div(children = [
        html.H6("{} scoreboard".format(type),
                style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                       'font-family': 'Arial Black'})
    ]),
                  html.Hr(className='othr'),
                  html.Table([
                      html.Thead([
                          html.Tr([
                              html.Th("", className='type_header'),
                              html.Th("TR", className='overall_TR_header'),
                              html.Th("WIN", className='overall_first_header'),
                              html.Th('W%', className='overall_firstpct_header'),
                              html.Th("SHP", className='overall_second_header'),
                              html.Th('2%', className='overall_secondpct_header'),
                              html.Th("THP", className='overall_third_header'),
                              html.Th('3%', className='overall_thirdpct_header'),
                              html.Th('Plc', className='overall_place_header'),
                              html.Th('P%', className='overall_placepct_header'),
                              html.Th('BO', className='overall_loss_header'),
                              html.Th("PG edge", className='overall_TR_header'),
                              html.Th("NG formula", className='overall_TR_header')
                          ])
                      ]),
                      html.Tr([
                          html.Td("Overall", className='type_header'),
                          html.Td("{}".format(FTR)),
                          html.Td("{}".format(FW), className='overall_first_pos'),
                          html.Td("{0:.2f}".format(FWpct), className='overall_first_pct'),
                          html.Td("{}".format(FS), className='overall_second_pos'),
                          html.Td("{0:.2f}".format(FSpct), className='overall_second_pct'),
                          html.Td("{}".format(FT), className='overall_third_pos'),
                          html.Td("{0:.2f}".format(FTpct), className='overall_third_pct'),
                          html.Td("{}".format(FP), className='overall_place_pos'),
                          html.Td("{0:.2f}".format(FPpct), className='overall_place_pct'),
                          html.Td("{}".format(FB), className='loss'),
                          html.Td("-"),
                          html.Td("-")
                      ]),
                      html.Tr([
                          html.Td("Filtered", className='type_header'),
                          html.Td("{}".format(STR)),
                          html.Td("{}".format(SW), className='first_pos'),
                          html.Td("{0:.2f}".format(SWpct), className='first_pct'),
                          html.Td("{}".format(SS), className='second_pos'),
                          html.Td("{0:.2f}".format(SSpct), className='second_pct'),
                          html.Td("{}".format(ST), className='third_pos'),
                          html.Td("{0:.2f}".format(STpct), className='third_pct'),
                          html.Td("{}".format(SP), className='place_pos'),
                          html.Td("{0:.2f}".format(SPpct), className='place_pct'),
                          html.Td("{}".format(SB), className='loss'),
                          html.Td("-"),
                          html.Td("-")
                      ])
                  ], id='{}table'.format(type))]
    return Overall_SB


def generate_qual_race_summary_table(xdf):
    header = html.H3("Qualifying Races",
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    fav_sb = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in xdf.reset_index()[
            ["Date", "Season", "Race No", "Venue ID"]]],
        data=xdf.reset_index()[
            ["Date", "Season", "Race No","Venue ID"]].to_dict(
            'records'),
        page_size=15,
        editable=False,
        export_columns='all',
        export_format='xlsx',
        #filterable = False
        filter_action="none",
        sort_action="native",
        sort_mode="multi",
        column_selectable=None,
        row_selectable=None,
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
        style_header={
            'backgroundColor': 'RebeccaPurple',
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
            'minWidth': 40,
            'minHeight': '20px'
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
        style_filter={
            'textAlign': 'center'
        },
        style_as_list_view=True
    )
    return header, fav_sb


def generate_qual_races_details_table(xdf, date, season, race,venue):
    header1 = html.H3("Qualifying Race Details",
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    # x = 'https://www.indiarace.com/Home/racingCenterEvent?venueId={}&event_date={}&race_type=RESULTS'.format(venue,date)
    x = 'https://www.indiarace.com/Home/oneResultByRaceNo?venueId={}&event_date={}&race_no={}'.format(venue, date, race)
    header2 = html.A(children = [html.H5("Date: {}   Season: {}   Race: {}".format(date, season, race),
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'}),
                                 html.H6("(click to go to website)",
                                         style={'textAlign': 'center', 'color': 'white',
                                                'margin': '0px',
                                                'font-family': 'Arial Black'})
                     ],
                     href=x, target='_blank')
    fav_sb = dash_table.DataTable(
        id='table2',
        columns=[{"name": i, "id": i, "hideable":True} for i in xdf.reset_index()[
            ['Sr #', 'Trainer', 'Class', 'Distance', 'Horse', 'Age', 'Wt',
       'Jockey', 'Dr', 'Result', 'LBW', 'Penalty', 'Time', 'Sh', 'Centre']]],
        data=xdf.reset_index()[
            ['Sr #', 'Trainer', 'Class', 'Distance', 'Horse', 'Age', 'Wt',
       'Jockey', 'Dr', 'Result', 'LBW', 'Penalty', 'Time', 'Sh', 'Centre']].to_dict(
            'records'),
        editable=False,
        #filterable = False
        filter_action="none",
        sort_action="native",
        sort_mode="multi",
        column_selectable=None,
        row_selectable=None,
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
        hidden_columns=['Trainer', 'Jockey', 'Class', 'Distance', 'Wt', 'Dr', 'Penalty', 'Time', 'Sh', 'Centre'],
        style_header={
            'backgroundColor': 'RebeccaPurple',
            'color': 'white',
            'fontWeight': 'bold',
            'font-size': 14,
            'textAlign': 'center'
        },
        style_cell={
            'backgroundColor': '#082255',
            'color': 'white',
            'border': '1px solid #bebebe',
            'font-size': 14,
            'textAlign': 'center',
            'minWidth': 30
        },
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'minHeight': '20px'
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
        # style_data_conditional=[
        # {
        #     'if': {
        #         'filter_query': '{Result} = 2'
        #     },
        #     'backgroundColor': 'dodgerblue',
        #     'color': 'white',
        #     'fontWeight': 'bold'
        # }
        # ],
        style_filter={
            'textAlign': 'center'
        },
        style_as_list_view=True
    )

    return header1, header2, fav_sb


def generate_qualified_horse_list_table(xdf):
    header = html.H3("Qualified Horses".format(),
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    fav_sb = dash_table.DataTable(
        id='table3',
        columns=[{"name": i, "id": i, "hideable":True} for i in xdf[
            ['Horse', 'Date', 'Season', 'Race No', 'Sr #', 'Trainer', 'Class',
       'Distance', 'Age', 'Wt', 'Jockey', 'Dr', 'Result', 'LBW',
       'Penalty', 'Time', 'Sh', 'Centre']].reset_index().columns],
        data=xdf[
            ['Horse', 'Date', 'Season', 'Race No', 'Sr #', 'Trainer', 'Class',
       'Distance', 'Age', 'Wt', 'Jockey', 'Dr', 'Result', 'LBW',
       'Penalty', 'Time', 'Sh', 'Centre']].reset_index().to_dict(
            'records'),
        page_size=20,
        editable=False,
        export_columns='all',
        export_format='xlsx',
        #filterable = False
        filter_action="none",
        sort_action="native",
        sort_mode="multi",
        column_selectable=None,
        row_selectable=None,
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
        hidden_columns=['Trainer', 'Jockey', 'Class', 'Distance', 'Dr', 'Penalty', 'Time', 'Sh', 'Centre'],
        style_header={
            'backgroundColor': 'RebeccaPurple',
            'color': 'white',
            'fontWeight': 'bold',
            'font-size': 14,
            'textAlign': 'center'
        },
        style_cell={
            'backgroundColor': '#082255',
            'color': 'white',
            'border': '1px solid #bebebe',
            'font-size': 14,
            'textAlign': 'center',
            'minWidth': 25,
        },
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'minHeight': '20px'
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
                'column_id': 'Horse'
            },
            'backgroundColor': 'dodgerblue',
            'color': 'white',
            'fontWeight': 'bold'
        },
        {
            'if': {
                'column_id': 'Age'
            },
            'backgroundColor': 'dodgerblue',
            'color': 'white',
            'fontWeight': 'bold'
        },
        {
            'if': {
                'column_id': 'Result'
            },
            'backgroundColor': 'dodgerblue',
            'color': 'white',
            'fontWeight': 'bold'
        },
        {
            'if': {
                'column_id': 'LBW'
            },
            'backgroundColor': 'dodgerblue',
            'color': 'white',
            'fontWeight': 'bold'
        }
        ],
        style_filter={
            'textAlign': 'center'
        },
        style_as_list_view=True
    )

    return header, fav_sb


def generate_tracked_runs_selected_horse_table(xdf, horse):
    header = html.H3("Run details - {}".format(horse),
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    run_links = []
    # first_run_link = html.Div()
    # second_run_link = html.Div()
    # tenth_run_link = html.Div()
    # twentyfirst_run_link = html.Div()
    # twentieth_run_link = html.Div()
    # twonotone_run_link = html.Div()
    # if 1 in xdf['RunNo.'].unique():
    #     venue1 = xdf.loc[xdf['RunNo.']==1]['Venue ID']
    #     date1 = xdf.loc[xdf['RunNo.']==1]['Date']
    #     x1 = 'https://www.indiarace.com/Home/racingCenterEvent?venueId={}&event_date={}&race_type=RESULTS'.format(venue1,
    #                                                                                                              date1)
    #     print(x1)
    #     first_run_link = html.Span(html.A(children=[html.H6("Run 1 link",
    #                  style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
    #                         'font-family': 'Arial Black'})
    #                  ],
    #                  href=x1, target='_blank'), className="pretty_container",
    #         style={'display': 'inline-block', 'background-color': '#061e44', 'box-shadow': 'none', 'margin': '0px',
    #                'padding': '0px', 'width':'130px'})
    for sr in xdf['Sr #'].unique():
        # print(sr)
        venue = xdf.loc[xdf['Sr #'] == sr, 'Venue ID'].iloc[0]
        date = xdf.loc[xdf['Sr #'] == sr, 'Date'].iloc[0]
        race = xdf.loc[xdf['Sr #'] == sr, 'Race No'].iloc[0]
        # print(venue)
        # x = 'https://www.indiarace.com/Home/racingCenterEvent?venueId={}&event_date={}&race_type=RESULTS'.format(
        #     venue,
        #     date)
        x = 'https://www.indiarace.com/Home/oneResultByRaceNo?venueId={}&event_date={}&race_no={}'.format(venue, date,
                                                                                                          race)
        # print(x)
        run_link = html.Span(html.A(children=[html.H6("Sr# {} Link".format(sr),
                                                                style={'textAlign': 'center', 'font-weight': 'bold',
                                                                       'color': '#59ff00',
                                                                       'margin': '0px',
                                                                       'font-family': 'Arial Black'})
                                                        ],
                                              href=x, target='_blank'), className="pretty_container",
                                       style={'display': 'inline-block', 'background-color': '#061e44',
                                              'box-shadow': 'none', 'margin': '0px',
                                              'padding': '0px', 'width': '180px'})
        run_links.append(run_link)

    fav_sb = dash_table.DataTable(
        id='table4',
        columns=[{"name": i, "id": i, "hideable":True} for i in xdf.reset_index()[
            ['RunNo.','Horse', 'Sr #', 'Date', 'Trainer', 'Season', 'Race No', 'Class',
       'Distance', 'Age', 'Wt', 'Jockey', 'Dr', 'Result', 'LBW',
       'Penalty', 'Time', 'Sh', 'Centre']]],
        data=xdf.reset_index()[
            ['RunNo.', 'Horse', 'Sr #', 'Date', 'Trainer', 'Season', 'Race No', 'Class',
       'Distance', 'Age', 'Wt', 'Jockey', 'Dr', 'Result', 'LBW',
       'Penalty', 'Time', 'Sh', 'Centre']].to_dict(
            'records'),
        editable=False,
        #filterable = False
        filter_action="none",
        sort_action="native",
        sort_mode="multi",
        column_selectable=None,
        row_selectable=None,
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
        hidden_columns=['Trainer', 'Jockey', 'Class', 'Distance', 'Dr', 'Penalty', 'Time', 'Sh', 'Centre'],
        style_header={
            'backgroundColor': 'RebeccaPurple',
            'color': 'white',
            'fontWeight': 'bold',
            'font-size': 14,
            'textAlign': 'center'
        },
        style_cell={
            'backgroundColor': '#082255',
            'color': 'white',
            'border': '1px solid #bebebe',
            'font-size': 14,
            'textAlign': 'center',
            'minWidth': 25
        },
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'minHeight': '20px'
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
                'column_id': 'RunNo.'
            },
            'backgroundColor': 'dodgerblue',
            'color': 'white',
            'fontWeight': 'bold'
        },
        {
            'if': {
                'column_id': 'RunRes'
            },
            'backgroundColor': 'dodgerblue',
            'color': 'white',
            'fontWeight': 'bold'
        },
        {
            'if': {
                'column_id': 'Result'
            },
            'backgroundColor': 'dodgerblue',
            'color': 'white',
            'fontWeight': 'bold'
        },
        {
            'if': {
                'column_id': 'LBW'
            },
            'backgroundColor': 'dodgerblue',
            'color': 'white',
            'fontWeight': 'bold'
        }
        ],
        style_filter={
            'textAlign': 'center'
        },
        style_as_list_view=True
    )
    ret = [header, fav_sb]
    ret.extend(run_links)
    return ret


def generate_horse_run_result_summary_table(xdf, table_id, title):
    header = html.H3(title,
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    fav_sb = dash_table.DataTable(
        id=table_id,
        columns=[{"name": i, "id": i, "hideable":True} for i in xdf.reset_index()[
            ['RunNo.','Horse', 'Sr #', 'Date', 'Trainer', 'Season', 'Race No', 'Class',
       'Distance', 'Age', 'Wt', 'Jockey', 'Dr', 'Result', 'LBW',
       'Penalty', 'Time', 'Sh', 'Centre']]],
        data=xdf.reset_index()[
            ['RunNo.', 'Horse', 'Sr #', 'Date', 'Trainer', 'Season', 'Race No', 'Class',
       'Distance', 'Age', 'Wt', 'Jockey', 'Dr', 'Result', 'LBW',
       'Penalty', 'Time', 'Sh', 'Centre']].to_dict(
            'records'),
        # columns=[{"name": i, "id": i} for i in xdf[
        #     ['Horse', 'Distance', 'Jockey', 'Dr', 'Result']].reset_index().columns],
        # data=xdf[
        #     ['Horse', 'Distance', 'Jockey', 'Dr', 'Result']].reset_index().to_dict(
        #     'records'),
        page_size=20,
        editable=False,
        export_columns='all',
        export_format='xlsx',
        #filterable = False
        filter_action="none",
        sort_action="native",
        sort_mode="multi",
        column_selectable=None,
        row_selectable=None,
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
        hidden_columns=['Trainer', 'Jockey', 'Class', 'Distance', 'Dr', 'Penalty', 'Time', 'RunNo.', 'Sh', 'Centre'],
        style_header={
            'backgroundColor': 'RebeccaPurple',
            'color': 'white',
            'fontWeight': 'bold',
            'font-size': 14,
            'textAlign': 'center'
        },
        style_cell={
            'backgroundColor': '#082255',
            'color': 'white',
            'border': '1px solid #bebebe',
            'font-size': 14,
            'textAlign': 'center',
            'minWidth': 15
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
        style_data_conditional=(
            [
                {
                    'if': {
                        'filter_query': '{{{}}} = "Yes"'.format(col),
                        'column_id': col
                    },
                    'color': '#2ECC71',
                    'fontWeight': 'bold'
                } for col in xdf.columns
            ] +
            [
                {
                    'if': {
                        'filter_query': '{{{}}} = "No"'.format(col),
                        'column_id': col
                    },
                    'color': '#ff2200',
                    'fontWeight': 'bold'
                } for col in xdf.columns
            ] +
            [
                {
                    'if': {
                        'column_id': 'Horse'
                    },
                    'backgroundColor': 'dodgerBlue',
                    'fontWeight': 'bold'
                },
                {
                    'if': {
                        'column_id': 'Result'
                    },
                    'backgroundColor': 'dodgerBlue',
                    'fontWeight': 'bold'
                },
                {
                    'if': {
                        'column_id': 'Dr'
                    },
                    'backgroundColor': 'dodgerBlue',
                    'fontWeight': 'bold'
                }
            ]
        ),
        style_filter={
            'textAlign': 'center'
        },
        style_as_list_view=True
    )

    return header, fav_sb



filters = [
    html.P("Method:", className="control_label"),
    html.Div(
        dcc.Dropdown(
            id='Method',
            className='dcc_control',
            options=[{'label': 'Method 10 - NDD', 'value': 'Method 10 - NDD'}],
            value='Method 10 - NDD',
            clearable=False
        ),
        className='dash-dropdown'
    ),
    html.P("Qualification filters:", className="control_label"),
    html.P("LBW range", className="control_label"),
    html.Div(
        dcc.RangeSlider(
            id='LBWslider',
            min=0,
            max=20,
            step=0.5,
            value=[2.5, 4.5],
            pushable=0.5,
            tooltip={'always_visible':True, 'placement': 'bottom'}
            # marks = lbwslide_marks
        )
    ),
    html.Br(),
    # html.P("Result range", className="control_label"),
    # html.Div(
    #     dcc.RangeSlider(
    #         id='horsenoslider',
    #         min=2,
    #         max=20,
    #         step=1,
    #         value=[2, 3],
    #         allowCross=False,
    #         tooltip={'always_visible':True, 'placement': 'bottom'}
    #         # marks = resultslide_marks
    #         # marks = {
    #         #     2: '2',
    #         #     3: '3',
    #         #     4: '4',
    #         #     5: '5',
    #         #     6: '6',
    #         #     7: '7',
    #         #     8: '8',
    #         #     9: '9',
    #         #     10: '10',
    #         #     11: '11',
    #         #     12: '12',
    #         #     13: '13',
    #         #     14: '14',
    #         #     15: '15',
    #         #     16: '16',
    #         #     17: '17',
    #         #     18: '18',
    #         #     19: '19',
    #         #     20: '20',
    #         # }
    #     )
    # ),
    html.Div(
        dcc.Dropdown(
            id='horsenoslider',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in
                     sorted([x for x in result_valid if str(x) != 'nan'])],
            value=[],
            placeholder='Result',
            clearable=True,
            multi=True
            # disabled=True
        ),
        className='dash-dropdown'
    ),
    # html.Br(),
    html.Div(
        dcc.Dropdown(
            id='first_age',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Age'].unique()) if str(x) != 'nan'])],
            value=[],
            placeholder='First place horse age',
            clearable=True,
            multi=True
            # disabled=True
        ),
        className='dash-dropdown'
    ),
    # html.Br(),
    html.Div(
        dcc.Dropdown(
            id='Classification',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in data['Class'].unique()],
            value=[None],
            placeholder='Class',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Season',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Season'].unique()) if str(x) != 'nan'])],
            value=[None],
            placeholder='Season',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Dr',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Dr'].unique()) if str(x) != 'nan'])],
            value=[None],
            placeholder='Dr',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Sh',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in data['Sh'].unique()],
            value=[None],
            placeholder='Sh',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Age',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Age'].unique()) if str(x) != 'nan'])],
            value=[None],
            placeholder='Age',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    # html.Div(
    #     dcc.Dropdown(
    #         id='Venue',
    #         className='dcc_control',
    #         options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Venue ID'].unique()) if str(x) != 'nan'])],
    #         value=[None],
    #         placeholder='Venue ID',
    #         clearable=True,
    #         multi=True
    #     ),
    #     className='dash-dropdown'
    # ),
    html.Div(
        dcc.Dropdown(
            id='Centre',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Centre'].unique()) if str(x) != 'nan'])],
            value=[None],
            placeholder='Centre',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Distance',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Distance'].unique()) if str(x) != 'nan'])],
            value=[None],
            placeholder='Distance',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Trainer',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Trainer'].unique()) if str(x) != 'nan'])],
            value=[None],
            placeholder='Trainer',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.Dropdown(
            id='Jockey',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Jockey'].unique()) if str(x) != 'nan'])],
            value=[None],
            placeholder='Jockey',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    )
]

filters1 = [
    html.P("Run 1 filters:", className="control_label"),
    html.Div(
        dcc.Dropdown(
            id='Classification1',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in data['Class'].unique()],
            value=[],
            placeholder='Class',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Classification1check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Dr1',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Dr'].unique()) if str(x) != 'nan'])],
            value=[],
            placeholder='Dr',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Dr1check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Sh1',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in data['Sh'].unique()],
            value=[],
            placeholder='Sh',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Sh1check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Age1',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Age'].unique()) if str(x) != 'nan'])],
            value=[],
            placeholder='Age',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Age1check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Venue1',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Venue ID'].unique()) if str(x) != 'nan'])],
            value=[],
            placeholder='Venue ID',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Venue1check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Distance1',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Distance'].unique()) if str(x) != 'nan'])],
            value=[],
            placeholder='Distance',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Distance1check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Trainer1',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Trainer'].unique()) if str(x) != 'nan'])],
            value=[],
            placeholder='Trainer',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Trainer1check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Jockey1',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Jockey'].unique()) if str(x) != 'nan'])],
            value=[],
            placeholder='Jockey',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Jockey1check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    )
]

filters2 = [
    html.P("Run 2 filters:", className="control_label"),
    html.Div(
        dcc.Dropdown(
            id='Classification2',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in data['Class'].unique()],
            value=[],
            placeholder='Class',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Classification2check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Same as Run1', 'value': 'samerun1'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Dr2',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Dr'].unique()) if str(x) != 'nan'])],
            value=[],
            placeholder='Dr',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Dr2check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Same as Run1', 'value': 'samerun1'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Sh2',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in data['Sh'].unique()],
            value=[],
            placeholder='Sh',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Sh2check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Same as Run1', 'value': 'samerun1'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Age2',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Age'].unique()) if str(x) != 'nan'])],
            value=[],
            placeholder='Age',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Age2check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Same as Run1', 'value': 'samerun1'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Venue2',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Venue ID'].unique()) if str(x) != 'nan'])],
            value=[],
            placeholder='Venue ID',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Venue2check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Same as Run1', 'value': 'samerun1'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Distance2',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Distance'].unique()) if str(x) != 'nan'])],
            value=[],
            placeholder='Distance',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Distance2check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Same as Run1', 'value': 'samerun1'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Trainer2',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Trainer'].unique()) if str(x) != 'nan'])],
            value=[],
            placeholder='Trainer',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Trainer2check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Same as Run1', 'value': 'samerun1'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    ),
    html.Div(
        dcc.Dropdown(
            id='Jockey2',
            className='dcc_control',
            options=[{"label": i, "value": i} for i in sorted([x for x in list(data['Jockey'].unique()) if str(x) != 'nan'])],
            value=[],
            placeholder='Jockey',
            clearable=True,
            multi=True
        ),
        className='dash-dropdown'
    ),
    html.Div(
        dcc.RadioItems(
            id='Jockey2check',
            options=[
                {'label': 'Same as qualifying Run', 'value': 'same'},
                {'label': 'Same as Run1', 'value': 'samerun1'},
                {'label': 'Selected', 'value': 'any'}
            ],
            value='any'
        )
    )
]


# row0 = dbc.Row(
#     [
#         dbc.Col(
#             [
#                 html.Div(id='Overall_scoreboard', className='pretty_container', style={'display': 'initial'})
#             ],
#             width=12, className='pretty_container twelve columns', id='base_right-column'
#         )
#     ], id='fixedontop', className='flex-display'
# )

rules = ["1) Pick range for LBW. There should be no horses which finished in this range\n", html.Br(),
        "2) In the selected races, horses which finished below the LBW range are qualified  \n", html.Br(),
        "3) Number of horses(Results) can be varied. Other filters can be applied\n", html.Br(),
        "4) Observe next two immediate runs for qualified horses\n"]

row0 = dbc.Row(
    [
        dbc.Col(
            children=[html.H2("Dynamic Handicapping", id='title'),
                      html.Div(children=filters, id='cross-filter-options', className='pretty_container')],
            width=3, className='three columns', style={'display': 'flex', 'flex-direction': 'column'}
        ),
        dbc.Col(
            [
                html.Div(id='methodtitle', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboard', className='pretty_container', style={'display': 'initial'}),
                # html.Div(id='scoreboard15', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboard2', className='pretty_container', style={'display': 'initial'}),
                # html.Div(id='scoreboard25', className='pretty_container', style={'display': 'initial'},
                #          children=[html.H6("Rules",
                # style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                #        'font-family': 'Arial Black'}),
                #                    html.P(children=rules)])
                html.Div(id='scoreboard25', className='pretty_container', style={'display': 'initial'})
            ],
            width=9, className='nine columns', id='right-column'
        )
    ], className='flex-display'
)

row1 = dbc.Row(
    [
        dbc.Col(
            html.Div(id='tableslot1left', className='pretty_container'),
            width=4, className='pretty_container four columns', id='row1left', style={'display': 'initial'}
        ),
        dbc.Col(
            html.Div(id='tableslot1right', className='pretty_container'),
            width=8, className='pretty_container eight columns', id='row1right', style={'display': 'initial'}
        )
    ], className='flex-display'
)

row2= dbc.Row(
    [
        dbc.Col(
            html.Div(id='tableslot2', className='pretty_container'),
            width=12, className='pretty_container twelve columns', id='row2', style={'display': 'initial'}
        )
    ], className='flex-display'
)

row3= dbc.Row(
    [
        dbc.Col(
            html.Div(id='tableslot3', className='pretty_container'),
            width=12, className='pretty_container twelve columns', id='row3', style={'display': 'initial'}
        )
    ], className='flex-display'
)

row3point5run1score= dbc.Row(
    [
        dbc.Col(
            html.Div(id='scoreboardrun1', className='pretty_container', style={'display': 'initial'}),
            width=12, className='pretty_container twelve columns', id='row3point5', style={'display': 'initial'}
        )
    ], className='flex-display'
)

row4= dbc.Row(
    [
        dbc.Col(
            html.Div(children=filters1, id='cross-filter1-options', className='pretty_container'),
            width=3, className='three columns', id='row4left', style={'display': 'flex', 'flex-direction': 'column'}
        ),
        dbc.Col(
            html.Div(id='tableslot4', className='pretty_container'),
            width=9, className='pretty_container twelve columns', id='row4', style={'display': 'initial'}
        )
    ], className='flex-display'
)

row4point5run1score= dbc.Row(
    [
        dbc.Col(
            html.Div(id='scoreboardrun2', className='pretty_container', style={'display': 'initial'}),
            width=12, className='pretty_container twelve columns', id='row4point5', style={'display': 'initial'}
        )
    ], className='flex-display'
)

row5= dbc.Row(
    [
        dbc.Col(
            html.Div(children=filters2, id='cross-filter2-options', className='pretty_container'),
            width=3, className='three columns', id='row5left', style={'display': 'flex', 'flex-direction': 'column'}
        ),
        dbc.Col(
            html.Div(id='tableslot5', className='pretty_container'),
            width=9, className='pretty_container twelve columns', id='row5', style={'display': 'initial'}
        )
    ], className='flex-display'
)

row6= dbc.Row(
    [
        dbc.Col(
            html.Div(id='tableslot6', className='pretty_container'),
            width=12, className='pretty_container twelve columns', id='row6', style={'display': 'initial'}
        )
    ], className='flex-display'
)

row7= dbc.Row(
    [
        dbc.Col(
            html.Div(id='tableslot7', className='pretty_container'),
            width=12, className='pretty_container twelve columns', id='row7', style={'display': 'initial'}
        )
    ], className='flex-display'
)

row8= dbc.Row(
    [
        dbc.Col(
            html.Div(id='tableslot8', className='pretty_container'),
            width=12, className='pretty_container twelve columns', id='row8', style={'display': 'initial'}
        )
    ], className='flex-display'
)

app.layout = html.Div([
    row0,
    row1,
    row2,
    row3,
    row6,
    row3point5run1score,
    row4,
    row7,
    row4point5run1score,
    row5,
    row8,
    html.Div(id="store_original_df", children=[data.to_json(orient='split')], style={'display': 'none'}),
    html.Div(id="store_filtered_df", style={'display': 'none'}),
    html.Div(id="store_race_stats", style={'display': 'none'}),
    html.Div(id="store_init_qualifying_races", style={'display': 'none'}),
    html.Div(id="store_init_qualifying_races_detailed", style={'display': 'none'}),
    html.Div(id="store_init_qualifying_horses", style={'display': 'none'}),
    html.Div(id="store_qualifying_races", style={'display': 'none'}),
    html.Div(id="store_qualifying_races_detailed", style={'display': 'none'}),
    html.Div(id="store_qualifying_horses", style={'display': 'none'}),
    html.Div(id="store_Tracked_runs", style={'display': 'none'}),
    html.Div(id="store_run0_summary", style={'display': 'none'}),
    html.Div(id="store_run2_initial", style={'display': 'none'}),
    html.Div(id="store_run1_summary", style={'display': 'none'}),
    html.Div(id="store_run2_summary", style={'display': 'none'}),
    html.Div(id="store_run1_final", style={'display': 'none'}),
    html.Div(id="store_run2_final", style={'display': 'none'}),
    html.Div(id="store_ytr", style={'display': 'none'})],
    id='mainContainer', style={'display': 'flex', 'flex-direction': 'column'})


@app.callback(
    [Output('store_race_stats', 'children')],
    [Input('store_original_df', 'children')]
)
def update_race_stats(dff):
    try:
        df = pd.read_json(dff[0], orient='split')
    except:
        raise PreventUpdate
    # print("Here!!!!!!!")
    race_stats = generate_race_stats(df)
    # print(len(race_stats))
    return [race_stats.to_json(orient='split')]


@app.callback(
    [Output('store_init_qualifying_races', 'children'),
     Output('store_init_qualifying_races_detailed', 'children')],
    [Input('store_race_stats', 'children'),
     Input('LBWslider', 'value'),
     Input('first_age', 'value'),
     Input('store_original_df', 'children')]
)
def init_qual_races(race_stats_json, lbwlim, firstage, dff):
    # print(lbwlim)
    # print(race_stats_json[0])
    # race_stats = pd.read_json(race_stats_json[0], orient='split')
    try:
        df = pd.read_json(dff[0], orient='split')
        race_stats = pd.read_json(race_stats_json, orient='split')
    except Exception as e:
        print(e)
        raise PreventUpdate
    # print("Here!!!!!!!")
    initial_qualifying_races, initial_qualifying_races_detailed = generate_initial_qualifying_races(lbwlim[0], lbwlim[1], race_stats, df, firstage)
    # print(len(initial_qualifying_races), len(initial_qualifying_races_detailed))
    return [initial_qualifying_races.to_json(orient='split')], [initial_qualifying_races_detailed.index.names, initial_qualifying_races_detailed.reset_index().to_json(orient='split')]


@app.callback(
    [Output('store_init_qualifying_horses', 'children')],
    [Input('store_init_qualifying_races_detailed', 'children'),
     Input('horsenoslider', 'value')],
    [State('LBWslider', 'value')]
)
def init_qual_horses(initial_qualifying_races_detailed_json, horselim, lbwlim):
    try:
        initial_qualifying_races_detailed = pd.read_json(initial_qualifying_races_detailed_json[1], orient='split').set_index(initial_qualifying_races_detailed_json[0])
    except:
        raise PreventUpdate
    initial_qualified_horses = generate_initial_qualified_horses(initial_qualifying_races_detailed, lbwlim[0], horselim)
    return [initial_qualified_horses.to_json(orient='split')]


@app.callback(
    [Output('store_qualifying_horses', 'children')],
    [Input('store_init_qualifying_horses', 'children'),
     Input('Classification', 'value'),
     Input('Season', 'value'),
     Input('Sh', 'value'),
     Input('Dr', 'value'),
     Input('Age', 'value'),
     Input('Centre', 'value'),
     Input('Distance', 'value'),
     Input('Trainer', 'value'),
     Input('Jockey', 'value')]
)
def qual_horses(initial_qualifying_horse_json, class_list, season_list, sh_list, dr_list,age_list, venue_list, distance_list, trainer_list, jockey_list):
    try:
        initial_qualified_horses = pd.read_json(initial_qualifying_horse_json, orient='split')
    except:
        raise PreventUpdate
    if class_list == [None] or class_list == []:
        class_list = data['Class'].unique()
    if season_list == [None] or season_list == []:
        season_list = data['Season'].unique()
    if sh_list == [None] or sh_list == []:
        sh_list = data['Sh'].unique()
    if dr_list == [None] or dr_list == []:
        dr_list = data['Dr'].unique()
    if age_list == [None] or age_list == []:
        age_list = data['Age'].unique()
    if venue_list == [None] or venue_list == []:
        venue_list = data['Centre'].unique()
    if distance_list == [None] or distance_list == []:
        distance_list = data['Distance'].unique()
    if trainer_list == [None] or trainer_list == []:
        trainer_list = data['Trainer'].unique()
    if jockey_list == [None] or jockey_list == []:
        jockey_list = data['Jockey'].unique()
    a = ['Class', 'Season', 'Age', 'Sh','Dr', 'Centre', 'Distance', 'Trainer', 'Jockey']
    b = [class_list, season_list, age_list, sh_list, dr_list, venue_list, distance_list, trainer_list, jockey_list]
    cond = []
    condition = []
    for col, filt in zip(a, b):
        cond.append(initial_qualified_horses[col].isin(filt))
        condition.append(conjunction(0, *cond))
    if condition:
        qualified_horses = initial_qualified_horses[conjunction(0, *condition)]
    # print(len(qualified_horses))
    return [qualified_horses.to_json(orient='split')]


@app.callback(
    [Output('store_qualifying_races', 'children'),
     Output('store_qualifying_races_detailed', 'children'),
     Output('store_Tracked_runs', 'children')
     ],
    [Input('store_qualifying_horses', 'children')],
    [State('store_original_df', 'children')]
)
def qual_races(qualifying_horse_json, dff):
    try:
        df = pd.read_json(dff[0], orient='split')
        qualified_horses = pd.read_json(qualifying_horse_json, orient='split')
    except:
        raise PreventUpdate
    qualifying_races = generate_qualifying_races(qualified_horses)
    qualifying_races_detailed = generate_qualifying_races_detailed(qualifying_races, df)
    # print(len(qualifying_races))
    # print("Here!!!!!!!")
    # print(len(qualifying_races))
    TrackedRuns = generate_TrackedRuns(df, qualified_horses)
    return [qualifying_races.to_json(orient='split')], [qualifying_races_detailed.index.names, qualifying_races_detailed.reset_index().to_json(orient='split')], [TrackedRuns.to_json(orient='split')]


@app.callback(
    [Output('store_run0_summary', 'children'),
     Output('store_run1_summary', 'children'),
     Output('store_run2_initial', 'children'),
     Output('store_ytr', 'children')],
    [Input('store_Tracked_runs', 'children')]
)
def tracked_runs(trackedRuns_json):
    try:
        TrackedRuns = pd.read_json(trackedRuns_json[0], orient='split')
    except:
        raise PreventUpdate
    Run0_df, Run1_df, Run2_df, ytr = generate_qualified_horse_summary(TrackedRuns)
    print("Run1", len(Run1_df), "run2", len(Run2_df), "ytr", len(ytr))
    return [Run0_df.to_json(orient='split')], [Run1_df.to_json(orient='split')], [Run2_df.to_json(orient='split')], [ytr.to_json(orient='split')]

'''********************************************************************************************'''

@app.callback(
    [Output('Age1', 'value'),
     Output('Age1', 'disabled')],
    [Input('Age1check', 'value')],
    [State('Age1', 'value')],
)
def modCheck(age1check, age1_list):
    # print(age1check, age1_list)
    if age1check == 'same' or age1check == 'samerun1':
        return [], True
    return age1_list, False



@app.callback(
    [Output('Classification1', 'value'),
     Output('Classification1', 'disabled')],
    [Input('Classification1check', 'value')],
    [State('Classification1', 'value')],
)
def modCheck(class1check, class1_list):
    # print(class1check, class1_list)
    if class1check == 'same' or class1check == 'samerun1':
        return [], True
    return class1_list, False



@app.callback(
    [Output('Sh1', 'value'),
     Output('Sh1', 'disabled')],
    [Input('Sh1check', 'value')],
    [State('Sh1', 'value')],
)
def modCheck(sh1check, sh1_list):
    # print(class1check, class1_list)
    if sh1check == 'same' or sh1check == 'samerun1':
        return [], True
    return sh1_list, False


@app.callback(
    [Output('Dr1', 'value'),
     Output('Dr1', 'disabled')],
    [Input('Dr1check', 'value')],
    [State('Dr1', 'value')],
)
def modCheck(dr1check, dr1_list):
    # print(class1check, class1_list)
    if dr1check == 'same' or dr1check == 'samerun1':
        return [], True
    return dr1_list, False



@app.callback(
    [Output('Venue1', 'value'),
     Output('Venue1', 'disabled')],
    [Input('Venue1check', 'value')],
    [State('Venue1', 'value')],
)
def modCheck(venue1check, venue1_list):
    # print(venue1check, venue1_list)
    if venue1check == 'same' or venue1check == 'samerun1':
        return [], True
    return venue1_list, False


@app.callback(
    [Output('Distance1', 'value'),
     Output('Distance1', 'disabled')],
    [Input('Distance1check', 'value')],
    [State('Distance1', 'value')],
)
def modCheck(distance1check, distance1_list):
    # print(distance1check, distance1_list)
    if distance1check == 'same' or distance1check == 'samerun1':
        return [], True
    return distance1_list, False


@app.callback(
    [Output('Trainer1', 'value'),
     Output('Trainer1', 'disabled')],
    [Input('Trainer1check', 'value')],
    [State('Trainer1', 'value')],
)
def modCheck(trainer1check, trainer1_list):
    # print(trainer1check, trainer1_list)
    if trainer1check == 'same' or trainer1check == 'samerun1':
        return [], True
    return trainer1_list, False


@app.callback(
    [Output('Jockey1', 'value'),
     Output('Jockey1', 'disabled')],
    [Input('Jockey1check', 'value')],
    [State('Jockey1', 'value')],
)
def modCheck(jockey1check, jockey1_list):
    # print(jockey1check, jockey1_list)
    if jockey1check == 'same' or jockey1check == 'samerun1':
        return [], True
    return jockey1_list, False

'''********************************************************************************************'''

@app.callback(
    [Output('Age2', 'value'),
     Output('Age2', 'disabled')],
    [Input('Age2check', 'value')],
    [State('Age2', 'value')],
)
def modCheck(age2check, age2_list):
    # print(age2check, age2_list)
    if age2check == 'same' or age2check == 'samerun1':
        return [], True
    return age2_list, False



@app.callback(
    [Output('Classification2', 'value'),
     Output('Classification2', 'disabled')],
    [Input('Classification2check', 'value')],
    [State('Classification2', 'value')],
)
def modCheck(class2check, class2_list):
    print(class2check, class2_list)
    if class2check == 'same' or class2check == 'samerun1':
        return [], True
    return class2_list, False


@app.callback(
    [Output('Dr2', 'value'),
     Output('Dr2', 'disabled')],
    [Input('Dr2check', 'value')],
    [State('Dr2', 'value')],
)
def modCheck(dr2check, dr2_list):
    print(dr2check, dr2_list)
    if dr2check == 'same' or dr2check == 'samerun1':
        return [], True
    return dr2_list, False


@app.callback(
    [Output('Sh2', 'value'),
     Output('Sh2', 'disabled')],
    [Input('Sh2check', 'value')],
    [State('Sh2', 'value')],
)
def modCheck(sh2check, sh2_list):
    print(sh2check, sh2_list)
    if sh2check == 'same' or sh2check == 'samerun1':
        return [], True
    return sh2_list, False


@app.callback(
    [Output('Venue2', 'value'),
     Output('Venue2', 'disabled')],
    [Input('Venue2check', 'value')],
    [State('Venue2', 'value')],
)
def modCheck(venue2check, venue2_list):
    # print(venue2check, venue2_list)
    if venue2check == 'same' or venue2check == 'samerun1':
        return [], True
    return venue2_list, False


@app.callback(
    [Output('Distance2', 'value'),
     Output('Distance2', 'disabled')],
    [Input('Distance2check', 'value')],
    [State('Distance2', 'value')],
)
def modCheck(distance2check, distance2_list):
    # print(distance2check, distance2_list)
    if distance2check == 'same' or distance2check == 'samerun1':
        return [], True
    return distance2_list, False


@app.callback(
    [Output('Trainer2', 'value'),
     Output('Trainer2', 'disabled')],
    [Input('Trainer2check', 'value')],
    [State('Trainer2', 'value')],
)
def modCheck(trainer2check, trainer2_list):
    # print(trainer2check, trainer2_list)
    if trainer2check == 'same' or trainer2check == 'samerun1':
        return [], True
    return trainer2_list, False


@app.callback(
    [Output('Jockey2', 'value'),
     Output('Jockey2', 'disabled')],
    [Input('Jockey2check', 'value')],
    [State('Jockey2', 'value')],
)
def modCheck(jockey2check, jockey2_list):
    # print(jockey2check, jockey2_list)
    if jockey2check == 'same' or jockey2check == 'samerun1':
        return [], True
    return jockey2_list, False


'''********************************************************************************************'''

@app.callback(
    [Output('store_run1_final', 'children'),
     Output('store_run2_summary', 'children')],
    [Input('Age1', 'value'),
     Input('Age1check', 'value'),
     Input('Classification1', 'value'),
     Input('Classification1check', 'value'),
     Input('Dr1', 'value'),
     Input('Dr1check', 'value'),
     Input('Sh1', 'value'),
     Input('Sh1check', 'value'),
     Input('Venue1', 'value'),
     Input('Venue1check', 'value'),
     Input('Distance1', 'value'),
     Input('Distance1check', 'value'),
     Input('Trainer1', 'value'),
     Input('Trainer1check', 'value'),
     Input('Jockey1', 'value'),
     Input('Jockey1check', 'value'),
     Input('store_run1_summary', 'children'),
     Input('store_run2_initial', 'children')
     ],
    [State('store_Tracked_runs', 'children')]
)
def modRun1(age_list, age_radio, class_list, class_radio,dr_list,dr_radio, sh_list, sh_radio, venue_list, venue_radio, distance_list, distance_radio, trainer_list, trainer_radio, jockey_list, jockey_radio,run1_json, run2_json, trackedRuns_json):
    print("called now:")
    try:
        TrackedRuns = pd.read_json(trackedRuns_json[0], orient='split')
        Run1df = pd.read_json(run1_json[0], orient='split')
        Run2df = pd.read_json(run2_json[0], orient='split')
    except Exception as e:
        print("this",e)
        raise PreventUpdate
    # TrackedRuns = pd.read_json(trackedRuns_json[0], orient='split')
    # print("TRlen", len(TrackedRuns))
    # Run1df = pd.read_json(run1_json[0], orient='split')
    # print("R1len", len(Run1df))
    # Run2df = pd.read_json(run2_json[0], orient='split')
    # print("read successful")
    # runone_list = [x for x in TrackedRuns['RunNo.'].unique() if '1' in str(x)]

    a = ['Class', 'Age', 'Dr','Sh', 'Venue ID', 'Distance', 'Trainer', 'Jockey']
    b = [class_list, age_list, dr_list, sh_list, venue_list, distance_list, trainer_list, jockey_list]
    c = [class_radio, age_radio, dr_radio, sh_radio, venue_radio, distance_radio, trainer_radio, jockey_radio]

    firstindex = Run1df.index
    print("Or1", len(firstindex))
    same_indexes = []
    for i in firstindex:
        flag=True
        for x,y,z in zip(a,b,c):
            if y==[] and z=='same':
                if (TrackedRuns.loc[i, x] == TrackedRuns.loc[i - 1, x]):
                    continue
                else:
                    flag = False
                    break
        if flag:
            same_indexes.append(i)

    Run1_dfnew = Run1df.loc[same_indexes]
    print("run1", len(Run1_dfnew))
    originalRun1 = Run1_dfnew.index
    cond = []
    condition = []
    for col, filt, type in zip(a, b, c):
        if type=='any' and filt!=[]:
            print(type, filt)
            cond.append(Run1_dfnew[col].isin(filt))
            condition.append(conjunction(0, *cond))
    if condition:
        Run1_dfnew = Run1_dfnew[conjunction(0, *condition)]
    dropped_indexes_run1 = [x for x in originalRun1 if x not in Run1_dfnew.index]
    print("samerun1:",len(same_indexes))
    pd.to_datetime(Run1_dfnew['Date'])

    run2new = []
    originalRun2 = Run2df.index
    print("or2", len(originalRun2))
    for i in Run1_dfnew.index:
        if i + 1 in originalRun2:
            run2new.append(i + 1)
    Run2_dfnew = Run2df.loc[run2new]
    print("run1", len(Run1_dfnew),"run2", len(Run2_dfnew))
    pd.to_datetime(Run2_dfnew['Date'])
    return [Run1_dfnew.to_json(orient='split')], [Run2_dfnew.to_json(orient='split')]

'''********************************************************************************************'''
@app.callback(
    [Output('store_run2_final', 'children')],
    [Input('Age2', 'value'),
     Input('Age2check', 'value'),
     Input('Classification2', 'value'),
     Input('Classification2check', 'value'),
     Input('Dr2', 'value'),
     Input('Dr2check', 'value'),
     Input('Sh2', 'value'),
     Input('Sh2check', 'value'),
     Input('Venue2', 'value'),
     Input('Venue2check', 'value'),
     Input('Distance2', 'value'),
     Input('Distance2check', 'value'),
     Input('Trainer2', 'value'),
     Input('Trainer2check', 'value'),
     Input('Jockey2', 'value'),
     Input('Jockey2check', 'value'),
     Input('store_run2_summary', 'children')
     ],
    [State('store_Tracked_runs', 'children')]
)
def modRun2(age_list, age_radio, class_list, class_radio,dr_list,dr_radio,sh_list,sh_radio, venue_list, venue_radio, distance_list, distance_radio, trainer_list, trainer_radio, jockey_list, jockey_radio,run2_json,trackedRuns_json):
    print("callednow2")
    try:
        TrackedRuns = pd.read_json(trackedRuns_json[0], orient='split')
        Run2df = pd.read_json(run2_json[0], orient='split')
    except Exception as e:
        print(e)
        raise PreventUpdate

    # runtwo_list = [x for x in TrackedRuns['RunNo.'].unique() if '2' in str(x)]

    a = ['Class', 'Age', 'Dr', 'Sh', 'Venue ID', 'Distance', 'Trainer', 'Jockey']
    b = [class_list, age_list, dr_list, sh_list, venue_list, distance_list, trainer_list, jockey_list]
    c = [class_radio, age_radio, dr_radio, sh_radio, venue_radio, distance_radio, trainer_radio, jockey_radio]

    secondindex = Run2df.index
    same_indexes = []
    for i in secondindex:
        flag=True
        for x,y,z in zip(a,b,c):
            if y==[] and z=='samerun1':
                if (TrackedRuns.loc[i, x] == TrackedRuns.loc[i - 1, x]):
                    continue
                else:
                    flag = False
                    break
            elif y==[] and z=='same':
                if (TrackedRuns.loc[i, x] == TrackedRuns.loc[i - 2, x]):
                    continue
                else:
                    flag = False
                    break
        if flag:
            same_indexes.append(i)

    Run2_dfnew = Run2df.loc[same_indexes]
    originalRun2 = Run2_dfnew.index
    cond = []
    condition = []
    for col, filt, type in zip(a, b, c):
        if type=='any' and filt!=[]:
            cond.append(Run2_dfnew[col].isin(filt))
            condition.append(conjunction(0, *cond))
    if condition:
        Run2_dfnew = Run2_dfnew[conjunction(0, *condition)]
    dropped_indexes_run2 = [x for x in originalRun2 if x not in Run2_dfnew.index]
    print("dropped:",len(same_indexes))
    pd.to_datetime(Run2_dfnew['Date'])

    return [[Run2_dfnew.to_json(orient='split')]]



@app.callback(
    [Output('methodtitle', 'children'),
     Output('scoreboard', 'children'),
     # Output('scoreboard15', 'children'),
     # Output('scoreboard15', 'style'),
     Output('scoreboard2', 'children'),
     Output('scoreboard2', 'style'),
     Output('scoreboard25', 'children'),
     Output('tableslot1left', 'children'),
     Output('tableslot1right', 'children'),
     Output('row1', 'style'),
     Output('tableslot2', 'children'),
     Output('row2', 'style'),
     Output('tableslot3', 'children'),
     Output('row3', 'style'),
     Output('tableslot4', 'children'),
     Output('row4', 'style'),
     Output('tableslot5', 'children'),
     Output('row5', 'style'),
     Output('row3point5run1score', 'style'),
     Output('scoreboardrun1', 'children'),
     Output('row4point5run1score', 'style'),
     Output('scoreboardrun2', 'children'),
     Output('tableslot6', 'children'),
     Output('row6', 'style'),
     Output('tableslot7', 'children'),
     Output('row7', 'style'),
     Output('tableslot8', 'children'),
     Output('row8', 'style')],
     [Input('store_original_df', 'children'),
     Input('store_race_stats', 'children'),
     Input("store_init_qualifying_races", 'children'),
     Input("store_init_qualifying_races_detailed", 'children'),
     Input("store_init_qualifying_horses", 'children'),
     Input("store_qualifying_races", 'children'),
     Input("store_qualifying_races_detailed", 'children'),
     Input("store_qualifying_horses", 'children'),
     Input("store_Tracked_runs", 'children'),
     Input("store_run0_summary", 'children'),
     Input("store_run1_final", 'children'),
     Input("store_run2_final", 'children'),
     Input("store_run1_summary", 'children'),
     Input("store_run2_initial", 'children'),
     Input("store_ytr", 'children'),
     Input('Method', 'value')]
)
def update_main(dff,race_stats_json,init_qual_races_json,init_qual_races_detailed_json,init_qual_horses_json,qual_races_json,qual_races_detailed_json,qual_horses_json,trackedRuns_json, run0_json, run1_json, run2_json, run1_overall,run2_overall,ytr_json, Method):
    # print('Here!!!!!!!!!!!!!', len(run2_json),'....',len(run1_json))
    if Method is None:
        print("Hello")
        raise PreventUpdate
    try:
        # df = pd.read_json(dff[0], orient='split')
        race_stats = pd.read_json(race_stats_json, orient='split')
        qualifying_races = pd.read_json(qual_races_json[0], orient='split')
    except Exception as e:
        print(e)
        raise PreventUpdate
    # print('Here!!!!!!!!!!!!!')
    # initial_qualifying_races = pd.read_json(init_qual_races_json[0], orient='split')
    # print("QualRaces", len(qualifying_races))
    if len(qualifying_races) == 0:
        return [[],'No Qualified Horses', [], {'display': 'none'}, [],[], [], {'display': 'none'}, [], {'display': 'none'}, [], {'display': 'none'}, [], {'display': 'none'}, [], {'display': 'none'}, {'display': 'none'}, [], {'display': 'none'}, [], [], {'display': 'none'}, [], {'display': 'none'}, [], {'display': 'none'}]


    try:
        qualifying_races['Date'] = qualifying_races['Date'].dt.date
        # initial_qualifying_races_detailed = pd.read_json(init_qual_races_detailed_json[1],orient='split').set_index(init_qual_races_detailed_json[0])
        qualifying_races_detailed = pd.read_json(qual_races_detailed_json[1],orient='split').set_index(qual_races_detailed_json[0])
        qualifying_races_detailed['LBW'] = round(qualifying_races_detailed['LBW'],2)
        qualifying_races_detailed['Time'] = round(qualifying_races_detailed['Time'], 3)
        # qualifying_races_detailed['Date'] = qualifying_races_detailed['Date'].dt.date
        # initial_qualified_horses = pd.read_json(init_qual_horses_json[0], orient='split')
        qualified_horses = pd.read_json(qual_horses_json, orient='split')
        qualified_horses['LBW'] = round(qualified_horses['LBW'],2)
        qualified_horses['Time'] = round(qualified_horses['Time'], 3)
        qualified_horses['Date'] = qualified_horses['Date'].dt.date
        TrackedRuns = pd.read_json(trackedRuns_json[0], orient='split')
        TrackedRuns['LBW'] = round(TrackedRuns['LBW'],2)
        TrackedRuns['Time'] = round(TrackedRuns['Time'], 3)
        TrackedRuns['Date'] = TrackedRuns['Date'].dt.date
        ytr = pd.read_json(ytr_json[0], orient='split')
        ytr['LBW'] = round(ytr['LBW'],2)
        ytr['Time'] = round(ytr['Time'], 3)
        ytr['Date'] = ytr['Date'].dt.date
        Run0df = pd.read_json(run0_json[0], orient='split')
        Run0df['LBW'] = round(Run0df['LBW'],2)
        Run0df['Time'] = round(Run0df['Time'], 3)
        Run0df['Date'] = Run0df['Date'].dt.date
        Run1df = pd.read_json(run1_json[0], orient='split')
        Run1df['LBW'] = round(Run1df['LBW'],2)
        Run1df['Time'] = round(Run1df['Time'], 3)
        if len(Run1df)>0:
            Run1df['Date'] = Run1df['Date'].dt.date
        Run2df = pd.read_json(run2_json[0], orient='split')
        Run2df['LBW'] = round(Run2df['LBW'],2)
        Run2df['Time'] = round(Run2df['Time'], 3)
        if len(Run2df)>0:
            Run2df['Date'] = Run2df['Date'].dt.date
        Run1df_overall = pd.read_json(run1_overall[0], orient='split')
        Run1df_overall['LBW'] = round(Run1df_overall['LBW'],2)
        Run1df_overall['Time'] = round(Run1df_overall['Time'], 3)
        if len(Run1df_overall)>0:
            Run1df_overall['Date'] = Run1df_overall['Date'].dt.date
        Run2df_overall = pd.read_json(run2_overall[0], orient='split')
        Run2df_overall['LBW'] = round(Run2df_overall['LBW'],2)
        Run2df_overall['Time'] = round(Run2df_overall['Time'], 3)
        if len(Run2df_overall)>0:
            Run2df_overall['Date'] = Run2df_overall['Date'].dt.date
    except Exception as e:
        print(e)
        raise PreventUpdate

    run1dist = generate_specific_stats(Run1df, 'Distance')
    run1seas = generate_specific_stats(Run1df, 'Season')
    run1dr = generate_specific_stats(Run1df, 'Dr')
    run1sh = generate_specific_stats(Run1df, 'Sh')
    run1class = generate_specific_stats(Run1df, 'Class')
    run1age = generate_specific_stats(Run1df, 'Age')
    run1centre = generate_specific_stats(Run1df, 'Centre')
    run1trainer = generate_specific_stats(Run1df, 'Trainer')
    run1jockey = generate_specific_stats(Run1df, 'Jockey')

    run1sep_return = generate_specific_scoreboards(run1dist, 'Distance')+generate_specific_scoreboards(run1seas, 'Season')+generate_specific_scoreboards(run1dr, 'Dr')+generate_specific_scoreboards(run1sh, 'Sh')+generate_specific_scoreboards(run1class, 'Class')+generate_specific_scoreboards(run1age, 'Age')+generate_specific_scoreboards(run1centre, 'Centre')+generate_specific_scoreboards(run1trainer, 'Trainer')+generate_specific_scoreboards(run1jockey, 'Jockey')

    run2dist = generate_specific_stats(Run2df, 'Distance')
    run2seas = generate_specific_stats(Run2df, 'Season')
    run2dr = generate_specific_stats(Run2df, 'Dr')
    run2sh = generate_specific_stats(Run2df, 'Sh')
    run2class = generate_specific_stats(Run2df, 'Class')
    run2age = generate_specific_stats(Run2df, 'Age')
    run2centre = generate_specific_stats(Run2df, 'Centre')
    run2trainer = generate_specific_stats(Run2df, 'Trainer')
    run2jockey = generate_specific_stats(Run2df, 'Jockey')

    run2sep_return = generate_specific_scoreboards(run2dist, 'Distance') + generate_specific_scoreboards(run2seas,
                                                                                                         'Season') + generate_specific_scoreboards(
        run2dr, 'Dr') + generate_specific_scoreboards(run2sh, 'Sh') + generate_specific_scoreboards(run2class,
                                                                                                    'Class') + generate_specific_scoreboards(
        run2age, 'Age') + generate_specific_scoreboards(run2centre, 'Centre') + generate_specific_scoreboards(
        run2trainer, 'Trainer') + generate_specific_scoreboards(run2jockey, 'Jockey')


    score25_return1 = [dcc.Graph(figure=generatebaroverall(Run1df, 'Run1'), style={'width': '50%', 'display' :'inline-block'}), dcc.Graph(figure=generatebaroverall(Run2df,'Run2'), style={'width': '50%', 'display' :'inline-block'})]
    score_return = generate_complete_scoreboard(Run0df, Run1df_overall, Run2df_overall, race_stats, qualifying_races)
    # score15_return = generate_runs_summary_scoreboard(qualified_horse_summary)
    # score15_style = {'display': 'initial'}
    score2_return = generate_overall_scoreboard_for_combo(Run1df_overall, Run2df_overall)
    score2_style = {'display': 'initial'}
    table1left = generate_qual_race_summary_table(qualifying_races)
    table1right = generate_qual_races_details_table(
        qualifying_races_detailed.loc[(
        qualifying_races.iloc[0]['Date'], qualifying_races.iloc[0]['Season'], qualifying_races.iloc[0]['Race No'])],
        qualifying_races.iloc[0]['Date'],
        qualifying_races.iloc[0]['Season'],
        qualifying_races.iloc[0]['Race No'],
        qualifying_races.iloc[0]['Venue ID']
    )
    row1style = {'display': 'initial'}
    table2 = generate_qualified_horse_list_table(qualified_horses)
    row2_style = {'display': 'initial'}
    table3 = generate_tracked_runs_selected_horse_table(
        TrackedRuns.loc[TrackedRuns['Horse']==qualified_horses.iloc[0]['Horse']],
        qualified_horses.iloc[0]['Horse']
    )
    row3_style = {'display': 'initial'}
    table4 = generate_horse_run_result_summary_table(Run1df, 'table5', 'Run 1')
    row4_style = {'display': 'initial'}
    table5 = generate_horse_run_result_summary_table(Run2df, 'table6', 'Run 2')
    row5_style = {'display': 'initial'}
    scorerun1 = generate_run1run2_scoreboard(Run1df_overall, Run1df, "Run 1")
    row3point5style = {'display': 'initial'}
    scorerun2 = generate_run1run2_scoreboard(Run2df_overall, Run2df, "Run 2")
    row4point5style = {'display': 'initial'}
    table6 = generate_horse_run_result_summary_table(ytr, 'table7', 'Yet To Run')
    row6_style = {'display': 'initial'}
    row7_style = {'display': 'initial'}
    row8_style = {'display': 'initial'}

    return [html.H2("{} method".format(Method),
                style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#00ffdd', 'margin': '0px',
                       'font-family': 'Arial Black'}),
        score_return, score2_return, score2_style, score25_return1, table1left, table1right, row1style, table2, row2_style, table3, row3_style,
            table4, row4_style, table5, row5_style, row3point5style, scorerun1, row4point5style, scorerun2, table6, row6_style, run1sep_return, row7_style, run2sep_return, row8_style]


@app.callback(
    [Output('tableslot1right', 'children')],
    [Input('table', 'active_cell')],
    [State('table', 'derived_viewport_data'),
     State('store_qualifying_races', 'children'),
     State('store_qualifying_races_detailed', 'children'),
     State('store_qualifying_horses', 'children'),
     State('store_Tracked_runs', 'children')]
)
def updateracedetailed(active_cell, data, qual_races, qual_races_detailed, qual_horses, TRuns):
    if active_cell is None:
        raise PreventUpdate
    # print(active_cell)
    # qualifying_races = pd.read_json(qual_races[0], orient='split')
    qualifying_races_detailed = pd.read_json(qual_races_detailed[1], orient='split').set_index(qual_races_detailed[0])
    qualifying_races_detailed['LBW'] = round(qualifying_races_detailed['LBW'], 2)
    # qualified_horses = pd.read_json(qual_horses[0], orient='split')
    # TrackedRuns = pd.read_json(TRuns[0], orient='split')
    row = active_cell['row']
    # column_id = active_cell['column_id']
    date = data[row]['Date']
    season = data[row]['Season']
    rno = data[row]['Race No']
    venue = data[row]['Venue ID']
    table1right = generate_qual_races_details_table(
            qualifying_races_detailed.loc[(date, season, rno)],
            date,
            season,
            rno,
        venue
        )
    return [table1right]

@app.callback(
    [Output('tableslot3', 'children')],
    [Input('table3', 'active_cell')],
    [State('table3', 'derived_viewport_data'),
     State('store_qualifying_races', 'children'),
     State('store_qualifying_races_detailed', 'children'),
     State('store_qualifying_horses', 'children'),
     State('store_Tracked_runs', 'children')]
)
def updatehorseruns(active_cell, data, qual_races, qual_races_detailed, qual_horses, TRuns):
    if active_cell is None:
        raise PreventUpdate
    # print(active_cell)
    # qualifying_races = pd.read_json(qual_races[0], orient='split')
    # qualifying_races_detailed = pd.read_json(qual_races_detailed[1], orient='split').set_index(qual_races_detailed[0])
    # qualified_horses = pd.read_json(qual_horses[0], orient='split')
    TrackedRuns = pd.read_json(TRuns[0], orient='split')
    TrackedRuns['LBW'] = round(TrackedRuns['LBW'],2)
    TrackedRuns['Time'] = round(TrackedRuns['Time'], 3)
    TrackedRuns['Date'] = TrackedRuns['Date'].dt.date
    row = active_cell['row']
    # column_id = active_cell['column_id']
    horse = data[row]['Horse']
    table3 = generate_tracked_runs_selected_horse_table(
        TrackedRuns.loc[TrackedRuns['Horse'] == horse],
        horse
    )
    return [table3]

if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)