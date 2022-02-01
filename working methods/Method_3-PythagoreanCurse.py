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



'''********************************************************'''
data = pd.read_excel(r'data\NEW METHOD 210822.xlsx')

data.rename(columns = {'Center':'Season', 'Horse Name': 'Horse', 'Dist': 'Distance', }, inplace = True)

data.Age.replace([np.nan, '3y', '5y', '4y', '6y', '7y', '8y', '9y', '2y'],[0, 3, 5, 4, 6, 7, 8, 9, 2],inplace=True)
data.Result.replace(['DNC', 'NC', 'DNF', 'DR', 'W'],[0,0,0,0,0], inplace=True)
data.Jockey.replace([np.nan], [""],inplace=True)
data.Trainer.replace([np.nan], [""],inplace=True)
data['LBW'].replace([np.nan, 'W'], [0,0], inplace=True)
data['Sh'].replace(['s', 'a', np.nan], ['S','A','-'], inplace=True)
try:
    data['Time in S'].replace([np.nan, 'DNF'], [0,0], inplace=True)
except:
    pass
data['RunNo.']=[-1]*len(data)

data['Dr'].replace([np.nan], [0], inplace=True)
data['Dr'] = data.Dr.astype(int)

data = data.loc[data['Distance']==1200]

data.drop(columns=['Time', 'Odds', 'Rtg'], inplace=True)
data.rename(columns = {'Time in S':'Time'}, inplace=True)
data['Date'] = data['Date'].dt.date

new = data['Venue ID'].str.split("=", n = 1, expand = True)
data['Venue ID'] = new[1]
data['Venue ID'] = data['Venue ID'].astype(int)
data['Venue Name'] = data['Venue ID'].replace([1,2,3,4,7,8,9,10,11], ['KOL', 'MUM', 'BLR', 'CHE', 'DEL', 'MYS', 'OOT', 'PUN', 'HYD'])
'''**************************************************************************'''

app = dash.Dash(__name__)

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
    race_stats = df.groupby(["Date", "Venue ID", "Season", "Race No", "Result"], as_index=True).size().unstack(
        fill_value=0)
    race_stats.reset_index(inplace=True)
    race_stats.index = np.arange(1, len(race_stats) + 1)
    return race_stats


def generate_qualifying_races(qualified_horses):
    qualifying_races = qualified_horses[["Date", "Season", "Race No", "Venue ID"]]
    return qualifying_races


def generate_qualifying_races_detailed(qualifying_races, df):
    qualifying_races_detailed = pd.merge(qualifying_races, df, on=["Date", "Season", "Race No"])
    qualifying_races_detailed.set_index(["Date", "Season", "Race No"], inplace=True)
    return qualifying_races_detailed


def generate_qualified_horses(df):
    qualified_horses = df.loc[(df['Dr'] >= 10) & (df['LBW'] >= 3.5) & (df['LBW'] <= 4)][
        ['Horse', 'Date', 'Season', 'Race No', 'Sr #', 'Trainer', 'Class', 'Distance', 'Age',
         'Wt', 'Jockey', 'Dr', 'Result', 'LBW', 'Penalty','Time','RunNo.', 'Venue ID', 'Centre']]
    qualified_horses.index = np.arange(1, len(qualified_horses) + 1)
    return qualified_horses

def generate_TrackedRuns(df, qualified_horses):
    for i in qualified_horses['Sr #']:
        df.loc[i - 1, 'RunNo.'] = 0
    races_Q_horses = pd.merge(qualified_horses[['Horse']], df, on=["Horse"])
    zerothindex = races_Q_horses.index[races_Q_horses['RunNo.'] == 0].tolist()
    removed_horses = []
    for i in zerothindex:
        if i + 1 in races_Q_horses.index:
            if (races_Q_horses.loc[i + 1, 'Horse'] == races_Q_horses.loc[i, 'Horse'] and races_Q_horses.loc[i + 1, 'Dr'] < 10):
                races_Q_horses.loc[i + 1, 'RunNo.'] = 1
            elif (races_Q_horses.loc[i + 1, 'Horse'] == races_Q_horses.loc[i, 'Horse'] and races_Q_horses.loc[i + 1, 'Dr'] >= 10):
                removed_horses.append(races_Q_horses.loc[i, 'Horse'])
                races_Q_horses = races_Q_horses.loc[races_Q_horses['Horse'] != races_Q_horses.loc[i, 'Horse']]
    TrackedRuns = races_Q_horses.loc[races_Q_horses['RunNo.'] >= 0]
    TrackedRuns.reset_index(inplace=True)
    return TrackedRuns, removed_horses


def generate_qualified_horse_summary(TrackedRuns):
    qualified_horse_summary = TrackedRuns.loc[TrackedRuns['RunNo.'] == 1].reset_index()
    # qualified_horse_summary = qualified_horse_summary[['Horse', 'Distance', 'Jockey', 'Dr', 'Result']]
    qualified_horse_summary.index = np.arange(1, len(qualified_horse_summary) + 1)
    return qualified_horse_summary



def conjunction(type, *conditions):
    if type:
        return functools.reduce(np.logical_or, conditions)
    else:
        return functools.reduce(np.logical_and, conditions)


def table_counts(df, odds, f):
    return df[odds].value_counts().get(f, 0)


def unique_non_null(s):
    return s.dropna().unique()

def safe_division(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0


def generate_complete_scoreboard(qualified_horse_summary, race_stats, qualifying_races, qualified_horses):
    total_races = len(race_stats)
    total_qualified_races = len(qualifying_races)
    total_qualified_horses = len(qualified_horses)
    qualified_pct = 100 * safe_division(total_qualified_races, total_races)
    ytr = len(qualified_horses)- len(qualified_horse_summary)
    nr1 = len(qualified_horse_summary)
    win = qualified_horse_summary['Result'].value_counts()[1]
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
                    html.Th("Yet to run/DQ", id='overall_TR_header'),
                    html.Th("NR1", id='overall_TR_header'),
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
                html.Td("{}".format(win),
                        className='first_pos'),
                html.Td("{}".format(edge),
                        className='place_pos'),
                html.Td("{}".format(n),
                        className='place_pos')
            ])
        ], id='overall_table')]
    return Overall_SB



def generate_overall_scoreboard_for_combo(TrackedRuns):
    first_run = TrackedRuns.loc[TrackedRuns['RunNo.']==1]
    FTR = len(first_run)
    FW = int(len(first_run.loc[first_run['Result'] == 1]))
    FS = int(len(first_run.loc[first_run['Result'] == 2]))
    FT = int(len(first_run.loc[first_run['Result'] == 3]))
    FP = int(len(first_run.loc[(first_run['Result'] == 1) | (first_run['Result'] == 2) | (first_run['Result'] == 3)]))
    FB = FTR - FP
    FWpct = round(100 * safe_division(FW, FTR), 2)
    FSpct = round(100 * safe_division(FS, FTR), 2)
    FTpct = round(100 * safe_division(FT, FTR), 2)
    FPpct = round(100 * safe_division(FP, FTR), 2)
    Overall_SB = [html.Div(children = [
        html.H6("Scoreboard",
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
                      ])
                  ], id='overall_sb_table')]
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
            ["Date", "Season", "Race No", "Venue ID"]].to_dict(
            'records'),
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


def generate_qual_races_details_table(xdf, date, season, race, venue):
    header1 = html.H3("Qualifying Race Detailed",
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    x = 'https://www.indiarace.com/Home/racingCenterEvent?venueId={}&event_date={}&race_type=RESULTS'.format(venue,
                                                                                                             date)
    header2 = html.A(children=[html.H5("Date: {}   Season: {}   Race: {}".format(date, season, race),
                                       style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00',
                                              'margin': '0px',
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
       'Jockey', 'Dr', 'Result', 'LBW', 'Penalty', 'Time']]],
        data=xdf.reset_index()[
            ['Sr #', 'Trainer', 'Class', 'Distance', 'Horse', 'Age', 'Wt',
       'Jockey', 'Dr', 'Result', 'LBW', 'Penalty', 'Time']].to_dict(
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
        hidden_columns=['Trainer', 'Jockey', 'Class', 'Distance', 'Wt', 'Penalty', 'Time'],
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
       'Penalty', 'Time']].reset_index().columns],
        data=xdf[
            ['Horse', 'Date', 'Season', 'Race No', 'Sr #', 'Trainer', 'Class',
       'Distance', 'Age', 'Wt', 'Jockey', 'Dr', 'Result', 'LBW',
       'Penalty', 'Time']].reset_index().to_dict(
            'records'),
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
        hidden_columns=['Trainer', 'Jockey', 'Class', 'Distance', 'Penalty', 'Time'],
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
                'column_id': 'Dr'
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
    first_run_link = html.Div()
    second_run_link = html.Div()
    if 1 in xdf['RunNo.'].unique():
        venue1 = xdf.iloc[1]['Venue ID']
        date1 = xdf.iloc[1]['Date']
        x1 = 'https://www.indiarace.com/Home/racingCenterEvent?venueId={}&event_date={}&race_type=RESULTS'.format(
            venue1,
            date1)
        print(x1)
        first_run_link = html.Span(html.A(children=[html.H6("Run 1 link",
                                                            style={'textAlign': 'center', 'font-weight': 'bold',
                                                                   'color': '#59ff00', 'margin': '0px',
                                                                   'font-family': 'Arial Black'})
                                                    ],
                                          href=x1, target='_blank'), className="pretty_container",
                                   style={'display': 'inline-block', 'background-color': '#061e44',
                                          'box-shadow': 'none', 'margin': '0px',
                                          'padding': '0px', 'width': '130px'})
    if 2 in xdf['RunNo.'].unique():
        venue2 = xdf.iloc[2]['Venue ID']
        date2 = xdf.iloc[2]['Date']
        x2 = 'https://www.indiarace.com/Home/racingCenterEvent?venueId={}&event_date={}&race_type=RESULTS'.format(
            venue2,
            date2)
        print(x2)
        second_run_link = html.Span(html.A(children=[html.H6("Run 2 Link",
                                                             style={'textAlign': 'center', 'font-weight': 'bold',
                                                                    'color': '#59ff00',
                                                                    'margin': '0px',
                                                                    'font-family': 'Arial Black'})
                                                     ],
                                           href=x2, target='_blank'), className="pretty_container",
                                    style={'display': 'inline-block', 'background-color': '#061e44',
                                           'box-shadow': 'none', 'margin': '0px',
                                           'padding': '0px', 'width': '130px'})
    fav_sb = dash_table.DataTable(
        id='table4',
        columns=[{"name": i, "id": i, "hideable":True} for i in xdf.reset_index()[
            ['RunNo.','Horse', 'Sr #', 'Date', 'Trainer', 'Season', 'Race No', 'Class',
       'Distance', 'Age', 'Wt', 'Jockey', 'Dr', 'Result', 'LBW',
       'Penalty', 'Time']]],
        data=xdf.reset_index()[
            ['RunNo.', 'Horse', 'Sr #', 'Date', 'Trainer', 'Season', 'Race No', 'Class',
       'Distance', 'Age', 'Wt', 'Jockey', 'Dr', 'Result', 'LBW',
       'Penalty', 'Time']].to_dict(
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
        hidden_columns=['Trainer', 'Jockey', 'Class', 'Distance', 'Penalty', 'Time'],
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
                'column_id': 'Dr'
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
    if len(xdf) == 0:
        first_run_link = html.Div("Removed because next run did not meet criteria mentioned above")

    return header, fav_sb, first_run_link, second_run_link


def generate_horse_run_result_summary_table(xdf):
    header = html.H3("Run Result Summary for Qualified Horses",
                     style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                            'font-family': 'Arial Black'})
    fav_sb = dash_table.DataTable(
        id='table5',
        columns=[{"name": i, "id": i, "hideable":True} for i in xdf[
            ['RunNo.','Horse', 'Sr #', 'Date', 'Trainer', 'Season', 'Race No', 'Class',
       'Distance', 'Age', 'Wt', 'Jockey', 'Dr', 'Result', 'LBW',
       'Penalty', 'Time', 'Sh', 'Centre']].reset_index().columns],
        data=xdf[
            ['RunNo.','Horse', 'Sr #', 'Date', 'Trainer', 'Season', 'Race No', 'Class',
       'Distance', 'Age', 'Wt', 'Jockey', 'Dr', 'Result', 'LBW',
       'Penalty', 'Time', 'Sh', 'Centre']].reset_index().to_dict(
            'records'),
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
            options=[{'label': 'Method 3 - Pythagorean Curse(Quantum)', 'value': 'Method 3 - Pythagorean Curse(Quantum)'}],
            value='Method 3 - Pythagorean Curse(Quantum)',
            clearable=False
        ),
        className='dash-dropdown'
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


rules = ["1. Applicable only to six-furlong races(1200m). \n", html.Br(),
         "2. Only horses with wide draw (10 or higher) to be looked at in the race video. \n",
         html.Br(),
         "3. They must exhibit some or all of thesymptoms mentioned in the theoretical discussion above, viz racing wide without cover or not being able to settle well, or wide at the turn, or throwing in the towel in the final furlong, etc. \n",
         html.Br(),
         "4. Confirm both from race video as well as the Dynamic Handicapping LBW (lengths behind winner) formula, that such horse has finished between 3.5 & 4 lengths from the winner. THIS IS VERY ESSENTIAL. Treat that range (3.5 – 4) as a sacrosanct quantum range. \n",
         html.Br(),
         "5. Next out, back the horse BLINDLY, without giving ANY CONSIDERATION, to class, trip distance, jockey, etc provided the gate number is <10 this time. If over 10, don’t bet, skip the race and erase the horse from your list. There is no point in following up with this kind of a horse. The reason is, after that laboured unlucky run, the horse must get to run a true race in next start itself. If it doesn’t get that chance, there is no point in persisting with it.\n",
         html.Br(),
         "6. If all rules are followed 100%, in letter as well as spirit, the strike rate of this method can be as high as 75+ percent for win, and 90% for place. \n"]

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
                html.Div(id='scoreboard2', className='pretty_container', style={'display': 'initial'}),
                html.Div(id='scoreboard25', className='pretty_container', style={'display': 'initial'},
                         children=[html.H6("Rules",
                style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#59ff00', 'margin': '0px',
                       'font-family': 'Arial Black'}),
                                   html.P(children=rules)])
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

row4= dbc.Row(
    [
        dbc.Col(
            html.Div(id='tableslot4', className='pretty_container'),
            width=12, className='pretty_container twelve columns', id='row4', style={'display': 'initial'}
        )
    ], className='flex-display'
)

app.layout = html.Div([
    row0,
    row1,
    row2,
    row3,
    row4,
    html.Div(id="store_original_df", children=[data.to_json(orient='split')], style={'display': 'none'}),
    html.Div(id="store_filtered_df", style={'display': 'none'}),
    html.Div(id="store_race_stats", style={'display': 'none'}),
    html.Div(id="store_qualifying_races", style={'display': 'none'}),
    html.Div(id="store_qualifying_races_detailed", style={'display': 'none'}),
    html.Div(id="store_qualifying_horses", style={'display': 'none'}),
    html.Div(id="store_Tracked_runs", style={'display': 'none'}),
    html.Div(id="store_run_result_summary", style={'display': 'none'})],
    id='mainContainer', style={'display': 'flex', 'flex-direction': 'column'})



@app.callback(
    [Output('methodtitle', 'children'),
     Output('scoreboard', 'children'),
     Output('scoreboard2', 'children'),
     Output('scoreboard2', 'style'),
     Output('tableslot1left', 'children'),
     Output('tableslot1right', 'children'),
     Output('row1', 'style'),
     Output('tableslot2', 'children'),
     Output('row2', 'style'),
     Output('tableslot3', 'children'),
     Output('row3', 'style'),
     Output('tableslot4', 'children'),
     Output('row4', 'style'),
     Output('store_filtered_df', 'children'),
     Output('store_race_stats', 'children'),
     Output("store_qualifying_races", 'children'),
     Output("store_qualifying_races_detailed", 'children'),
     Output("store_qualifying_horses", 'children'),
     Output("store_Tracked_runs", 'children'),
     Output("store_run_result_summary", 'children')
     ],
    [Input('Method', 'value')
     ],
    prevent_initial_call=True
)
def update_main(Method, *searchparam):
    if Method is None:
        print("Hello")
        raise PreventUpdate
    df = data
    race_stats = generate_race_stats(df)
    qualified_horses = generate_qualified_horses(df)
    qualified_horses['LBW'] = round(qualified_horses['LBW'],2)
    qualified_horses['Time'] = round(qualified_horses['Time'], 3)
    qualifying_races = generate_qualifying_races(qualified_horses)
    if len(qualified_horses) == 0:
        return [[],'No Qualified Horses', [], [], {'display': 'none'}, [], {'display': 'none'}, [], {'display': 'none'}, [], {'display': 'none'},
                [],[],[],[],[],[],[]]
    qualifying_races_detailed = generate_qualifying_races_detailed(qualifying_races, df)
    qualifying_races_detailed['LBW'] = round(qualifying_races_detailed['LBW'],2)
    qualifying_races_detailed['Time'] = round(qualifying_races_detailed['Time'], 3)
    TrackedRuns, removed_horses = generate_TrackedRuns(df, qualified_horses)
    TrackedRuns['LBW'] = round(TrackedRuns['LBW'],2)
    TrackedRuns['Time'] = round(TrackedRuns['Time'], 3)
    qualified_horse_summary = generate_qualified_horse_summary(TrackedRuns)
    if len(qualifying_races) == 0:
        score_return = "No Qualified Horses"
        score2_return = []
        score2_style = {'display': 'none'}
        table1left = []
        table1right = []
        row1style = {'display': 'none'}
        table2 = []
        row2_style = {'display': 'none'}
        table3 = []
        row3_style = {'display': 'none'}
        table4= []
        row4_style = {'display': 'none'}
    else:
        score_return = generate_complete_scoreboard(qualified_horse_summary, race_stats, qualifying_races, qualified_horses)
        score2_return = generate_overall_scoreboard_for_combo(TrackedRuns)
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
            TrackedRuns.loc[TrackedRuns['Horse']==qualified_horses.loc[1,'Horse']],
            qualified_horses.loc[1, 'Horse']
        )
        row3_style = {'display': 'initial'}
        table4 = generate_horse_run_result_summary_table(qualified_horse_summary)
        row4_style = {'display': 'initial'}

    return [html.H2("{} method".format(Method),
                style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#00ffdd', 'margin': '0px',
                       'font-family': 'Arial Black'}),
        score_return, score2_return, score2_style, table1left, table1right, row1style, table2, row2_style, table3, row3_style,
            table4, row4_style,
            [df.to_json(orient='split')], [race_stats.to_json(orient='split')],
            [qualifying_races.to_json(orient='split')],
            [qualifying_races_detailed.index.names, qualifying_races_detailed.reset_index().to_json(orient='split')],
            [qualified_horses.to_json(orient='split')],
            [TrackedRuns.to_json(orient='split')],
            [qualified_horse_summary.to_json(orient='split')]]


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
    print(active_cell)
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
    print(active_cell)
    qualifying_races = pd.read_json(qual_races[0], orient='split')
    qualifying_races_detailed = pd.read_json(qual_races_detailed[1], orient='split').set_index(qual_races_detailed[0])
    qualified_horses = pd.read_json(qual_horses[0], orient='split')
    TrackedRuns = pd.read_json(TRuns[0], orient='split')
    TrackedRuns['LBW'] = round(TrackedRuns['LBW'],2)
    TrackedRuns['Time'] = round(TrackedRuns['Time'], 3)
    TrackedRuns['Date'] = TrackedRuns['Date'].dt.date
    row = active_cell['row']
    column_id = active_cell['column_id']
    horse = data[row]['Horse']
    table3 = generate_tracked_runs_selected_horse_table(
        TrackedRuns.loc[TrackedRuns['Horse'] == horse],
        horse
    )
    return [table3]


if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)