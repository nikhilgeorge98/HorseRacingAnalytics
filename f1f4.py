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

df = pd.concat(pd.read_excel(r'C:\Users\Testing\F-1234-FINAL200307.xlsx', sheet_name=None), ignore_index=True)

df.rename(columns = {'F#':'F'}, inplace = True)
df['fav_num']=df.apply(lambda row:int(row.F.strip('F')), axis=1)
df.RESULT.replace(['NR','Rain','q'],[np.nan,np.nan,0], inplace=True)
df.dropna(subset=['RESULT'], inplace=True)
df['RESULT']=df['RESULT'].astype('int64')
df['Truth']=df['fav_num']==df['RESULT']
df['f_r'] = list(zip(df.fav_num, df.RESULT))

combo=[(1, 0),(1, 1),(1, 2),(1, 3),(1, 4),(2, 0),(2, 1),(2, 2),(2, 3),(2, 4),(3, 0),(3, 1),(3, 2),(3, 3),(3, 4),(4, 0),(4, 1),(4, 2),(4, 3),(4, 4)]
race_stats = df.groupby(["Venue","Season Code","Date", "R No.", "f_r"], as_index=True).size().unstack(fill_value=0)
mis_col = [i for i in combo if i not in race_stats.columns]
for col in mis_col:
    race_stats[col] = [0] * len(race_stats)
race_stats = race_stats[combo]

race_stats['F12'] = race_stats[(1,1)]*race_stats[(2,2)]
race_stats['F13'] = race_stats[(1,1)]*race_stats[(3,2)]
race_stats['F14'] = race_stats[(1,1)]*race_stats[(4,2)]

race_stats['F21'] = race_stats[(2,1)]*race_stats[(1,2)]
race_stats['F23'] = race_stats[(2,1)]*race_stats[(3,2)]
race_stats['F24'] = race_stats[(1,1)]*race_stats[(4,2)]

race_stats['F31'] = race_stats[(3,1)]*race_stats[(1,2)]
race_stats['F32'] = race_stats[(3,1)]*race_stats[(2,2)]
race_stats['F34'] = race_stats[(3,1)]*race_stats[(4,2)]

race_stats['F41'] = race_stats[(4,1)]*race_stats[(1,2)]
race_stats['F42'] = race_stats[(4,1)]*race_stats[(2,2)]
race_stats['F43'] = race_stats[(4,1)]*race_stats[(3,2)]

race_stats['F123'] = race_stats[(1,1)]*race_stats[(2,2)]*race_stats[(3,3)]
race_stats['F124'] = race_stats[(1,1)]*race_stats[(2,2)]*race_stats[(4,3)]
race_stats['F132'] = race_stats[(1,1)]*race_stats[(3,2)]*race_stats[(2,3)]
race_stats['F134'] = race_stats[(1,1)]*race_stats[(3,2)]*race_stats[(4,3)]
race_stats['F142'] = race_stats[(1,1)]*race_stats[(4,2)]*race_stats[(2,3)]
race_stats['F143'] = race_stats[(1,1)]*race_stats[(4,2)]*race_stats[(3,3)]

race_stats['F213'] = race_stats[(2,1)]*race_stats[(1,2)]*race_stats[(3,3)]
race_stats['F214'] = race_stats[(2,1)]*race_stats[(1,2)]*race_stats[(4,3)]
race_stats['F231'] = race_stats[(2,1)]*race_stats[(3,2)]*race_stats[(1,3)]
race_stats['F234'] = race_stats[(2,1)]*race_stats[(3,2)]*race_stats[(4,3)]
race_stats['F241'] = race_stats[(2,1)]*race_stats[(4,2)]*race_stats[(1,3)]
race_stats['F243'] = race_stats[(2,1)]*race_stats[(4,2)]*race_stats[(3,3)]

race_stats['F312'] = race_stats[(3,1)]*race_stats[(1,2)]*race_stats[(2,3)]
race_stats['F314'] = race_stats[(3,1)]*race_stats[(1,2)]*race_stats[(4,3)]
race_stats['F321'] = race_stats[(3,1)]*race_stats[(2,2)]*race_stats[(1,3)]
race_stats['F324'] = race_stats[(3,1)]*race_stats[(2,2)]*race_stats[(4,3)]
race_stats['F341'] = race_stats[(3,1)]*race_stats[(4,2)]*race_stats[(1,3)]
race_stats['F342'] = race_stats[(3,1)]*race_stats[(4,2)]*race_stats[(2,3)]

race_stats['F412'] = race_stats[(4,1)]*race_stats[(1,2)]*race_stats[(2,3)]
race_stats['F413'] = race_stats[(4,1)]*race_stats[(1,2)]*race_stats[(3,3)]
race_stats['F421'] = race_stats[(4,1)]*race_stats[(2,2)]*race_stats[(1,3)]
race_stats['F423'] = race_stats[(4,1)]*race_stats[(2,2)]*race_stats[(3,3)]
race_stats['F431'] = race_stats[(4,1)]*race_stats[(3,2)]*race_stats[(1,3)]
race_stats['F432'] = race_stats[(4,1)]*race_stats[(3,2)]*race_stats[(2,3)]

race_stats['F1234'] = race_stats[(1,1)]*race_stats[(2,2)]*race_stats[(3,3)]*race_stats[(4,4)]
race_stats['F1243'] = race_stats[(1,1)]*race_stats[(2,2)]*race_stats[(4,3)]*race_stats[(3,4)]
race_stats['F1324'] = race_stats[(1,1)]*race_stats[(3,2)]*race_stats[(2,3)]*race_stats[(4,4)]
race_stats['F1342'] = race_stats[(1,1)]*race_stats[(3,2)]*race_stats[(4,3)]*race_stats[(2,4)]
race_stats['F1423'] = race_stats[(1,1)]*race_stats[(4,2)]*race_stats[(2,3)]*race_stats[(3,4)]
race_stats['F1432'] = race_stats[(1,1)]*race_stats[(4,2)]*race_stats[(3,3)]*race_stats[(2,4)]

race_stats['F2134'] = race_stats[(2,1)]*race_stats[(1,2)]*race_stats[(3,3)]*race_stats[(4,4)]
race_stats['F2143'] = race_stats[(2,1)]*race_stats[(1,2)]*race_stats[(4,3)]*race_stats[(3,4)]
race_stats['F2314'] = race_stats[(2,1)]*race_stats[(3,2)]*race_stats[(1,3)]*race_stats[(4,4)]
race_stats['F2341'] = race_stats[(2,1)]*race_stats[(3,2)]*race_stats[(4,3)]*race_stats[(1,4)]
race_stats['F2413'] = race_stats[(2,1)]*race_stats[(4,2)]*race_stats[(1,3)]*race_stats[(3,4)]
race_stats['F2431'] = race_stats[(2,1)]*race_stats[(4,2)]*race_stats[(3,3)]*race_stats[(1,4)]

race_stats['F3124'] = race_stats[(3,1)]*race_stats[(1,2)]*race_stats[(2,3)]*race_stats[(4,4)]
race_stats['F3142'] = race_stats[(3,1)]*race_stats[(1,2)]*race_stats[(4,3)]*race_stats[(2,4)]
race_stats['F3214'] = race_stats[(3,1)]*race_stats[(2,2)]*race_stats[(1,3)]*race_stats[(4,4)]
race_stats['F3241'] = race_stats[(3,1)]*race_stats[(2,2)]*race_stats[(4,3)]*race_stats[(1,4)]
race_stats['F3412'] = race_stats[(3,1)]*race_stats[(4,2)]*race_stats[(1,3)]*race_stats[(2,4)]
race_stats['F3421'] = race_stats[(3,1)]*race_stats[(4,2)]*race_stats[(2,3)]*race_stats[(1,4)]

race_stats['F4123'] = race_stats[(4,1)]*race_stats[(1,2)]*race_stats[(2,3)]*race_stats[(3,4)]
race_stats['F4132'] = race_stats[(4,1)]*race_stats[(1,2)]*race_stats[(3,3)]*race_stats[(2,4)]
race_stats['F4213'] = race_stats[(4,1)]*race_stats[(2,2)]*race_stats[(1,3)]*race_stats[(3,4)]
race_stats['F4231'] = race_stats[(4,1)]*race_stats[(2,2)]*race_stats[(3,3)]*race_stats[(1,4)]
race_stats['F4312'] = race_stats[(4,1)]*race_stats[(3,2)]*race_stats[(1,3)]*race_stats[(2,4)]
race_stats['F4321'] = race_stats[(4,1)]*race_stats[(3,2)]*race_stats[(2,3)]*race_stats[(1,4)]

race_stats.reset_index(inplace=True)
race_stats.sort_values(['Date','Season Code','R No.'], inplace=True)

'''**************************************************************************'''

app =dash.Dash(__name__)


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



def table_counts(dff, odds, f):
    return dff[odds].value_counts().get(f, 0)


def unique_non_null(s):
    return s.dropna().unique()


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
                    {'label': 'Final Odds', 'value': 'FO'}
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


row0 = dbc.Row(
    [
        dbc.Col(
            [
                html.Div(id='Overall_scoreboard', className='pretty_container', style={'display': 'none'})
            ],
            width=12, className='pretty_container twelve columns', id='base_right-column'
        )
    ], id='fixedontop', className='flex-display'
)



row1 = dbc.Row(
    [
        dbc.Col(
                    children=[html.H2("F1-F4 Analysis", id='title'),
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


app.layout = html.Div([
                       row0,
                       row1,
                       html.Div(id="store_original_df", style={'display': 'none'}),
                       html.Div(id="store_racewise_df", style={'display': 'none'})],
                      id='mainContainer', style={'display': 'flex', 'flex-direction': 'column'})