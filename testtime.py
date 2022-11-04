import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import numpy as np
import pandas as pd

data = np.column_stack((np.arange(10), np.arange(10) * 2))
df = pd.DataFrame(columns=["a column", "another column"], data=data)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Button("Save", id='down_Button'),
    html.A(id='my-link'),

])


@app.callback(Output('my-link', 'href'), [Input('down_Button', 'n_clicks')])
def update_link(value):
    df.to_csv("testereee2.csv")
    return '/dash/urlToDownload?value={}'.format(value)


# @app.server.route('/dash/urlToDownload')
# def download_csv():
#     return flask.send_file(df.to_csv(),
#                            mimetype='text/csv',
#                            attachment_filename='downloadFile.csv',
#                            as_attachment=True)


if __name__ == '__main__':
    app.run_server(debug=True)