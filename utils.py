# all the utilities function to process the data

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Button
import numpy as np
from bokeh.plotting import figure, output_notebook
from bokeh.palettes import Dark2
from bokeh.layouts import layout, column
from bokeh.models import RangeSlider, Button, CustomJS, Spacer, HoverTool, ColumnDataSource, Slider, Legend, Range1d
output_notebook()

os.environ['JUPYTER_BOKEH_EXTERNAL_URL'] = "https://labs.shmmh.co"
os.environ['JUPYTERHUB_SERVICE_PREFIX'] = "https://labs.shmmh.co"




def parse_data(file_path):
    df = pd.read_csv(file_path, header=None)

    # Initialize variables
    channels = {}

    # Iterate through the dataframe by 6 columns
    for i in range(0, len(df.columns), 6):
        # Extract relevant columns for the current channel
        channel_data = df.iloc[:, i:i+6].dropna(how='all', axis=0)
        if not channel_data.empty:
            # Extract time and voltage values
            time_values = channel_data.iloc[:, 3].values
            voltage_values = channel_data.iloc[:, 4].values
            
            record_length = {'value': channel_data.iloc[0,1], 'unit': channel_data.iloc[0,2]}
            sample_interval = {'value': channel_data.iloc[1,1], 'unit': channel_data.iloc[1,2]}
            trigger_point = {'value': channel_data.iloc[2,1], 'unit': channel_data.iloc[2,2]}
            
            # Extract other trace information
            trace_info = {key: value for key, value in zip(channel_data.iloc[6:, 0], channel_data.iloc[6:, 1])}

            
            # Remove the empty 'NaN' key-value pair
            del trace_info[np.nan]
            
            # Split the Note info into data,time, oscilliscope model
            note_info = trace_info['Note'].split()
            trace_info['Date'] = note_info[3]
            trace_info['Time'] = note_info[2]
            trace_info['Model'] = note_info[0]
            
            trace_info['Record Length'] = record_length
            trace_info['Sample Interval'] = sample_interval
            trace_info['Trigger Point'] = trigger_point
            del trace_info['Note']
            
            channel_name  = trace_info['Source']
            # Store data in the channels dictionary
            channels[channel_name] = {'x': time_values, 'y': voltage_values, 'Trace Info': trace_info}
    return channels

def plot_data(x, y, peak):

    show_peaks, peaks = peak
    
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.subplots_adjust(bottom=0.1, left=0.12)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    x_ax = plt.axes([0.15, 0.02, 0.7, 0.04])
    x_limits = RangeSlider(x_ax, label='x lim:',valmin=xmin, valmax=xmax, valinit=(xmin,xmax), valstep=abs(xmax-xmin)/100, orientation='horizontal')

    y_ax = plt.axes([0.05, 0.2, 0.02, 0.7])
    y_limits = RangeSlider(y_ax,label='y lim',valmin=ymin, valmax=ymax, valinit=(ymin,ymax), valstep=abs(ymax-ymin)/100, orientation='vertical')

    reset_ax = plt.axes([0.8, 0.9, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset')

    if show_peaks:
        ax.plot(x[peaks], y[peaks], 'xr')
        for i, peak in enumerate(peaks):
            ax.annotate( f'{i}', xy=(x[peak], y[peak]), xytext=(0, 5), textcoords='offset points', ha='center')
        

    line = ax.plot(x,y)

    def reset(val):
        x_limits.reset()
        y_limits.reset()
        fig.canvas.draw_idle()

    def updatex(val):
        x_min, x_max = x_limits.val
        ax.set_xlim([x_min, x_max])
        fig.canvas.draw_idle()
        
    def updatey(val):
        y_min, y_max = y_limits.val
        ax.set_ylim([y_min, y_max])
        fig.canvas.draw_idle()
        
        


    x_limits.on_changed(updatex)
    y_limits.on_changed(updatey)
    reset_button.on_clicked(reset)

    plt.show()
    return x,y, peaks

def plot_data_bokeh(data):
    def bkapp(doc):
        tools = 'pan, box_zoom,zoom_in, zoom_out, wheel_zoom ,undo, redo, reset, save'
        p = figure( title="Oscilloscope Data", x_axis_label='Time/s', y_axis_label='Volts/mV', sizing_mode='scale_width', max_height=800, resizable=True, toolbar_location='left', tools = tools)
        xmins, ymins = [], []
        for i,channel in enumerate(data):
            ch = data[channel]
        
            x, y = ch['x'], ch['y']
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            xmins.extend([xmin,xmax])
            ymins.extend([ymin,ymax])
            
            p.line(x, y, line_width=2, legend_label=f'{channel}', color=Dark2[8][i])
        
        xmin, xmax = min(xmins), max(xmins)
        ymin, ymax = min(ymins), max(ymins)
        p.x_range = Range1d(xmin,xmax)
        p.y_range = Range1d(ymin,ymax)
        
        hover = HoverTool(tooltips=[('x', '@x'), ('y', '@y'),('index', '$index')])
        p.add_tools(hover)
        
        x_range_slider = RangeSlider( 
            title="X", 
            start=xmin, 
            end=xmax, 
            step= (xmax-xmin)/ 1000, 
            value=(p.x_range.start, p.x_range.end),
            sizing_mode='scale_width',
            max_width=500
        ) 
        y_range_slider = RangeSlider( 
            title="Y", 
            start=ymin, 
            end=ymax, 
            step= (ymax-ymin)/ 10000, 
            value=(p.y_range.start, p.y_range.end),
            sizing_mode='scale_width',
            max_width=500
        ) 
        
        x_range_slider.js_link("value", p.x_range, "start", attr_selector=0) 
        x_range_slider.js_link("value", p.x_range, "end", attr_selector=1) 
        y_range_slider.js_link("value", p.y_range, "start", attr_selector=0) 
        y_range_slider.js_link("value", p.y_range, "end", attr_selector=1)
        
        reset_slider_callback = CustomJS(args=dict(x_range_slider=x_range_slider, y_range_slider=y_range_slider), code="""
            x_range_slider.value = [x_range_slider.start, x_range_slider.end];
            y_range_slider.value = [y_range_slider.start, y_range_slider.end];
        """)
        reset_button = Button(label="Reset")
        
        reset_button.on_click(reset_slider_callback)
        
        layout = column(reset_button, x_range_slider,Spacer(height=30),y_range_slider,Spacer(height=30),p,sizing_mode='scale_width', max_height=900)
        
        p.legend.click_policy="hide"
    return bkapp


from scipy import constants as cnst


def find_calibration(channel, path_diff,peaks, peak_ind=(0,1)):
    x,y = channel['x'], channel['y']
    ind1, ind2 = peak_ind
    
    time_diff = np.abs((x[peaks[ind2]] - x[peaks[ind1]])) * 1000 # in ms
    fringes = np.abs(ind2 - ind1)

    df = ((cnst.c)/ (2* path_diff)) / 10**6 # in MHz. This is the frequncy per fringe.
    
    hor_div = ((fringes / time_diff) * (df)) # in MHz/ms
    
    print('P214 - P95 =', time_diff, 'ms')
    print('hor_div =', hor_div, ' Mhz/ms')
    print('Freq per 200 us ( 1 unit ) =', hor_div * 200 / 1000, ' MHz' )
    print('df =',df, 'MHz' )
    print('firnges = ', fringes)

    return hor_div, time_diff # MHz/ms and ms