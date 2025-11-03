import numpy as np
import pandas as pd
from pathlib import Path
import os
from scipy.signal import savgol_filter


def getTrackCustom(filename, return_width=False):
    
    track_file = os.path.join(str(Path(__file__).parent), filename)
    map_df = pd.read_csv(track_file, sep=',', index_col=None)

    print(map_df.head())
    s = map_df['s'].to_numpy()
    x = map_df['x'].to_numpy() 
    y = map_df['y'].to_numpy()
    heading = map_df['heading'].to_numpy()
    heading[heading > np.pi] -= 2 * np.pi
    heading[heading < -np.pi] += 2 * np.pi
    curvature = map_df['curvature'].to_numpy()
    
    if return_width:
        track_width = map_df['track_width'].to_numpy()
        return s, x, y, heading, curvature, track_width
    
    return s, x, y, heading, curvature
