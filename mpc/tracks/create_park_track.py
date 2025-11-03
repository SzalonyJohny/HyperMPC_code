import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpc.tracks.track_preprocesor import TrackReader

# Configuration dictionary for track generation
CONFIG = {
    "track_name": "drift_park_track",
    "x_start": 0.0,
    "y_start": 0.0,
    "track_width_init": 2.0,
    "track_width_end": 0.5,
    "ds": 0.05,
    "track_length": 10.0,
    "tunnel_length": 2.0,
}


def generate_track_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates track data based on the provided configuration.

    Args:
        config: A dictionary containing track parameters.

    Returns:
        A tuple containing two pandas DataFrames:
        - The preprocessed track data.
        - The raw track data for simulation.
    """
    s = np.arange(0, config["track_length"], config["ds"])
    x = config["x_start"] + s
    y = config["y_start"] + np.zeros_like(s)

    # Calculate track width with a smooth transition
    transition_point = config["track_length"] - config["tunnel_length"]
    width_range = config["track_width_init"] - config["track_width_end"]
    track_width = (1 - np.tanh(5 * (s - transition_point))) * width_range + config["track_width_end"]

    curvature = np.zeros_like(s)
    heading = np.arctan2(np.gradient(y, config["ds"]), np.gradient(x, config["ds"]))
    heading = np.unwrap(heading)

    # Create preprocessed track DataFrame
    prep_track_data = pd.DataFrame({
        's': s,
        'x': x,
        'y': y,
        'heading': heading,
        'curvature': curvature,
        'track_width': track_width
    })

    # Create raw track DataFrame
    raw_track_data = pd.DataFrame({
        'x_m': x,
        'y_m': y,
        'w_tr_right_m': track_width / 2,
        'w_tr_left_m': track_width / 2
    })

    return prep_track_data, raw_track_data


def save_track_data(prep_df: pd.DataFrame, raw_df: pd.DataFrame, track_name: str):
    """
    Saves the generated track data to CSV files.

    Args:
        prep_df: DataFrame with preprocessed track data.
        raw_df: DataFrame with raw track data.
        track_name: The base name for the track files.
    """
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    raw_track_file = output_dir / f"{track_name}.csv"
    raw_df.to_csv(raw_track_file, index=False, sep=',')
    print(f"Saved raw track data to: {raw_track_file}")

    prep_track_file = output_dir / f"prep_{track_name}.csv"
    prep_df.to_csv(prep_track_file, index=False, sep=',')
    print(f"Saved preprocessed track data to: {prep_track_file}")


def plot_track(df: pd.DataFrame):
    """
    Plots the track centerline and boundaries.

    Args:
        df: DataFrame containing the preprocessed track data.
    """
    plt.figure(figsize=(15, 5))
    plt.plot(df['x'], df['y'], label='Track Path')
    plt.fill_between(
        df['x'],
        df['y'] - df['track_width'] / 2,
        df['y'] + df['track_width'] / 2,
        color='lightblue',
        alpha=0.5,
        label='Track Width'
    )
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Track Visualization')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    """
    Main function to generate, save, and visualize the track.
    """
    prep_track, raw_track = generate_track_data(CONFIG)
    save_track_data(prep_track, raw_track, CONFIG["track_name"])
    plot_track(prep_track)

    # Visualize the created track file using the TrackReader
    track_path = Path(__file__).parent / (CONFIG["track_name"] + ".csv")
    if track_path.exists():
        track = TrackReader(track_path, reverse=True, flip=False, corrected=False, loopback=False)
        track.plot_track()
        plt.show()
    else:
        print(f"Track file not found for TrackReader visualization: {track_path}")


if __name__ == "__main__":
    main()
