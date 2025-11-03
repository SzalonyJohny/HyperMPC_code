import torch
import pandas as pd
import numpy as np
import pathlib
import wandb
torch.manual_seed(42)
np.random.seed(42)

dataset_path = pathlib.Path("./data/drone_dataset")

file_list = list(dataset_path.glob("*.csv"))
config_file = dataset_path / "config.json"

print(file_list)
N = len(file_list)

train_ratio = 0.7
val_ratio = 0.2

# sample train episodes
train_files = np.random.choice(file_list, int(N * train_ratio), replace=False)
file_list = [f for f in file_list if f not in train_files]

# sample val episodes
val_files = np.random.choice(file_list, int(N * val_ratio), replace=False)
file_list = [f for f in file_list if f not in val_files]

# the rest is test
test_files = file_list

print(f"Train files: {len(train_files)}")
print(f"Val files: {len(val_files)}")
print(f"Test files: {len(test_files)}")

# save in one df
def one_df_from_files(files):
    dfs = []
    last_t = 0.0
    for i, file in enumerate(files):
        df = pd.read_csv(file)
        df["run_id"] = i    
        df["t"] += last_t
        last_t = df["t"].iloc[-1] + 0.01
        dfs.append(df)
    return pd.concat(dfs)


train_df = one_df_from_files(train_files)
val_df = one_df_from_files(val_files)
test_df = one_df_from_files(test_files)

print(train_df.head())

# mkdir
outpath = pathlib.Path("./data/drone_dataset_split")
if outpath.exists():
    import shutil
    shutil.rmtree(outpath)

outpath.mkdir(parents=True)
train_df.to_csv(outpath / "train.csv", index=False)
val_df.to_csv(outpath / "val.csv", index=False)
test_df.to_csv(outpath / "test.csv", index=False)

import shutil
shutil.copy(config_file, outpath / "config.json")

run = wandb.init(
    project="hpm_drone_dataset",
    config={}, # todo load from json
)


artifact = wandb.Artifact("drone_dataset", type="dataset")
artifact.add_dir(outpath)
artifact.save()

run.link_artifact(
    artifact=artifact,
    target_path="f1tenth-org/wandb-registry-dataset/drone_datasets_split",
)


print("Done")
