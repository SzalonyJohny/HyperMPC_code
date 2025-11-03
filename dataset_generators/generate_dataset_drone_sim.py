from dataset_generators.drone_sim import DroneSim
import pathlib
import json
import wandb
import numpy as np
import pandas as pd
import torch
import shutil


if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)
    
    config = {
        'freq': 100,
        'episode_len_s': 20,
        'u_control_pt': 25,
        'max_deflaction': 0.5,
        'mujoco_sub_steps': 10,
        'ball_mass': 0.5,
        'rope_lenght' : 1.0,
        'render': False,
        'episode_count': 300
    }
    
    run = wandb.init(
        project="hpm_drone_dataset",
        config=config,
    )

    sim = DroneSim(config)

    outpath = pathlib.Path("./data/drone_dataset")

    if outpath.exists():
        import shutil
        shutil.rmtree(outpath)

    outpath.mkdir(parents=True)

    with open(outpath / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    for i in range(config["episode_count"]):
        df = sim.generate_episode()
        df.to_csv(outpath / f"episode_{i}.csv", index=False)
        
        print(f"Episode {i} saved")

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
        shutil.rmtree(outpath)

    outpath.mkdir(parents=True)
    train_df.to_csv(outpath / "train.csv", index=False)
    val_df.to_csv(outpath / "val.csv", index=False)
    test_df.to_csv(outpath / "test.csv", index=False)

    shutil.copy(config_file, outpath / "config.json")

    artifact = wandb.Artifact("drone_dataset", type="dataset")
    artifact.add_dir(outpath)
    artifact.save()

    l = round(config["rope_lenght"], 2)
    m = round(config["ball_mass"], 2)

    run.link_artifact(
        artifact=artifact,
        target_path="f1tenth-org/wandb-registry-dataset/drone_datasets_split",
        aliases=[f"{l}m{m}kg_v2"] # 0m0.5kg
    )

    print("Done")

        