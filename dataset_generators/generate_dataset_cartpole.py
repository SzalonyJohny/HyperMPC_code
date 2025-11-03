from dataset_generators.generate_episode_dynmodel import CartPoleSimTorch
import pathlib
import json
import wandb

if __name__ == "__main__":

    
    config = {
    "Ts": 0.01,
    "episode_len_s": 5,
    "control_points_count": 20,
    "u_max": 20.0,
    "u_max_Bspline_factor": 2.0,
    "v1_init_range": 10.0 / 5.0,
    "dtheta_init_range": 5.0 / 2.0,
    "batch_size": 16,
    "integration_method": "rk4",
    "compile": True,
    "chunk_mode": False,
    "chunk_size": 10, 
    "number_of_runs": 180,
    }

    run = wandb.init(
        project="hpm_cartpole_dataset",
        config=config,
    )

    sim = CartPoleSimTorch(config)

    outpath = pathlib.Path("./data/cartpole_dataset")

    if outpath.exists():
        import shutil
        shutil.rmtree(outpath)

    outpath.mkdir(parents=True)

    with open(outpath / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    for i in range(config["number_of_runs"]):
        df = sim.generate_episode()
        df.to_csv(outpath / f"episode_{i}.csv", index=False)
        print(f"Episode {i} saved")

    artifact = wandb.Artifact("cartpole_dataset", type="dataset")
    artifact.add_dir(outpath)
    artifact.save()

    run.link_artifact(
        artifact=artifact,
        target_path="f1tenth-org/wandb-registry-dataset/cartpole_datasets",
    )
    
    print("Done")
