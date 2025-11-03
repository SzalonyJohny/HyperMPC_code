from dataset_generators.acrobot_sim_mujoco import *
from dataset_generators.acrobot_sim_torch import *
from dataset_generators.acrobot_backlash_sim_mujoco import *
from dataset_generators.pendulum_backlash_sim_mujoco import *
import pathlib
import json
import wandb

if __name__ == "__main__":

    psim = {
    "sim_time_step": 10e-5,
    "slider_range":  0.2,
    "m": 1.0,
    "l": 0.5,
    "r": 0.025*5, # Not used
    "f": 0.0,
    "b": 0.05,
    "backlash": 15,
    }

    dgen_settings = {
        "sim_implementation": "mujoco",
        # debug
        "render": False,
        "render_width": 480,
        "render_height": 480,
        # episode parametes
        "sample_per_second": 100,
        "episode_len_s": 10,
        # dataset size and path
        "number_of_runs": 360,  # 360 * 10s = 1 hour
        # to bias dataset with more data q1 = pi
        "q_range": (- np.pi, np.pi),
        "dq_range": (10.0, 10.0),
        # control signal
        "u_max": 1.0,
        "u_control_pt": 15,
    }

    merge_config = {"psim": psim, "dgen_settings": dgen_settings}

    run = wandb.init(
        project="hpm_pendulum_dataset",
        config=merge_config,
    )

    sim = PendulumSimMujocoBacklash(psim, dgen_settings)

    outpath = pathlib.Path("./data/pendulum_dataset")

    if outpath.exists():
        import shutil
        shutil.rmtree(outpath)

    outpath.mkdir(parents=True)

    with open(outpath / "config.json", "w") as f:
        json.dump(dgen_settings, f, indent=4)

    with open(outpath / "config_sim.json", "w") as f:
        json.dump(psim, f, indent=4)

    for i in range(dgen_settings["number_of_runs"]):
        df = sim.generate_episode()
        df.to_csv(outpath / f"episode_{i}.csv", index=False)
        print(f"Episode {i} saved")

    artifact = wandb.Artifact("pendulum_dataset", type="dataset")
    artifact.add_dir(outpath)
    artifact.save()

    run.link_artifact(
        artifact=artifact,
        target_path="f1tenth-org/wandb-registry-dataset/pendulum_datasets",
    )

    print("Done")
