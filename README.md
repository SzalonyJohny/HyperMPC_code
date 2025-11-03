# Predict_Prediction_Model

This repository contains the code for the paper:

### [Beyond Constant Parameters: Hyper Prediction Models and HyperMPC](https://arxiv.org/abs/2508.06181)
*Jan Węgrzynowski, Piotr Kicki, Grzegorz Czechmanowski, Maciej Krupka, Krzysztof Walas*

**Abstract:** *Model Predictive Control (MPC) is among the most widely adopted and reliable methods for robot control, relying critically on an accurate dynamics model. However, existing dynamics models used in the gradient-based MPC are limited by computational complexity and state representation. To address this limitation, we propose the Hyper Prediction Model (HyperPM)- a novel approach in which we project the unmodeled dynamics onto a time-dependent dynamics model. This time-dependency is captured through time-varying model parameters, whose evolution over the MPC prediction horizon is learned using a neural network. Such formulation preserves the computational efficiency and robustness of the base model while equipping it with the capacity to anticipate previously unmodeled phenomena. We evaluated the proposed approach on several challenging systems, including real-world F1TENTH autonomous racing, and demonstrated that it significantly reduces long-horizon prediction errors. Moreover, when integrated within the MPC framework (HyperMPC), our method consistently outperforms existing state-of-the-art techniques.*

### Citation

If you find our work useful, please consider citing:
```bibtex
@misc{wegrzynowski2025hypermpc,
    title={Beyond Constant Parameters: Hyper Prediction Models and HyperMPC},
    author={Jan Węgrzynowski and Piotr Kicki and Grzegorz Czechmanowski and Maciej Krupka and Krzysztof Walas},
    year={2025},
    eprint={2508.06181},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

# Usage
Firstly, build the docker image using the provided build script:
```bash
./docker/build.sh
```
Than run the docker container:
```bash
./docker/run.sh
```
For our experiments, we used Weights & Biases for experiment tracking.
Please set up your W&B account and login inside the docker container:
```bash
wandb login
```

## Datasets for simulation experiments
- Datasets are stored are downloaded to the `saved_artifacts/` folder, the path must be set  manualy in the configuration files if the wandb is not used.
- Scripts for dataset creation are available in the `dataset_generators/` folder.

## Training and Evaluation
-   **Training:** The training script is `main.py`, with configuration files located in the `conf/` directory.
-   **Evaluation:** To run tests, use the `test_model.py` script. The configuration for testing is in `conf/config_test.yaml`.
-   **MPC Experiments:** MPC experiments can be executed with `run_mpc_car.py`, using configuration files from the `conf_mpc/` directory.

TODO: fix residual model for drone experiments, the path for residual nn is in the `saved_artifacts/` folder.