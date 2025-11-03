import torch
import robot_model.drone.drone_model

def test_dynamics():
    model = robot_model.drone.drone_model.drone_dynamics()
    