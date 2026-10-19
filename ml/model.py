import torch
import torch.nn as nn


class StressModel(nn.Module):

    def __init__(
        self,
        input_size,
        num_classes=2
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                128
            ),

            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),


            nn.Linear(
                128,
                64
            ),

            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),


            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),
            nn.Dropout(0.15),


            nn.Linear(
                32,
                16
            ),

            nn.ReLU(),


            nn.Linear(
                16,
                num_classes
            )
        )


    def forward(self, x):

        return self.network(x)