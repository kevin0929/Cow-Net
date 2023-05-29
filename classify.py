import torch
import torch.nn as nn


class CowNet(nn.Module):
    def __init__(self):
        super(CowNet, self).__init__()

        # MLP for coordinates and bounding box dimensions
        self.mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # CNN for image data (using ResNet)
        self.cnn = torch.hub.load("pytorch/vision:v0.9.0", "resnet18", pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 512)   # according to the output layer of CNN model

        # FC layer after concat
        self.output_fc = nn.Sequential(
            nn.Linear(576, 256),    # 512 + 64 = 576
            nn.ReLU(),
            nn.Linear(256, 5),  # 5 classed
            nn.Softmax(dim=1)
        )
    
    
    def forward(self, cow_info: torch.Tensor, cow_img: bytes):
        # MLP process and CNN process
        mlp_output = self.mlp(cow_info)
        cnn_output = self.cnn(cow_img)
        
        # concat output
        concat_output = torch.cat((mlp_output, cnn_output), dim=1)
        
        # classify output class
        behavior_class = self.output_fc(concat_output)
        
        return behavior_class
