
import torchvision
from torchvision import transforms
import torch
from torch import nn



class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

class EmotionDetection(nn.Module):
      
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1, 
                      padding=1),
            nn.BatchNorm2d(hidden_units),  
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) 
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=128,
                      out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64,
                      out_features=output_shape),
            nn.Dropout(p=0.2)
        )
    
    def forward(self, x: torch.Tensor):
      # return self.classifier(self.conv_block_1(x)) 
      return self.classifier(self.conv_block_2(self.conv_block_1(x))) 

torch.manual_seed(42)
model = EmotionDetection(input_shape=3,
                  hidden_units=64, 
                  output_shape=7)

model.load_state_dict(torch.load("model_weights.pth",
                                 map_location=torch.device('cpu')))
# model = torch.load('model.pth')
def predict(img):
    
    image = torch.unsqueeze(transform(img), 0)

    print(f"Custom image shape: {image.shape}\n")
    print(f"Custom image dtype: {image.dtype}")
    
    model.eval()
    with torch.inference_mode():
        pred = model(image)
    
    return class_names[torch.argmax(torch.softmax(pred, dim=1), dim=1)] , torch.softmax(pred, dim=1).max()
    