import torch.nn as nn

class lalaDQN(nn.Module): # p: 58224
    def __init__(self, in_channels=1, action_num=4):
        super(lalaDQN, self).__init__()

        self.action_num = action_num

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1) # p: 800
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1) # p: 18432
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # p: 36864
        self.fc1_adv = nn.Linear(64, 16) # p: 1024
        self.fc1_val = nn.Linear(64, 16) # p: 1024
        self.fc2_adv = nn.Linear(16, action_num) # p: 64
        self.fc2_val = nn.Linear(16, 1) # p: 16
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_state):
        output = self.relu(self.conv1(input_state))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv3(output))
        output = output.view(output.size(0), -1)
        adv = self.relu(self.fc1_adv(output))
        val = self.relu(self.fc1_val(output))
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(input_state.size(0), self.action_num)
        output = val + adv - adv.mean(1).unsqueeze(1).expand(input_state.size(0), self.action_num)
        return output