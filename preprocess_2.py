
## AI Assisted preprocessing functions to be used in context
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ContourDataset(Dataset):
    def __init__(self, df, normalize=True):
        self.df = df.reset_index(drop=True)
        self.normalize = normalize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image, mask = ppf.load_sample(row)

        if self.normalize:
            image = self.normalize_image(image)

        # Convert to torch tensors
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask  = torch.from_numpy(mask).unsqueeze(0)   # (1, H, W)

        return image, mask

    @staticmethod
    def normalize_image(image):
        mean = image.mean()
        std = image.std() + 1e-8
        return (image - mean) / std
##



dataset = ContourDataset(df)

image, mask = dataset[0]

print(image.shape, image.dtype)
print(mask.shape, mask.dtype)


plt.imshow(image[0], cmap="gray")
plt.contour(mask[0], colors="r")
plt.axis("off")
plt.show()

DataLoader(dataset, batch_size=8, shuffle=True)
# Double checking that we have 45 unique patients
unique_pids = df["dicom_pid"].unique()
print(len(unique_pids))

rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
rng.shuffle(unique_pids)

train_frac = 0.9
n_train = int(train_frac * len(unique_pids))

train_pids = unique_pids[:n_train]
val_pids   = unique_pids[n_train:]

train_df = df[df["dicom_pid"].isin(train_pids)].reset_index(drop=True)
val_df   = df[df["dicom_pid"].isin(val_pids)].reset_index(drop=True)

train_dataset = ContourDataset(train_df)
val_dataset = ContourDataset(val_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,        # start small
    shuffle=True,        # important for training
    num_workers=0,       # 0 on Windows if issues
    pin_memory=False      # speeds up GPU transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)
'''
img, mask = train_dataset[0]
print(img.shape, img.dtype)
print(mask.shape, mask.unique())

imgs, masks = next(iter(train_loader))
print(imgs.shape)
'''

import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder (skip connections!)
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

criterion = nn.BCEWithLogitsLoss()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)


images, masks = next(iter(train_loader))
images = images.to(device)
masks = masks.to(device)



'''
imgs = imgs.to(device, dtype=torch.float32)
masks = masks.to(device, dtype=torch.float32)

print("Images shape:", imgs.shape)
print("Masks shape: ", masks.shape)
print("Images device:", imgs.device)
print("Masks device: ", masks.device)
print("Model device:", next(model.parameters()).device)
'''

'''
with torch.no_grad():
    outputs = model(images)

with torch.no_grad():
    preds = model(imgs)

print("Predictions shape:", preds.shape)
'''

num_epochs = 5

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for imgs, masks in train_loader:
        imgs = imgs.to(device)
        masks = masks.to(device).float()

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch}: loss = {train_loss/len(train_loader):.4f}")
'''
print(preds.dtype)
print(masks.dtype)
'''
'''
import torch
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
'''

