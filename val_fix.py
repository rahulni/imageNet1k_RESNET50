# # import os, json, shutil

# # VAL_DIR = "/mnt/imagenet/val"
# # META_PATH = "/mnt/imagenet/imagenet_class_index.json"

# # # Load mapping: {int -> WNID}
# # idx_to_wnid = {int(k): v[0] for k, v in json.load(open(META_PATH)).items()}

# # # Get sorted folders (0001, 0002, ...)
# # val_folders = sorted([d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))])

# # if len(val_folders) != 1000:
# #     raise ValueError(f"Expected 1000 val folders, found {len(val_folders)}")

# # for i, folder in enumerate(val_folders, start=1):
# #     wnid = idx_to_wnid[i - 1]
# #     src = os.path.join(VAL_DIR, folder)
# #     dst = os.path.join(VAL_DIR, wnid)
# #     if src != dst:
# #         shutil.move(src, dst)
# #         print(f"Renamed {folder} → {wnid}")

# # print("✅ Validation folder structure fixed!")


# from datasets import load_dataset
# from torchvision import transforms

# # Load ImageNet-1K validation split from Hugging Face
# imagenet_val_data = load_dataset("ILSVRC/imagenet-1k", split="validation")

# # Wrap HF dataset in a simple PyTorch Dataset
# from torch.utils.data import Dataset

# class HFImageNetValDataset(Dataset):
#     def __init__(self, hf_dataset, transform=None):
#         self.ds = hf_dataset
#         self.transform = transform

#     def __len__(self):
#         return len(self.ds)

#     def __getitem__(self, idx):
#         sample = self.ds[idx]
#         img = sample["image"].convert("RGB")
#         label = int(sample["label"])
#         if self.transform:
#             img = self.transform(img)
#         return img, label

# val_dataset = HFImageNetValDataset(imagenet_val_data, transform=val_transform)
# print("✅ Validation dataset loaded from Hugging Face with", len(val_dataset), "samples.")

python3 - <<'EOF'
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, IterableDataset

imagenet_val_stream = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
print("Loaded validation stream.")

tform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

class TestVal(IterableDataset):
    def __init__(self, stream, transform): self.stream, self.transform = stream, transform
    def __iter__(self):
        for s in self.stream:
            img = s["image"]
            if not isinstance(img, Image.Image): img = Image.open(img).convert("RGB")
            yield self.transform(img), s["label"]

ds = TestVal(imagenet_val_stream, tform)
print(next(iter(ds)))
EOF
