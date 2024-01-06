import time
import torch
import torch.nn.functional as F

num_classes = 2
batch_size = 5000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = torch.randint(0, num_classes, (batch_size,)).to(device)
one_hot_labels = torch.empty(batch_size, num_classes).to(device)

for _ in range(5000):  # for caching
    one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
start_time = time.time()
for _ in range(5000):
    one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
scatter_time = time.time() - start_time

for _ in range(5000):
    one_hot_labels[torch.arange(batch_size), labels] = 1
start_time = time.time()
for _ in range(5000):
    one_hot_labels[torch.arange(batch_size), labels] = 1
advanced_indexing_time = time.time() - start_time

for _ in range(5000):
    one_hot_labels = F.one_hot(labels, num_classes)
start_time = time.time()
for _ in range(5000):
    one_hot_labels = F.one_hot(labels, num_classes)
pytorch_time = time.time() - start_time

print(f"Scatter Time: {scatter_time}")
print(f"Advanced Indexing Time: {advanced_indexing_time}")
print(f"PyTorch Time: {pytorch_time}")
