from torchvision.models import wide_resnet50_2
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np

transform = transforms.Compose([
    transforms.Resize((32, 32)),            # Resize MNIST to match CIFAR
    transforms.Grayscale(num_output_channels=3),  # Make MNIST 3-channel
    transforms.ToTensor(),                  # Convert PIL → Tensor
])

cifar = CIFAR10(root='data', train=True, download=True)
data_np_cifar = cifar.data.astype(np.float32) / 255.0  # (50000, 32, 32, 3)
data_np_cifar = np.transpose(data_np_cifar, (0, 3, 1, 2))  # (50000, 3, 32, 32)

mnist = MNIST(root='data', train=True, download=True)
data_np_mnist = mnist.data[:50000].numpy().astype(np.float32) / 255.0  # (50000, 28, 28)
data_np_mnist = data_np_mnist[:, None, :, :]  # (50000, 1, 28, 28)
mnist_tensor = torch.from_numpy(data_np_mnist).float()
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)  # (50000, 3, 32, 32)
data_np_mnist = mnist_tensor.detach().cpu().numpy()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = wide_resnet50_2(weights=True).to(device)
cifar_loader = torch.utils.data.DataLoader(
    CIFAR10(root='data', train=True, download=True, transform=transform),
    batch_size=1, shuffle=True, num_workers=0
)
mnist_loader = torch.utils.data.DataLoader(
    MNIST(root='data', train=True, download=True, transform=transform),
    batch_size=1, shuffle=True, num_workers=0
)

def get_gradient_vector(model, dataloader):
    model.eval()
    model.zero_grad()
    x, y = next(iter(dataloader))
    x, y = x.to(device), y.to(device)

    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))  # flatten
    grad_vector = torch.cat(grads)
    return grad_vector

grad_cifar = get_gradient_vector(model, cifar_loader)
grad_mnist = get_gradient_vector(model, mnist_loader)

l2_diff = torch.norm(grad_cifar - grad_mnist).item()
cos_sim = torch.nn.functional.cosine_similarity(
    grad_cifar.unsqueeze(0), grad_mnist.unsqueeze(0)
).item()

print(f"L2 norm of gradient difference: {1/l2_diff:.4f}")
print(f"Cosine similarity between gradients: {cos_sim:.4f}")

