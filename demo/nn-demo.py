import torch
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

torch.manual_seed(42)  # reproducibility


# ------------------------------------------------------------------ #
#  data generation & plotting                                         #
# ------------------------------------------------------------------ #
def generate_data(
    n_samples: int = 300,
    centers: int = 3,
    cluster_std: float = 0.7,
    skew_factor: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_per_class = n_samples // centers
    X_parts, y_parts = [], []
    blob_centers = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([3.0, 0.0]),
        torch.tensor([1.5, 2.5]),
    ]

    for i in range(centers):
        pts = torch.randn(n_per_class, 2) * cluster_std + blob_centers[i]
        if i in (1, 2):
            skew = torch.tensor(
                [[1.0, skew_factor * (i - 1)], [skew_factor * (i - 1), 1.0]]
            )
            pts = torch.mm(pts - blob_centers[i], skew) + blob_centers[i]
        X_parts.append(pts)
        y_parts.append(torch.full((n_per_class,), i, dtype=torch.long))

    return torch.cat(X_parts), torch.cat(y_parts)


def plot_raw_data(X: torch.Tensor, y: torch.Tensor) -> None:
    Xl, yl = X.tolist(), y.tolist()
    plt.scatter([p[0] for p in Xl], [p[1] for p in Xl], c=yl, alpha=0.8, cmap="viridis")
    plt.title("Raw data")
    plt.show()


# ------------------------------------------------------------------ #
#  model, loss, SGD step                                              #
# ------------------------------------------------------------------ #
Model = Dict[str, torch.Tensor]


def model_init(inp: int = 2, hid: int = 20, out: int = 3) -> Model:
    return {
        "w1": torch.randn(hid, inp, requires_grad=True),
        "b1": torch.randn(hid, requires_grad=True),
        "w2": torch.randn(out, hid, requires_grad=True),
        "b2": torch.randn(out, requires_grad=True),
    }


def params(model: Model) -> List[torch.Tensor]:
    return [model["w1"], model["b1"], model["w2"], model["b2"]]


def forward(model: Model, x: torch.Tensor) -> torch.Tensor:
    x = torch.mm(x, model["w1"].t()) + model["b1"]
    x = torch.max(torch.tensor(0.0), x)  # ReLU
    x = torch.mm(x, model["w2"].t()) + model["b2"]
    return x


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=1)
    chosen = torch.gather(logp, 1, targets.unsqueeze(1)).squeeze(1)
    return -chosen.mean()


def sgd_step(ps: List[torch.Tensor], lr: float = 0.1) -> None:
    """
    Vanilla SGD:  p ← p - lr * p.grad , then reset gradients.
    Operates in-place; returns nothing.
    """
    with torch.no_grad():
        for p in ps:
            if p.grad is not None:
                p -= lr * p.grad
                p.grad.zero_() if p.grad is not None else None


# ------------------------------------------------------------------ #
#  training loop                                                      #
# ------------------------------------------------------------------ #
def train(
    model: Model,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1000,
    lr: float = 0.1,
    record_every: int = 100,
) -> Tuple[List[float], List[int]]:
    losses, steps = [], []
    ps = params(model)

    for epoch in range(epochs):
        # forward & loss
        logits = forward(model, X)
        loss = cross_entropy_loss(logits, y)

        # zero existing grads, back-prop, SGD update
        for p in ps:
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        sgd_step(ps, lr)

        if (epoch + 1) % record_every == 0:
            losses.append(loss.item())
            steps.append(epoch + 1)
            print(f"epoch {epoch+1:4d}/{epochs}  loss {loss.item():.4f}")

    return losses, steps


# ------------------------------------------------------------------ #
#  decision-boundary plotting                                         #
# ------------------------------------------------------------------ #
def plot_results(X: torch.Tensor, y: torch.Tensor, model: Model) -> None:
    Xl, yl = X.tolist(), y.tolist()
    x_min, x_max = min(p[0] for p in Xl) - 1, max(p[0] for p in Xl) + 1
    y_min, y_max = min(p[1] for p in Xl) - 1, max(p[1] for p in Xl) + 1

    xs, ys = torch.arange(x_min, x_max, 0.1), torch.arange(y_min, y_max, 0.1)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    mesh = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    with torch.no_grad():
        logits = forward(model, mesh)
        Z = torch.argmax(logits, dim=1).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, cmap="viridis")
    plt.scatter([p[0] for p in Xl], [p[1] for p in Xl], c=yl, alpha=0.8, cmap="viridis")
    plt.title("Decision boundary")
    plt.show()


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    X, y = generate_data()
    plot_raw_data(X, y)

    net = model_init()
    losses, steps = train(net, X, y, epochs=3000, lr=0.1, record_every=100)

    plt.plot(steps, losses)
    plt.title("Training loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    plot_results(X, y, net)
