import torch
import torch.nn as nn


def adversarial_train_step(model, X_batch, y_batch, criterion, optimizer, device,
                           epsilon=0.1, alpha=0.5, schema=None, feature_names=None,
                           feature_types=None):
    """
    Performs one training step with mixed clean and adversarial data.
    Uses constraint projection when schema is provided.
    """
    model.eval()

    steps = 3
    step_size = 1.25 * epsilon / steps

    x_adv = X_batch.clone().detach()
    x_adv.requires_grad = True

    use_constraints = schema is not None and feature_names is not None and feature_types is not None
    if use_constraints:
        from attacks.capgd import project_constraints

    for _ in range(steps):
        outputs = model(x_adv)
        loss = criterion(outputs, y_batch)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad
            x_adv = x_adv + step_size * grad.sign()
            delta = torch.clamp(x_adv - X_batch, -epsilon, epsilon)
            x_adv = X_batch + delta

            if use_constraints:
                x_adv = project_constraints(x_adv, X_batch, schema, feature_names, feature_types)

            x_adv.requires_grad = True

    # Train on mixed batch
    model.train()
    optimizer.zero_grad()

    out_clean = model(X_batch)
    loss_clean = criterion(out_clean, y_batch)

    out_adv = model(x_adv.detach())
    loss_adv = criterion(out_adv, y_batch)

    total_loss = alpha * loss_clean + (1 - alpha) * loss_adv

    total_loss.backward()
    optimizer.step()

    return total_loss.item()
