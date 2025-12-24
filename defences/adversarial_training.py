import torch
import torch.nn as nn

def adversarial_train_step(model, X_batch, y_batch, criterion, optimizer, device, epsilon=0.1, alpha=0.5):
    """
    Performs one training step with mixed clean and adversarial data.
    """
    # 1. Generate Adversarial Examples
    model.eval()
    
    # Simple FGSM or PGD-1 for speed in training?
    # Or PGD-3/5.
    # Let's do PGD-3 for MVP robustness.
    steps = 3
    step_size = 1.25 * epsilon / steps
    
    x_adv = X_batch.clone().detach()
    x_adv.requires_grad = True
    
    # For constraints: we assume data is standardized (approx -3 to 3).
    # Simple clipping to typical range or assume unbounded for feature space training?
    # We should respecting min/max of the batch approx?
    # For MVP, just L-inf ball.
    
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
            x_adv.requires_grad = True
            
    # 2. Train on Mixed Batch (or just Adv?)
    model.train()
    optimizer.zero_grad()
    
    # Clean Loss
    out_clean = model(X_batch)
    loss_clean = criterion(out_clean, y_batch)
    
    # Adv Loss
    out_adv = model(x_adv.detach())
    loss_adv = criterion(out_adv, y_batch)
    
    total_loss = alpha * loss_clean + (1 - alpha) * loss_adv
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()
