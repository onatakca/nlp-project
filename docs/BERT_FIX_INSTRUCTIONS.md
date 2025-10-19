# Quick Fix for BERT DataParallel Error

## The Problem
When using `DataParallel` with 8 GPUs, the loss comes back as a **tensor of 8 values** (one per GPU), not a single scalar. You need to average them.

## The Fix

In cell `cell-23` (the training functions), find these two lines:

```python
loss = outputs.loss
# Scale loss for gradient accumulation
loss = loss / GRADIENT_ACCUMULATION_STEPS
```

**Replace with:**
```python
loss = outputs.loss
# FIX: DataParallel returns loss per GPU, average them
if loss.dim() > 0:  # If it's a vector (multi-GPU)
    loss = loss.mean()
# Scale loss for gradient accumulation
loss = loss / GRADIENT_ACCUMULATION_STEPS
```

**Do this in BOTH places** (inside the `if scaler is not None:` block AND the `else:` block).

## Step-by-Step Instructions

1. **Restart the kernel** - In Jupyter: Kernel → Restart & Clear Output
2. **Run all cells up to cell-22** (before the training functions)
3. **In cell-23**, manually edit the code to add the fix above
4. **Run the rest of the cells**

## Or Use This Complete Fixed Function

Delete everything in cell-23 and replace with:

```python
import time

def train_epoch_optimized(model, dataloader, optimizer, scheduler, device, scaler=None):
    """Optimized training with mixed precision and gradient accumulation"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc='Training')
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Mixed precision forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
                # FIX: DataParallel returns loss per GPU, average them
                if loss.dim() > 0:
                    loss = loss.mean()
                # Scale loss for gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            # Backward pass with scaling
            scaler.scale(loss).backward()

            # Update weights every N steps
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            # Regular FP32 training
            outputs = model(**batch)
            loss = outputs.loss
            # FIX: DataParallel returns loss per GPU, average them
            if loss.dim() > 0:
                loss = loss.mean()
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        progress_bar.set_postfix({'loss': loss.item() * GRADIENT_ACCUMULATION_STEPS})

    return total_loss / len(dataloader)


def evaluate_optimized(model, dataloader, device):
    """Optimized evaluation"""
    model.eval()
    predictions = []
    true_labels = []

    progress_bar = tqdm(dataloader, desc='Evaluating')
    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Use mixed precision for inference too
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    return np.array(predictions), np.array(true_labels)

print("✅ Training functions defined (FIXED)")
```

Then run the training cell.
