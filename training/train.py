import torch
import copy
import numpy as np

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=20, verbose=True):
    """
    Train a PyTorch model with early stopping based on validation loss.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to train on (cpu or cuda)
        epochs: Number of epochs to train
        verbose: Whether to print training progress
    
    Returns:
        history: Dictionary with training history
        best_model_weights
    """

    model.to(device)

    best_val_loss = float('inf')
    # best_val_loss = 0.1
    patience = 5
    patience_counter = 0
    history = {'epoch': [], 'train_loss': [],'val_loss': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_batches = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_batches += loss.item()

        # Validation phase
        model.eval()
        val_loss_batches = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)

                val_loss_batches += criterion(y_pred, y_batch).item()

        # Calculate average loss (across batches) for the epoch
        avg_train_loss = train_loss_batches / len(train_loader)
        avg_val_loss = val_loss_batches / len(val_loader)
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}"
            ) 

        # Early stopping 
        if avg_val_loss < best_val_loss or epoch == 0:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1 
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
                break
    
    # Load best model weights before returning
    model.load_state_dict(best_model_weights)

    history['best_epoch'] = best_epoch # = history['epoch'][np.argmin(history['val_loss'])]
    history['best_val_loss'] = best_val_loss
    
    return history, model


def predict_model(model, test_loader, device):

    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds)

    return preds
