import torch
from privacy_analysis.compute_privacy_sgm import compute_dp_sgd_privacy
from math import sqrt



def DP_ACE(model, D_loader, D_s_loader, test_loader, lr, sigma, l, T, delta):
    """
    Parameters:
        D_loader: private training data loader
        n: private batch size
        D_s_loader: public training data loader
        n_s: public batch size
        lr: learning rate
        sigma: noise scale
        l: loss function
        T: number of training iterations
    """
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    n_private = len(D_loader.dataset)
    batch_size_private = D_loader.batch_size
    
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    pub_iter = iter(D_s_loader)

    C = 0

    for epoch in range(T):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for priv_inputs, priv_labels in D_loader:
            
            try:
                pub_inputs, pub_labels = next(pub_iter)
            except StopIteration:
                pub_iter = iter(D_s_loader)
                pub_inputs, pub_labels = next(pub_iter) 
            
            optim.zero_grad()
            pub_out = model(pub_inputs)
            pub_loss = l(pub_out, pub_labels)
            pub_loss.backward()
            
            pub_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    pub_grad_norm += p.grad.norm().item() ** 2
            pub_grad_norm = pub_grad_norm ** 0.5

            pub_grad = []
            for p in model.parameters():
                pub_grad.append(p.grad.detach().clone())
            pub_grad = [g.clone() for g in pub_grad]
            
            grads = [torch.zeros_like(p) for p in model.parameters()]
            noisy_grad_norms = []
            
            for i, (x, y) in enumerate(zip(priv_inputs, priv_labels)):
                optim.zero_grad()
                output = model(x.unsqueeze(0))
                loss = l(output, y.unsqueeze(0))
                loss.backward()
                
                noisy_norm = 0
                for j, p in enumerate(model.parameters()):
                    if p.grad is not None:
                        grads[j] += p.grad.clone()
                        noisy_norm += p.grad.norm().item() * (pub_grad_norm)/(max(pub_grad_norm, p.grad.norm().item()))
                noisy_norm += torch.normal(0, pub_grad_norm * sqrt(2) * sigma, (1,)).item()    
                noisy_grad_norms.append(noisy_norm)
            
            avg_pub_grad_norm = pub_grad_norm
            avg_noisy_priv_grad_norm = sum(noisy_grad_norms) / len(noisy_grad_norms)
            
            

            C = (0.3 * avg_pub_grad_norm + 0.7 * avg_noisy_priv_grad_norm)
            
            total_grad_norm = 0.0
            for g in grads:
                total_grad_norm += g.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            clip_factor = min(1.0, C / (total_grad_norm + 1e-8))
            
            for i, g in enumerate(grads):
                grads[i] = g * clip_factor
                grads[i] += torch.normal(0, sigma * C * sqrt(2), g.shape)
            
            optim.zero_grad()
            for p, g in zip(model.parameters(), grads):
                if p.requires_grad:
                    p.grad = g / len(priv_inputs)
            
            optim.step()

            total_loss += pub_loss.item() * priv_inputs.size(0)
            with torch.no_grad():
                priv_out = model(priv_inputs)
                _, predicted = torch.max(priv_out.data, 1)
                total += priv_labels.size(0)
                correct += (predicted == priv_labels).sum().item()
        
        epoch_loss = total_loss / total
        epoch_accuracy = 100 * correct / total

        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = l(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_accuracy = 100 * test_correct / test_total
        test_loss = test_loss / test_total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        eps, _ = compute_dp_sgd_privacy(n_private, batch_size_private, sigma, epoch + 1, delta)
        print(f"Epoch [{epoch+1}/{T}] ... Training Loss: {epoch_loss:.4f} ... Training Accuracy: {epoch_accuracy:.2f}% ... Test Loss: {test_loss:.4f} ... Test Accuracy: {test_accuracy:.2f}% ... Epsilon: {eps:.4f} ... Delta: {delta}")

    return train_losses, train_accuracies, test_losses, test_accuracies