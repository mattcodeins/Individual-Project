def to_numpy(x):
    return x.detach().cpu().numpy()  # convert a torch tensor to a numpy array


def training_loop():
    model.train()
    logs = []
    for i in range(N_epochs):
        # train step is whole training dataset (minibatched inside function)
        loss, nll, kl = train_step(model, opt, nelbo, train_loader, device=device)
        logs.append([to_numpy(nll), to_numpy(kl), to_numpy(loss), to_numpy(nll)/to_numpy(kl)])

        if (i+1) % 1 == 0:
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    m, v = predict(model, x_test.reshape((-1,784)), K=50)
                    pred = m.max(1, keepdim=True)[1]
                    correct += pred.eq(y_test.to(device).view_as(pred)).sum()
            print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
                correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

            logs.append([to_numpy(nll), to_numpy(kl), to_numpy(loss),
                            to_numpy(torch.log(1 + torch.exp(model._prior_std)))])

            print("Epoch {}, nll={}, kl={}, nelbo={}, prior_std={}"
                    .format(i+1, logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3]))
    logs = np.array(logs)



def train_step(model, opt, nelbo, dataloader, device):
    tloss, tnll, tkl = 0,0,0
    for _, (x, y) in enumerate(dataloader):
        opt.zero_grad()
        batch_size = x.shape[0]
        minibatch_ratio = batch_size / len(dataloader.dataset)
        x = x.to(device).reshape((batch_size, 784)); y = y.to(device)
        y_pred = model(x)
        loss, nll, kl = nelbo(model, (y_pred, y), minibatch_ratio)
        loss.backward(retain_graph=True)
        opt.step()
        tloss += loss; tnll += nll; tkl += kl
    return tloss, tnll, tkl