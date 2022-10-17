import torch
from mmseg.core.evaluation import metrics
from tqdm import tqdm
import numpy as np


def training_loop(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, print_every):
    print("Starting training")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #单卡
    model = model.to(device)
   
    train_losses, train_mIoUs, val_losses, val_mIoUs = [], [], [], []

    for epoch in range(1, num_epochs+1):
        model, train_loss, train_mIoU = train_epoch(model,
                                                   optimizer,
                                                   loss_fn,
                                                   train_loader,
                                                   val_loader,
                                                   device,
                                                   print_every)
        val_loss, val_mIoU = validate(model, loss_fn, val_loader, device)
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
              f"Train acc.: {sum(train_mIoU)/len(train_mIoU):.3f}, "
              f"Val. loss: {val_loss:.3f}, "
              f"Val. acc.: {val_mIoU:.3f}")
        train_losses.extend(train_loss)
        train_mIoUs.extend(train_mIoU)
        val_losses.append(val_loss)
        val_mIoUs.append(val_mIoU)
    return model, train_losses, train_mIoUs, val_losses, val_mIoUs

def train_epoch(model, optimizer, loss_fn, train_loader, val_loader, device, print_every):
    # Train:
    model.train()
    train_loss_batches, train_IoU_batches = [], []
    num_batches = len(train_loader)
    for batch_index, (x, y) in tqdm(enumerate(train_loader, 1)):
        inputs, labels = x, y
        labels = labels.to(device) - 1
        optimizer.zero_grad()
        deeplabv3p_logits_res = inputs['deeplabv3p'].to(device)
        pspnet_logits_res = inputs['pspnet'].to(device)
        fcn_logits_res = inputs['fcn'].to(device)
        z = model.forward(deeplabv3p_logits_res, pspnet_logits_res, fcn_logits_res)
        loss = loss_fn(z, labels)
        loss.backward()
        optimizer.step()
        # 多GPU
        # optimizer.module.step()
        train_loss_batches.append(loss.item())
        pred = z.argmax(1)
        mIoU = np.nanmean(metrics.mean_iou(labels.cpu().numpy(), pred.cpu().numpy(), 150, -1)['IoU'])
        train_IoU_batches.append(mIoU.item())

        # delete caches
        del inputs, deeplabv3p_logits_res, pspnet_logits_res, fcn_logits_res, z, labels, mIoU, pred, loss
        torch.cuda.empty_cache()

        # If you want to print your progress more often than every epoch you can
        # set `print_every` to the number of batches you want between every status update.
        # Note that the print out will trigger a full validation on the full val. set => slows down training
        if print_every is not None and batch_index % print_every == 0:
            val_loss, val_acc = validate(model, loss_fn, val_loader, device)
            model.train()
            print(f"\tBatch {batch_index}/{num_batches}: "
                  f"\tTrain loss: {sum(train_loss_batches[-print_every:])/print_every:.3f}, "
                  f"\tTrain IoU.: {sum(train_IoU_batches[-print_every:])/print_every:.3f}, "
                  f"\tVal. loss: {val_loss:.3f}, "
                  f"\tVal. IoU.: {val_acc:.3f}")

    return model, train_loss_batches, train_IoU_batches

def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_mIoU_cum = 0
    model.eval()
    print("Validating...")
    with torch.no_grad():
        for batch_index, (x, y) in tqdm(enumerate(val_loader, 1)):
            inputs, labels = x, y
            labels -= 1
            labels = labels.to(device)
            deeplabv3p_logits_res = inputs['deeplabv3p'].to(device)
            pspnet_logits_res = inputs['pspnet'].to(device)
            fcn_logits_res = inputs['fcn'].to(device)
            z = model.forward(deeplabv3p_logits_res, pspnet_logits_res, fcn_logits_res)
            batch_loss = loss_fn(z, labels)
            val_loss_cum += batch_loss.item()
            pred = z.argmax(1)
            mIoU = np.nanmean(metrics.mean_iou(labels.cpu().numpy(), pred.cpu().numpy(), 150, -1)['IoU'])
            val_mIoU_cum += mIoU.item()

            del inputs, deeplabv3p_logits_res, pspnet_logits_res, fcn_logits_res, z, labels, mIoU, pred, batch_loss
            torch.cuda.empty_cache()
            
    return val_loss_cum/len(val_loader), val_mIoU_cum/len(val_loader)

def plot_data(train_accs, val_accs, train_losses, val_losses, step_size):
    import matplotlib.pyplot as plt
    train_accs_step = train_accs[::step_size]
    val_accs_step = val_accs
    x = [str(i) for i in range(len(train_accs_step))]

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 4))

    ax1.plot(x, train_accs_step,'-o',label='Training accuracy')
    ax1.plot(x, val_accs_step,'-o',label='Validation accuracy')
    ax1.legend(loc='best')

    train_losses_step = train_losses[::step_size]
    val_losses_step = val_losses
    x = [str(i) for i in range(len(train_losses_step))]
    ax2.plot(x, train_losses_step,'-o',label='Training loss')
    ax2.plot(x, val_losses_step,'-o',label='Validation loss')
    ax2.legend(loc='best')
    plt.show()