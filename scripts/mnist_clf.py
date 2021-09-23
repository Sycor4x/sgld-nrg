import numpy as np
import torch


def clf_train_one_epoch(writer):
    pre_train_buff_size = max(1, int(512 / user_args.batch_size + 0.5))
    pre_train_buff_xe = np.zeros(pre_train_buff_size)
    pre_train_buff_acc = np.zeros(pre_train_buff_size)
    pre_train_buff_pnrg = np.zeros(pre_train_buff_size)
    pre_val_buff = np.zeros(len(test))
    pre_val_buff_acc = np.zeros(len(test))
    pre_val_buff_pnrg = np.zeros(len(test))
    pre_train_counter = 0
    pointer = 0
    xe_loss = 0.0
    dir = user_args.save.absolute().parent
    stem = user_args.save.absolute().stem
    print(stem)
    for pre_train_epoch in range(user_args.pre_epochs):
        print(f"Pre-train epoch {pre_train_epoch} of {user_args.pre_epochs}")
        start_time = datetime.datetime.now()
        for i, (x_train, y_train) in enumerate(train):
            my_net.train()
            pre_train_counter += x_train.size(0)
            y_logit = my_net(x_train)
            xe_loss = xe_loss_fn(y_logit, y_train)
            pnrg = my_net.logsumexp_logits(x_train).mean()
            total_loss = xe_loss + user_args.pretrain_energy_penalty * pnrg
            total_loss.backward()
            pre_train_optim.step()
            pre_train_optim.zero_grad()
            pre_train_buff_acc[pointer % pre_train_buff_size] = get_accuracy(
                y_logit, y_train
            )
            pre_train_buff_xe[pointer % pre_train_buff_size] = xe_loss.item()
            pre_train_buff_pnrg[pointer % pre_train_buff_size] = pnrg
            pointer += 1
            if i % pre_train_buff_size == 0 and i > 0:
                elapsed = (datetime.datetime.now() - start_time) / datetime.timedelta(
                    seconds=1
                )
                epoch_remaining = estimate_time_remaining(
                    elapsed_seconds=elapsed, batch_offset=i, num_batch=len(train)
                )
                print(
                    f"\tPre-train batch {i} of {len(train)} - est. {epoch_remaining:.3f} hours remaining"
                )
                writer.add_scalar(
                    "pre/accuracy/train", pre_train_buff_acc.mean(), pre_train_counter
                )
                writer.add_scalar(
                    "pre/xe_loss/train", pre_train_buff_xe.mean(), pre_train_counter
                )
                writer.add_scalar(
                    "combined/seconds_per_instance",
                    elapsed / ((i + 1) * user_args.batch_size),
                    pre_train_counter,
                )
                writer.add_scalar(
                    "combined/accuracy", pre_train_buff_acc.mean(), pre_train_counter
                )
                writer.add_scalar(
                    "combined/xe", pre_train_buff_xe.mean(), pre_train_counter
                )
                writer.add_scalar(
                    "combined/+nrg", pre_train_buff_pnrg.mean(), pre_train_counter
                )
                for name, weight in my_net.named_parameters():
                    writer.add_histogram(f"param/{name}", weight, pre_train_counter)
                    writer.add_histogram(f"grad/{name}", weight.grad, pre_train_counter)

        for j, (x_val, y_val) in enumerate(test):
            my_net.eval()
            y_logit = my_net(x_val)
            xe_loss = xe_loss_fn(y_logit, y_val)
            pre_val_buff[j] = xe_loss.item()
            pre_val_buff_acc[j] = get_accuracy(y_logit, y_val)
            pre_val_buff_pnrg[j] = my_net.logsumexp_logits(x_val).mean()
        writer.add_scalar(
            "pre/accuracy/val", pre_val_buff_acc.mean(), pre_train_counter
        )
        writer.add_scalar("pre/xe_loss/val", pre_val_buff.mean(), pre_train_counter)
        writer.add_scalar("pre/+nrg/val", pre_val_buff_pnrg.mean(), pre_train_counter)
        fname = f"{stem}-pre-epoch-{pre_train_epoch}.pth"
        pre_dest = dir.joinpath(fname)
        print(f"\tSaving epoch {pre_train_epoch} to {pre_dest}")
        torch.save(
            {
                "epoch": pre_train_epoch,
                "model_state_dict": my_net.state_dict(),
                "optimizer_state_dict": pre_train_optim.state_dict(),
                "loss": xe_loss,
            },
            str(pre_dest),
        )
    return pre_train_counter


if __name__ == "__main__":
    pass
