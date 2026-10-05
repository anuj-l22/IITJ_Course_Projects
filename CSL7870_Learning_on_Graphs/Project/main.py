import torch
from torch_geometric.loader import DataLoader
from ogb.graphproppred import Evaluator
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import numpy as np
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
# GOOD / graph_ood import (support both namespaces)
# GOOD / graph_ood imports (prefer installed wheel; fallback to bundled GOOD)
# GOOD / graph_ood imports (prefer installed wheel; fallback to bundled GOOD)
try:
    from graph_ood.data.good_datasets.good_cmnist import GOODCMNIST
    from graph_ood.data.good_datasets.good_motif import GOODMotif
    from graph_ood.data.good_datasets.good_hiv import GOODHIV
    from graph_ood.data.good_datasets.good_pcba import GOODPCBA
except ImportError:
    from GOOD.data.good_datasets.good_cmnist import GOODCMNIST
    from GOOD.data.good_datasets.good_motif import GOODMotif
    from GOOD.data.good_datasets.good_hiv import GOODHIV
    from GOOD.data.good_datasets.good_pcba import GOODPCBA

import random
from gnn2 import GINNet
from model import CausalGraphon
from graphon import stat_graph
import pdb
from torch.utils.tensorboard import SummaryWriter
from algos.dg import RDM, URMHook
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
import csv, os

def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.CEX = False

def eval1(model, loader, device):

    model.eval()
    correct = 0
    for data in loader:

        data = data.to(device)
        with torch.no_grad():
            pred = model(data)['pred_cau'].max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

def eval2(model, evaluator, loader, device):
    model.eval()

    y_true = []
    y_pred = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)['pred_cau']
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    output = evaluator.eval(input_dict)
    return output

def print_data(args):
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.BCEWithLogitsLoss()
    if args.dataset == "cmnist":
        dataset, meta_info = GOODCMNIST.load(args.data_dir, domain='color', shift=args.shift, generate=False)
        num_class = 10
        num_layer = 5
        in_dim = 3
        eval_metric = "rocauc"
        cri = criterion1
        eval_name = None
        test_batch_size = args.batch_size

    elif args.dataset == "motif":
        dataset, meta_info = GOODMotif.load(args.data_dir, domain=args.domain, shift=args.shift, generate=False)
        num_class = 3
        num_layer = 3
        in_dim = 1
        eval_metric = "rocauc"
        cri = criterion1
        eval_name = None
        test_batch_size = args.batch_size

    elif args.dataset == "hiv":
        dataset, meta_info = GOODHIV.load(args.data_dir, domain=args.domain, shift=args.shift, generate=False)
        num_class = 1
        num_layer = 3
        in_dim = 9
        eval_metric = "rocauc"
        cri = criterion2
        eval_name = "ogbg-molhiv"
        test_batch_size = 256

    elif args.dataset == "pcba":
        dataset, meta_info = GOODPCBA.load(args.data_dir, domain=args.domain, shift=args.shift, generate=False)
        in_dim = 9
        num_class = 128
        num_layer = 5
        eval_metric = "ap"
        cri = criterion2
        eval_name = "ogbg-molpcba"
        test_batch_size = 1024

    else:
        assert False
    print(meta_info)
    return dataset, meta_info, in_dim, num_class, num_layer, eval_metric, cri, eval_name, test_batch_size


def main(args, trail):

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset, meta_info, in_dim, num_class, num_layer, eval_metric, criterion, eval_name, test_batch_size = print_data(args)
    if args.layer != -1:
        num_layer = args.layer
    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset["val"], batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(dataset["test"],  batch_size=test_batch_size, shuffle=False)

    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(dataset["train"])
    # model = GINNet(num_class=num_class, dataset=args.dataset, num_layer=num_layer, in_dim=in_dim, emb_dim=args.emb_dim, dropout_rate=args.dropout_rate, args=args).to(device)                       
    model = CausalGraphon(args=args, num_class=num_class, 
                            in_dim=in_dim,
                            emb_dim=args.emb_dim,
                            fro_layer=num_layer,
                            bac_layer=num_layer,
                            cau_layer=num_layer,
                            dropout_rate=args.dropout_rate,
                            cau_gamma=args.cau_gamma,
                            env_gamma=args.env_gamma,
                            use_linear=args.use_linear,
                            graphon=args.graphon,
                            N=int(median_num_nodes)).to(device)
    model.ratio = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)

    if args.save_model and trail==1:
        torch.save([args, num_class, in_dim, num_layer, int(median_num_nodes)], "./run/{}-{}-{}-args.pt".format(args.time, args.dataset, args.domain))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2reg)

    if args.lr_scheduler == 'step':
        # sch_attacker = StepLR(opt_attacker, step_size=args.lr_decay, gamma=args.lr_gamma)
        sch = StepLR(optimizer, step_size=args.lr_decay, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'multi':
        # sch_attacker = MultiStepLR(opt_attacker, milestones=args.milestones, gamma=args.lr_gamma)
        sch = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cos':
        # sch_attacker = CosineAnnealingLR(opt_attacker, T_max=args.epochs)
        sch = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        pass
    scaler = GradScaler(enabled=args.amp)
    writer = SummaryWriter(log_dir=os.path.join(
        args.logdir, f'{args.time}-{args.dataset}-{args.domain}-tr{trail}'))

    # Hooks / wrappers for algorithms
    rdm = RDM(weight=args.rdm_lambda) if args.algo == 'rdm' else None
    urm = URMHook(feat_dim=args.emb_dim, device=device,
                d_lr=args.urm_d_lr, d_steps=args.urm_d_steps) if args.algo == 'urm' else None

    # ERM++: SWA
    use_swa = (args.algo == 'ermpp' and args.swa)
    swa_model = AveragedModel(model) if use_swa else None
    swa_scheduler = SWALR(optimizer, swa_lr=args.lr) if use_swa else None

    best_val, best_test, best_epoch = 0.0, 0.0, 0
    epochs_bad = 0

    results = {'highest_valid': 0,
               'update_test': 0,
               'update_epoch': 0, }
    start_time = time.time()


    # for epoch in range(1, args.epochs+1):
    #     start_time_local = time.time()
    #     total_loss = 0
        
    #     show  = int(float(len(train_loader)) / 2.0)
    #     correct = 0
    #     SufLo = 0
    #     InvLo = 0
    #     GraLo = 0
    #     for step, batch in enumerate(train_loader):

    #         batch = batch.to(device)
    #         model.train()
    #         out = model(batch, epoch=epoch)
    #         r = torch.min(out['causal']['edge_key'].mean(), model.ratio)
    #         r_real = out['causal']['edge_key'].mean() # learned causal feature ratio, check this each epoch
            
    #         pred = out['pred_cau'].max(1)[1]
    #         correct += pred.eq(batch.y.view(-1)).sum().item()
    #         optimizer.zero_grad()
    #         if args.dataset == "motif" or args.dataset == "cmnist":
    #             one_hot_target = batch.y.view(-1)
    #             uniform_target = torch.ones_like(out['pred_cau']) / num_class
    #             cau_loss = criterion(out['pred_cau'], one_hot_target)
    #             if args.random_add == 'shuffle':
    #                 inv_loss = criterion(out['pred_add'], one_hot_target)
    #             else:
    #                 inv_loss = 0
    #             env_loss = F.kl_div(F.log_softmax(out['pred_env'], dim=-1), uniform_target, reduction='batchmean')
    #             gra_loss = out['graphon_loss'] + 1.0 * (r-model.ratio) ** 2
    #             reg_loss = out['cau_loss_reg']
    #             loss = args.cau * cau_loss + args.env * env_loss + args.gra * gra_loss + args.reg * reg_loss + args.inv * inv_loss
    #         else:
    #             is_labeled = batch.y == batch.y
    #             uniform_target = torch.ones_like(out['pred_cau']) / num_class
    #             cau_loss = criterion(out['pred_cau'].to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
    #             if args.random_add == 'shuffle':
    #                 inv_loss = criterion(out['pred_add'].to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
    #             else:
    #                 inv_loss = 0
    #             env_loss = F.kl_div(F.log_softmax(out['pred_env'], dim=-1), uniform_target, reduction='batchmean')
    #             gra_loss = out['graphon_loss'] + 1.0 * (r-model.ratio) ** 2
    #             reg_loss = out['cau_loss_reg']
    #             loss = args.cau * cau_loss + args.env * env_loss + args.gra * gra_loss + args.reg * reg_loss + args.inv * inv_loss
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #         SufLo += cau_loss + reg_loss
    #         InvLo += inv_loss
    #         GraLo += gra_loss
    #         if step % show == 0:
    #             print("Ep:[{}/{}] TrIter:[{:<3}/{}] Lo:[{:.4f}] R:[{:.8f}]".format(epoch, args.epochs, step, len(train_loader), total_loss / (step + 1), model.ratio))
        
    #     train_result = correct / len(train_loader.dataset)
    #     epoch_loss = total_loss / len(train_loader)
    #     SufLo = SufLo / len(train_loader)
    #     InvLo = InvLo / len(train_loader)
    #     GraLo = GraLo / len(train_loader)

    #     if args.dataset == "motif" or args.dataset == "cmnist":
    #         valid_result = eval1(model, valid_loader, device)
    #         test_result = eval1(model, test_loader, device)
    #     else:
    #         evaluator = Evaluator(eval_name)
    #         valid_result = eval2(model, evaluator, valid_loader, device)[eval_metric]
    #         test_result  = eval2(model, evaluator, test_loader,  device)[eval_metric] 

    #     if args.save_model and epoch%10==0 and trail<=3:
    #         torch.save(model.state_dict(), "./model_{}/{}-{}-tr{}-ep{}.pt".format(args.time, args.dataset, args.domain, trail, epoch))
    #     if valid_result > results['highest_valid']:
    #         results['highest_valid'] = valid_result
    #         results['update_test'] = test_result
    #         results['update_epoch'] = epoch
    #         if args.save_model and trail<=3:
    #             torch.save(model.state_dict(), "./model_{}/{}-{}-tr{}-ep{}.pt".format(args.time, args.dataset, args.domain, trail, epoch))

    #     print("-" * 150)
    #     print("Tr:[{}/{}], Ep:[{}/{}] | Lo:[{:.4f}], SufLo:[{:.4f}], InvLo:[{:.4f}], GraLo:[{:.4f}] | tr:[{:.2f}], va:[{:.2f}], te:[{:.2f}] | Best va:[{:.2f}], te:[{:.2f}] at:[{}] | ep time:{:.2f} min"
    #                     .format(trail, args.trails, epoch, args.epochs, 
    #                             epoch_loss, SufLo, InvLo, GraLo,
    #                             train_result*100, valid_result*100, test_result*100,
    #                             results['highest_valid']*100, results['update_test']*100, results['update_epoch'],
    #                             (time.time()-start_time_local) / 60))
    #     print("-" * 150)
    # total_time = time.time() - start_time
    # print("Best va:[{:.2f}], te:[{:.2f}] at epoch:[{}] | Total time:{}"
    #         .format(results['highest_valid']*100, results['update_test']*100, results['update_epoch'],
    #                 time.strftime('%H:%M:%S', time.gmtime(total_time))))
    # return results['update_test']

    for epoch in range(1, args.epochs + 1):
        start_time_local = time.time()
        total_loss = 0.0

        show = max(1, int(len(train_loader) / 2.0))
        correct = 0
        SufLo = 0.0
        InvLo = 0.0
        GraLo = 0.0

        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            model.train()

            with autocast(enabled=args.amp):
                out = model(batch, epoch=epoch)

                # causal ratio tracking (UIL)
                r = torch.min(out['causal']['edge_key'].mean(), model.ratio)
                r_real = out['causal']['edge_key'].mean()

                # predictions for running train accuracy (classification)
                if args.dataset in ["motif", "cmnist"]:
                    pred_cls = out['pred_cau'].max(1)[1]
                    correct += pred_cls.eq(batch.y.view(-1)).sum().item()

                # ---- supervised losses & per-sample risks ----
                if args.dataset in ["motif", "cmnist"]:
                    target = batch.y.view(-1)
                    logits = out['pred_cau']
                    if (args.algo == 'ermpp') and (args.label_smoothing > 0.0):
                        C = logits.size(-1)
                        ls = args.label_smoothing
                        one_hot = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1)
                        smooth = one_hot * (1 - ls) + ls / C
                        cau_loss = -(smooth * torch.log_softmax(logits, dim=-1)).sum(dim=1).mean()
                        per_sample_ce = -(smooth * torch.log_softmax(logits, dim=-1)).sum(dim=1)
                    else:
                        cau_loss = F.cross_entropy(logits, target)
                        per_sample_ce = F.cross_entropy(logits, target, reduction='none')
                else:
                    is_labeled = batch.y == batch.y
                    logits = out['pred_cau'].to(torch.float32)[is_labeled]
                    targets = batch.y.to(torch.float32)[is_labeled]
                    cau_loss = F.binary_cross_entropy_with_logits(logits, targets)
                    per_sample_ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

                # invariance aux (repo behavior)
                if args.random_add == 'shuffle':
                    if args.dataset in ["motif", "cmnist"]:
                        inv_loss = F.cross_entropy(out['pred_add'], batch.y.view(-1))
                    else:
                        inv_loss = F.binary_cross_entropy_with_logits(
                            out['pred_add'].to(torch.float32)[is_labeled], targets
                        )
                else:
                    inv_loss = cau_loss.new_tensor(0.0)

                # env & graphon terms (UIL)
                uniform_target = torch.ones_like(out['pred_env']) / out['pred_env'].size(-1)
                env_loss = F.kl_div(F.log_softmax(out['pred_env'], dim=-1), uniform_target, reduction='batchmean')
                gra_loss = out['graphon_loss'] + 1.0 * (r - model.ratio) ** 2
                reg_loss = out['cau_loss_reg']

                # ---- algorithm switch ----
                if args.algo == 'uil':
                    loss = args.cau * cau_loss + args.env * env_loss + args.gra * gra_loss + args.reg * reg_loss + args.inv * inv_loss

                elif args.algo in ['erm', 'ermpp']:
                    loss = cau_loss

                elif args.algo == 'rdm':
                    # per-domain risk alignment
                    rdm_loss, _rdm_logs = rdm.loss(batch, per_sample_ce, env_attr='env_id', approx=True)
                    loss = cau_loss + rdm_loss  # rdm already includes its weight at construction

                elif args.algo == 'urm':
                    # discriminator step(s) on current features (requires h_graph_cau from model forward)
                    z_cau = out.get('h_graph_cau', None)
                    if z_cau is None:
                        raise RuntimeError("URM requires model(...) to return 'h_graph_cau'. Patch model.py as shown.")
                    if urm is not None:
                        for _ in range(args.urm_d_steps):
                            _d_loss, _acc_u, _acc_z = urm.step_D(z_cau)
                    urm_g_loss, _ulog = urm.gen_loss(z_cau, weight=args.urm_lambda)
                    loss = cau_loss + urm_g_loss

                else:
                    raise ValueError(f"Unknown algo: {args.algo}")

            optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().cpu())
            SufLo += float((cau_loss + reg_loss).detach().cpu())
            InvLo += float(inv_loss.detach().cpu() if torch.is_tensor(inv_loss) else inv_loss)
            GraLo += float(gra_loss.detach().cpu())

            if step % show == 0:
                print("Ep:[{}/{}] TrIter:[{:<3}/{}] Lo:[{:.4f}] R:[{:.8f}]".format(
                    epoch, args.epochs, step, len(train_loader), total_loss / (step + 1), float(model.ratio))
                )

        # --- end minibatches: epoch metrics ---
        train_result = correct / len(train_loader.dataset) if args.dataset in ["motif","cmnist"] else float('nan')
        epoch_loss = total_loss / len(train_loader)
        SufLo = SufLo / max(1, len(train_loader))
        InvLo = InvLo / max(1, len(train_loader))
        GraLo = GraLo / max(1, len(train_loader))

        # eval gating
        do_eval = (epoch % args.eval_every == 0)
        if do_eval:
            if args.dataset in ["motif", "cmnist"]:
                valid_result = eval1(model, valid_loader, device)
                test_result = eval1(model, test_loader, device)
            else:
                evaluator = Evaluator(eval_name)
                valid_result = eval2(model, evaluator, valid_loader, device)[eval_metric]
                test_result = eval2(model, evaluator, test_loader, device)[eval_metric]

            # scheduler step
            if args.lr_scheduler != 'none':
                sch.step()

            # save-on-best
            if valid_result > results['highest_valid']:
                results['highest_valid'] = valid_result
                results['update_test'] = test_result
                results['update_epoch'] = epoch
                if args.save_model and trail <= 3:
                    os.makedirs(f"./model_{args.time}", exist_ok=True)
                    torch.save(model.state_dict(), f"./model_{args.time}/{args.dataset}-{args.domain}-tr{trail}-ep{epoch}.pt")
                epochs_bad = 0
            else:
                epochs_bad += 1

            # TB + CSV
            writer.add_scalar('loss/epoch', epoch_loss, epoch)
            if args.dataset in ["motif","cmnist"]:
                writer.add_scalar('acc/train', train_result, epoch)
                writer.add_scalar('acc/valid', valid_result, epoch)
                writer.add_scalar('acc/test',  test_result,  epoch)
            else:
                writer.add_scalar(f'{eval_metric}/valid', valid_result, epoch)
                writer.add_scalar(f'{eval_metric}/test',  test_result,  epoch)

            os.makedirs(os.path.dirname(args.csv_log), exist_ok=True)
            write_header = not os.path.exists(args.csv_log)
            with open(args.csv_log, 'a', newline='') as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(['time','dataset','domain','algo','seed','trail','epoch','train','valid','test','loss'])
                w.writerow([args.time, args.dataset, args.domain, args.algo, args.seed, trail, epoch,
                            f"{train_result:.6f}" if train_result==train_result else "",  # NaN-safe
                            f"{valid_result:.6f}", f"{test_result:.6f}", f"{epoch_loss:.6f}"])

            print("-" * 150)
            print("Tr:[{}/{}], Ep:[{}/{}] | Lo:[{:.4f}], SufLo:[{:.4f}], InvLo:[{:.4f}], GraLo:[{:.4f}] | "
                  "tr:[{}], va:[{:.2f}], te:[{:.2f}] | Best va:[{:.2f}], te:[{:.2f}] at:[{}] | ep time:{:.2f} min".format(
                      trail, args.trails, epoch, args.epochs,
                      epoch_loss, SufLo, InvLo, GraLo,
                      f"{train_result*100:.2f}" if train_result==train_result else "NA",
                      valid_result*100, test_result*100,
                      results['highest_valid']*100, results['update_test']*100, results['update_epoch'],
                      (time.time() - start_time_local) / 60))
            print("-" * 150)

            # early stopping
            if args.patience > 0 and epochs_bad >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break
        else:
            # still advance scheduler if set, even when we skip eval this epoch
            if args.lr_scheduler != 'none':
                sch.step()

    total_time = time.time() - start_time
    print("Best va:[{:.2f}], te:[{:.2f}] at epoch:[{}] | Total time:{}".format(
          results['highest_valid']*100, results['update_test']*100, results['update_epoch'],
          time.strftime('%H:%M:%S', time.gmtime(total_time))))
    return results['update_test']


def config_and_run(args):
    print_args(args)
    set_seed(args.seed)
    final_test_acc = []
    for trail in range(1, args.trails+1):
        args.seed += 10
        set_seed(args.seed)
        test_auc = main(args, trail)
        final_test_acc.append(test_auc)
    print("sj: finall test acc OOD: [{:.2f}±{:.2f}]".format(np.mean(final_test_acc) * 100, np.std(final_test_acc) * 100))
    print("sj: all OOD:{}\n\n".format(final_test_acc))


if __name__ == "__main__":

    def arg_parse():
        str2bool = lambda x: x.lower() == "true"
        parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
        
        parser.add_argument('--seed', type=int,   default=666)
        parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
        parser.add_argument('--data_dir', type=str, default="../dataset", help="dataset path")
        parser.add_argument('--dataset', type=str, default="hiv")
        parser.add_argument('--domain', type=str, default='color', help='basis, size, scaffold, color')
        parser.add_argument('--shift', type=str, default='covariate', help='concept or covariate')
        parser.add_argument('--save_model', type=str2bool, default='False')
        parser.add_argument('--time', type=str, default='2301121840', help='current time')
        
        parser.add_argument('--emb_dim', type=int, default=300)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--lr', type=float, default=0.0005)
        parser.add_argument('--trails', type=int, default=10, help='number of runs (default: 0)')
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--layer', type=int, default=3)
        parser.add_argument('--use_linear',type=str2bool, default=False)
        
        parser.add_argument('--virtual',type=str2bool, default=False)
        parser.add_argument('--lr_scheduler', type=str, default="cos")
        parser.add_argument('--l2reg', type=float, default=1e-6)
        parser.add_argument('--lr_decay', type=int, default=100)
        parser.add_argument('--lr_gamma', type=float, default=0.1)
        parser.add_argument('--milestones', nargs='+', type=int, default=[40,60,80])
        parser.add_argument('--dropout_rate', type=float, default=0.75)
        parser.add_argument('--cau_gamma', type=float, default=0.6)
        parser.add_argument('--env_gamma', type=float, default=1.0)
        parser.add_argument('--random_add', type=str, default='shuffle')
        parser.add_argument('--with_random', type=str2bool, default=True)
        
        parser.add_argument('--graphon', type=str2bool, default=True, help='with gra loss')
        parser.add_argument('--graphon_pretrain', type=int, default=80)
        parser.add_argument('--graphon_frequency', type=int, default=10)
        parser.add_argument('--num_env', type=int, default=5, help='env number')
        
        parser.add_argument('--cau', type=float, default=1.0, help='cau loss coefficient')
        parser.add_argument('--env', type=float, default=0, help='env loss coefficient')
        parser.add_argument('--inv', type=float, default=0.01, help='invariance loss coefficient of add env to cau')
        parser.add_argument('--gra', type=float, default=0.1, help='gra loss coefficient')
        parser.add_argument('--reg', type=float, default=0.01, help='regularization coefficient')
        
        parser.add_argument('--algo', type=str, default='uil',
                    choices=['uil','erm','ermpp','rdm','urm'],
                    help='training algorithm: UIL (default) or ERM/ERM++/RDM/URM')
        # ERM++ (lite)
        parser.add_argument('--swa', type=lambda x: x.lower()=='true', default=False)
        parser.add_argument('--swa_start', type=int, default=60)
        parser.add_argument('--swa_freq', type=int, default=5)
        parser.add_argument('--label_smoothing', type=float, default=0.0)

        # RDM
        parser.add_argument('--rdm_lambda', type=float, default=0.1)

        # URM
        parser.add_argument('--urm_lambda', type=float, default=0.05)
        parser.add_argument('--urm_d_lr', type=float, default=1e-4)
        parser.add_argument('--urm_d_steps', type=int, default=1)

        # graphon learners
        parser.add_argument('--graphon_type', type=str, default='empirical',
                            choices=['empirical','lowrank','mlp'])
        parser.add_argument('--graphon_rank', type=int, default=8)
        parser.add_argument('--graphon_coord', type=int, default=16)
        parser.add_argument('--graphon_hidden', type=int, default=64)

        # runtime
        parser.add_argument('--amp', type=lambda x: x.lower()=='true', default=False)
        parser.add_argument('--eval_every', type=int, default=1)
        parser.add_argument('--patience', type=int, default=20)

        # logging
        parser.add_argument('--logdir', type=str, default='./runs')
        parser.add_argument('--csv_log', type=str, default='./logs/metrics.csv')

        args = parser.parse_args()
        return args

    args = arg_parse()
    config_and_run(args)
