from utils.tools import *
from scipy.linalg import hadamard
from network import *
import os

import torch
import torch.optim as optim
import time
import numpy as np
import argparse
import random
import pickle


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="manual to this script")
parser.add_argument("--gpus", type=str, default="0")
parser.add_argument("--hash_dim", type=int, default=32)
parser.add_argument("--noise_rate", type=float, default=1.0)
parser.add_argument("--dataset", type=str, default="flickr")
parser.add_argument("--num_gradual", type=int, default=100)
parser.add_argument("--k", type=int, default=20)
parser.add_argument("--margin", type=float, default=0.2)
parser.add_argument("--shift", type=float, default=1.0)
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--beta", type=float, default=0.7)
parser.add_argument("--Lambda", type=float, default=0.2)

args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

bit_len = args.hash_dim
noise_rate = args.noise_rate
dataset = args.dataset
num_gradual = args.num_gradual
k = args.k
margin = args.margin
shift = args.shift
alpha = args.alpha
beta = args.beta
Lambda = args.Lambda

if dataset == "flickr":
    train_size = 10000
elif dataset == "ms-coco":
    train_size = 10000
elif dataset == "nuswide21":
    train_size = 10500
n_class = 0
tag_len = 0
torch.multiprocessing.set_sharing_strategy("file_system")


def get_config():
    config = {
        "optimizer": {
            "type": optim.RMSprop,
            "optim_params": {"lr": 1e-5, "weight_decay": 10**-5},
        },
        "txt_optimizer": {
            "type": optim.RMSprop,
            "optim_params": {"lr": 1e-5, "weight_decay": 10**-5},
        },
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "dataset": dataset,
        "epoch": 100,
        "device": torch.device("cuda:0"),
        "bit_len": bit_len,
        "noise_type": "symmetric",
        "noise_rate": noise_rate,
        "random_state": 1,
        "n_class": n_class,
        "tag_len": tag_len,
        "train_size": train_size,
        "margin": margin,
        "shift": shift,
        "alpha": alpha,
        "beta": beta,
        "Lambda": Lambda,
        "k": k,
    }
    return config


class MultiLabelLoss(nn.Module):
    def __init__(
        self,
        margin=0.2,
        lambda_contrast=0.6,
        lambda_quant=0.4,
        shift=1.0,
    ):
        super(MultiLabelLoss, self).__init__()
        self.margin = margin
        self.lambda_contrast = lambda_contrast
        self.lambda_quant = lambda_quant
        self.shift = shift

    def compute_label_similarity(self, label):
        intersection = torch.matmul(label, label.T)
        union = (
            torch.sum(label, dim=1, keepdim=True)
            + torch.sum(label, dim=1, keepdim=True).T
            - intersection
        )
        return intersection / (union + 1e-8)

    def contrast_loss(self, u, v, label, label_confidence, class_threshold):

        weighted_label = label * torch.where(
            label_confidence >= class_threshold.unsqueeze(0), 1.0, 0.0
        )

        sim_matrix = self.compute_label_similarity(weighted_label)

        batch_size = u.size(0)
        eye = torch.eye(batch_size, device=u.device)

        positive_mask = (sim_matrix > 0).float() * (1 - eye)
        negative_mask = (sim_matrix == 0).float() * (1 - eye)

        S = torch.mm(u, v.t())
        diag_s = torch.diag(S).unsqueeze(1)

        mask_te = (S >= (diag_s - self.margin)).float().detach()
        cost_te = torch.where(mask_te.bool(), S, S - self.shift)

        mask_im = (S >= (diag_s.T - self.margin)).float().detach()
        cost_im = torch.where(mask_im.bool(), S, S - self.shift)

        loss_neg = torch.mean(negative_mask * (cost_te.exp() + cost_im.exp()))

        loss_pos = torch.mean(positive_mask * (-S - sim_matrix).exp())

        return (loss_pos + loss_neg - S.diag().mean()).mean()

    def quant_loss(self, u, v, label=None, label_confidence=None):
        loss_u = torch.norm(u - u.sign(), p=1) / u.numel()
        loss_v = torch.norm(v - v.sign(), p=1) / v.numel()
        return loss_u + loss_v

    def forward(self, u, v, label, label_confidence, class_threshold):
        loss_contrast = self.contrast_loss(
            u, v, label, label_confidence, class_threshold
        )
        loss_quant = self.quant_loss(u, v)

        return self.lambda_contrast * loss_contrast + self.lambda_quant * loss_quant


def classification_loss(p_u, p_v, label, label_confidence, class_threshold):

    loss_cls_u = F.binary_cross_entropy(p_u, label_confidence)
    loss_cls_v = F.binary_cross_entropy(p_v, label_confidence)
    classification_loss = (loss_cls_u + loss_cls_v) / 2.0
    return (classification_loss).mean()


def revise_label_each_epoch(
    u, v, label, label_confidence, k, device, epoch, Lambda=0.5
):
    
    if k == 0:
        return (label_confidence * label).to(device)
    u = u.detach()
    v = v.detach()

    u_norm = torch.nn.functional.normalize(u, dim=1)
    u_sim_matrix = torch.mm(u_norm, u_norm.t()).fill_diagonal_(0)

    v_norm = torch.nn.functional.normalize(v, dim=1)
    v_sim_matrix = torch.mm(v_norm, v_norm.t()).fill_diagonal_(0)

    sim = (u_sim_matrix + v_sim_matrix) / 2

    top_sim, nearest_indices = torch.topk(sim, k=k, dim=1)
    top_sim_norm = top_sim / (top_sim.sum(dim=1, keepdim=True) + 1e-8)
    nearest_labels = label_confidence[nearest_indices] * top_sim_norm.unsqueeze(-1)
    new_label = (nearest_labels.sum(dim=1)) * label

    new_label = Lambda * new_label + (1 - Lambda) * label_confidence
    # new_label = new_label * label

    return (new_label ).to(device)


def train(config, bit, seed):

    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = (
        get_data(config)
    )
    config["num_train"] = num_train
    net = ImgModule(y_dim=4096, bit=bit, hiden_layer=3, num_classes=n_class).to(device)
    txt_net = TxtModule(y_dim=tag_len, bit=bit, hiden_layer=2, num_classes=n_class).to(
        device
    )
    W_u = torch.Tensor(bit_len, n_class)
    W_u = torch.nn.init.orthogonal_(W_u, gain=1)
    W_u = W_u.clone().detach().requires_grad_(True).to(device)
    W_u = torch.nn.Parameter(W_u)
    net.register_parameter("W_u", W_u)  

    W_v = torch.Tensor(bit_len, n_class)
    W_v = torch.nn.init.orthogonal_(W_v, gain=1)
    W_v = W_v.clone().detach().requires_grad_(True).to(device)
    W_v = torch.nn.Parameter(W_v)
    txt_net.register_parameter("W_v", W_v)  
    get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
    params_dnet = get_grad_params(net)
    optimizer = config["optimizer"]["type"](
        params_dnet, **(config["optimizer"]["optim_params"])
    )
    txt_optimizer = config["txt_optimizer"]["type"](
        txt_net.parameters(), **(config["txt_optimizer"]["optim_params"])
    )

    criterion = MultiLabelLoss(
        shift=config["shift"],
        margin=config["margin"],
        lambda_contrast=config["alpha"],
        lambda_quant=config["beta"],
    )

    i2t_mAP_list = []
    t2i_mAP_list = []

    epoch_list = []
    bestt2i = 0
    besti2t = 0

    os.makedirs("./NACD/checkpoint", exist_ok=True)
    os.makedirs("./NACD/logs123_lambda0.2_0.2_1.0_0.5_0.5", exist_ok=True)
    os.makedirs("./NACD/PR", exist_ok=True)
    os.makedirs("./NACD/map", exist_ok=True)

    all_indexs = []
    all_labels = []
    with torch.no_grad():
        for i, (image, tag, tlabel, label, ind) in enumerate(train_loader):
            label = label.to(device)
            all_indexs.append(ind)
            all_labels.append(label)
    all_labels = torch.cat(all_labels, dim=0)
    all_indexs = torch.cat(all_indexs, dim=0)
    sorted_indices = torch.argsort(all_indexs)
    revise_label_confidence = all_labels[sorted_indices].float()

    class_threshold = torch.ones(n_class).float().to(device) * 0.5

    with open(
        "./NACD/logs123_lambda0.2_0.2_1.0_0.5_0.5/data_{}_seed_{}_noiseRate_{}_bit_{}.txt".format(
            config["dataset"],
            seed,
            config["noise_rate"],
            bit,
        ),
        "w",
    ) as f:
        for epoch in range(config["epoch"]):
            current_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
            print(
                "%s[%2d/%2d][%s] bit:%d, dataset:%s, training...."
                % (
                    config["info"],
                    epoch + 1,
                    config["epoch"],
                    current_time,
                    bit,
                    config["dataset"],
                ),
                end="",
            )
            net.eval()
            txt_net.eval()

            train_loss = 0
            if (epoch + 1) % 20 == 0:
                print("calculating test binary code......")
                img_tst_binary, img_tst_label = compute_img_result(
                    test_loader, net, device=device
                )
                print("calculating dataset binary code.......")
                img_trn_binary, img_trn_label = compute_img_result(
                    dataset_loader, net, device=device
                )
                txt_tst_binary, txt_tst_label = compute_tag_result(
                    test_loader, txt_net, device=device
                )
                txt_trn_binary, txt_trn_label = compute_tag_result(
                    dataset_loader, txt_net, device=device
                )
                print("calculating map.......")
                t2i_mAP = calc_map_k(
                    img_trn_binary.numpy(),
                    txt_tst_binary.numpy(),
                    img_trn_label.numpy(),
                    txt_tst_label.numpy(),
                    device=device,
                )

                i2t_mAP = calc_map_k(
                    txt_trn_binary.numpy(),
                    img_tst_binary.numpy(),
                    txt_trn_label.numpy(),
                    img_tst_label.numpy(),
                    device=device,
                )

                if t2i_mAP + i2t_mAP > bestt2i + besti2t:
                    bestt2i = t2i_mAP
                    besti2t = i2t_mAP

                t2i_mAP_list.append(t2i_mAP.item())
                i2t_mAP_list.append(i2t_mAP.item())
                epoch_list.append(epoch)
                print(
                    "%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f,t2i_mAP:%.3f, i2t_mAP:%.3f \n"
                    % (
                        config["info"],
                        epoch + 1,
                        bit,
                        config["dataset"],
                        config["noise_rate"],
                        t2i_mAP,
                        i2t_mAP,
                    )
                )
                f.writelines(
                    "%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f,t2i_mAP:%.3f, i2t_mAP:%.3f\n"
                    % (
                        config["info"],
                        epoch + 1,
                        bit,
                        config["dataset"],
                        config["noise_rate"],
                        t2i_mAP,
                        i2t_mAP,
                    )
                )

            net.train()
            txt_net.train()
            cr_loss = 0
            cl_loss = 0
            for i, (image, tag, tlabel, label, ind) in enumerate(train_loader):
                image = image.float().to(device)
                tag = tag.float().to(device)
                label = label.to(device)
                optimizer.zero_grad()
                txt_optimizer.zero_grad()
                u = net(image)
                v = txt_net(tag)

                p_u = torch.sigmoid(torch.matmul(u, W_u))  
                p_v = torch.sigmoid(torch.matmul(v, W_v))  

                cls_loss = classification_loss(
                    p_u, p_v, label, revise_label_confidence[ind], class_threshold
                )

                cross_loss = criterion(
                    u, v, label, revise_label_confidence[ind], class_threshold
                )

                pseudo_label_weight = (
                    1.0 * epoch / config["epoch"] * (0.8 - 0.95) + 0.95
                )
                
                revise_label_confidence[ind] = (
                    (pseudo_label_weight) * revise_label_confidence[ind] + (1 - pseudo_label_weight) * (p_u.detach() + p_v.detach()) / 2
                ) * label

                loss = cls_loss + cross_loss
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                txt_optimizer.step()
            
            print("loss：{}".format(train_loss / len(train_loader)))

            if epoch >= 10:
                img_feature = []
                tag_feature = []
                all_indexs = []
                all_labels = []
                all_prob = []
                for i, (image, tag, tlabel, label, ind) in enumerate(train_loader):
                    image = image.float().to(device)
                    tag = tag.float().to(device)
                    label = label.to(device)
                    optimizer.zero_grad()
                    txt_optimizer.zero_grad()
                    u = net(image)
                    v = txt_net(tag)

                    img_feature.append(u)
                    tag_feature.append(v)
                    all_indexs.append(ind)
                    all_labels.append(label)

                sorted_indices = torch.argsort(torch.cat(all_indexs))
                u_fea = torch.cat(img_feature)[sorted_indices]
                v_fea = torch.cat(tag_feature)[sorted_indices]
                all_labels = torch.cat(all_labels)[sorted_indices]

                revise_label_confidence = revise_label_each_epoch(
                    u_fea,
                    v_fea,
                    all_labels,
                    revise_label_confidence,
                    config["k"],
                    device,
                    epoch,
                    config["Lambda"],
                )

                mask = revise_label_confidence > 0
                class_sums = torch.sum(revise_label_confidence * mask, dim=0)
                class_counts = torch.sum(mask, dim=0)

                class_counts = torch.where(
                    class_counts == 0, torch.tensor(1.0, device=device), class_counts
                )

                class_threshold = class_sums / class_counts
        f.writelines(
            f"best result : bit:{bit}, dataset:{config['dataset']}, noise_rate:{config['noise_rate']:.1f}, t2i_mAP:{bestt2i:.3f}, i2t_mAP:{besti2t:.3f}, average:{(besti2t + bestt2i) / 2.0 * 100.0:.1f}\n"
        )

def test(config, bit, model_path="./NACD/checkpoint/best_model.pth"):
    device = config["device"]
    _, test_loader, dataset_loader, _, _, _ = get_data(config)
    net = ImgModule(y_dim=4096, bit=bit, hiden_layer=3).to(device)
    txt_net = TxtModule(y_dim=tag_len, bit=bit, hiden_layer=2).to(device)
    W = torch.Tensor(n_class, bit_len)
    W = torch.nn.init.orthogonal_(W, gain=1)
    W = torch.tensor(W, requires_grad=True).to(device)
    W = torch.nn.Parameter(W)
    net.register_parameter("W", W)
    
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint["net_state_dict"])
    txt_net.load_state_dict(checkpoint["txt_net_state_dict"])
    net.eval()
    txt_net.eval()
    print("calculating test binary code......")
    print("calculating test binary code......")
    img_tst_binary, img_tst_label = compute_img_result(test_loader, net, device=device)
    print("calculating dataset binary code.......")
    img_trn_binary, img_trn_label = compute_img_result(
        dataset_loader, net, device=device
    )
    txt_tst_binary, txt_tst_label = compute_tag_result(
        test_loader, txt_net, device=device
    )
    txt_trn_binary, txt_trn_label = compute_tag_result(
        dataset_loader, txt_net, device=device
    )
    print("calculating map.......")
    t2i_mAP = calc_map_k(
        img_trn_binary.numpy(),
        txt_tst_binary.numpy(),
        img_trn_label.numpy(),
        txt_tst_label.numpy(),
        device=device,
    )
    i2t_mAP = calc_map_k(
        txt_trn_binary.numpy(),
        img_tst_binary.numpy(),
        txt_trn_label.numpy(),
        img_tst_label.numpy(),
        device=device,
    )
    print("Test Results: t2i_mAP: %.3f, i2t_mAP: %.3f" % (t2i_mAP, i2t_mAP))


if __name__ == "__main__":
    data_name_list = ["flickr"]
    bit_list = [32,64,128]
    noise_rate_list = [1.0,1.5,2.0,2.5]
    for rand_num in [123]:
        for data_name in data_name_list:
            for rate in noise_rate_list:
                for bit in bit_list:
                    setup_seed(rand_num)
                    bit_len = bit
                    noise_rate = rate
                    dataset = data_name
                    if dataset == "nuswide21":
                        n_class = 21
                        tag_len = 1000
                        k = 20
                        margin = 0.2
                        shift = 1.0
                    elif dataset == "flickr":
                        n_class = 24
                        tag_len = 1386
                        k = 20
                        margin = 0.2
                        shift = 1.0

                    elif dataset == "ms-coco":
                        n_class = 80
                        tag_len = 300
                        k = 20
                        margin = 0.2
                        shift = 1.0

                    config = get_config()
                    print(config)
                    train(config, bit, rand_num)
                    # test(config, bit)
