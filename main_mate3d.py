import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms as T
from scipy.stats import pearsonr, spearmanr, kendalltau
from torch.utils.data import DataLoader

from QAdataset.Mate3D import Mate3DData, QACollator
from model.vit_w_clip import ViTWClip

from logger import Logger


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(model, loader, val_loader, optimizer, lr_scheduler, max_epoch, logger=None):

    model.train()

    best_metric = 0
    best_res = None

    for epoch in range(1, max_epoch+1):

        total_loss = 0

        tqdm_item =  tqdm(enumerate(loader), total=len(loader), leave=False)
        for i, batch in tqdm_item:
            optimizer.zero_grad()

            if len(batch) == 3:
                model_images, prompt, mos = batch
            elif len(batch) == 4:
                model_images, model_d_images, prompt, mos = batch
                model_d_images = model_d_images.cuda()
            elif len(batch) == 5:
                model_images, model_d_images, order, prompt, mos = batch
                model_d_images = model_d_images.cuda()
                order = order.cuda()
            elif len(batch) == 6:
                model_images, model_d_images, order, prompt, prompt_zh, mos = batch
                prompt = prompt.cuda()
                prompt_zh = prompt_zh
                model_d_images = model_d_images.cuda()
                order = order.cuda()
            model_images = model_images.cuda()
            mos = mos.float().cuda()

            # pred, cos_loss = model(model_images, model_d_images, prompt)

            pred, loss_cos = model(model_images, model_d_images, order, prompt, prompt_zh)

            loss = F.mse_loss(pred, mos) + loss_cos
            total_loss += loss.detach().item()
            avg_loss = total_loss / (1+i)

            tqdm_item.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.7f}"})

            loss.backward()
            optimizer.step()

        logger.log(f"Epoch {epoch}: Loss: {avg_loss}")
        lr_scheduler.step()

        metric, result = test(model, val_loader, epoch, logger=logger)

        if metric>best_metric:
            logger.log(f"Best Epoch: {epoch}")
            best_res = result
            best_metric = metric

    return best_res



def test(model, loader, epoch=None, logger=None):
    model.eval()

    geo_gt = np.array([])
    texture_gt = np.array([])
    align_gt = np.array([])
    overall_gt = np.array([])

    geo_preds = np.array([])
    texture_preds = np.array([])
    align_preds = np.array([])
    overall_preds = np.array([])

    for i, batch in enumerate(loader):
        if len(batch) == 3:
                model_images, prompt, mos = batch
        elif len(batch) == 4:
            model_images, model_d_images, prompt, mos = batch
            model_d_images = model_d_images.cuda()
        elif len(batch) == 5:
            model_images, model_d_images, order, prompt, mos = batch
            model_d_images = model_d_images.cuda()
            order = order.cuda()
        elif len(batch) == 6:
            model_images, model_d_images, order, prompt, prompt_zh, mos = batch
            prompt = prompt.cuda()
            prompt_zh = prompt_zh
            model_d_images = model_d_images.cuda()
            order = order.cuda()
        model_images = model_images.cuda()

        with torch.no_grad():
            # pred, _ = model(model_images, model_d_images, order, prompt)
            pred, _ = model(model_images, model_d_images, order, prompt, prompt_zh)
            pred = pred.cpu().numpy()

        mos = mos.numpy()

        geo_preds = np.append(geo_preds, pred[:,0])
        texture_preds = np.append(texture_preds, pred[:,1])
        align_preds = np.append(align_preds, pred[:,2])
        overall_preds = np.append(overall_preds, pred[:,3])
        geo_gt = np.append(geo_gt, mos[:,0])
        texture_gt = np.append(texture_gt, mos[:,1])
        align_gt = np.append(align_gt, mos[:,2])
        overall_gt = np.append(overall_gt, mos[:,3])

    geo_plcc, geo_srcc, geo_krcc, geo_rmse = pearsonr(geo_preds, geo_gt)[0], spearmanr(geo_preds, geo_gt)[0], kendalltau(geo_preds, geo_gt)[0], np.mean((geo_preds - geo_gt)**2)
    texture_plcc, texture_srcc, texture_krcc, texture_rmse = pearsonr(texture_preds, texture_gt)[0], spearmanr(texture_preds, texture_gt)[0], kendalltau(texture_preds, texture_gt)[0], np.mean((texture_preds - texture_gt)**2)
    align_plcc, align_srcc, align_krcc, align_rmse = pearsonr(align_preds, align_gt)[0], spearmanr(align_preds, align_gt)[0], kendalltau(align_preds, align_gt)[0], np.mean((align_preds - align_gt)**2)
    overall_plcc, overall_srcc, overall_krcc, overall_rmse = pearsonr(overall_preds, overall_gt)[0], spearmanr(overall_preds, overall_gt)[0], kendalltau(overall_preds, overall_gt)[0], np.mean((overall_preds - overall_gt)**2)

    if epoch is not None:
        logger.log(f"Epoch {epoch}: Geo PLCC: {geo_plcc:.4f}, SRCC: {geo_srcc:.4f}, KRCC: {geo_krcc:.4f}, RMSE: {geo_rmse:.4f}")
        logger.log(f"Epoch {epoch}: Texture PLCC: {texture_plcc:.4f}, SRCC: {texture_srcc:.4f}, KRCC: {texture_krcc:.4f}, RMSE: {texture_rmse:.4f}")
        logger.log(f"Epoch {epoch}: Align PLCC: {align_plcc:.4f}, SRCC: {align_srcc:.4f}, KRCC: {align_krcc:.4f}, RMSE: {align_rmse:.4f}")
        logger.log(f"Epoch {epoch}: Overall PLCC: {overall_plcc:.4f}, SRCC: {overall_srcc:.4f}, KRCC: {overall_krcc:.4f}, RMSE: {overall_rmse:.4f}")

    if epoch is not None:
        model.train()

    result = {
        "Geo": {"plcc": geo_plcc, "srcc": geo_srcc, "krcc": geo_krcc, "rmse": geo_rmse},
        "Texture": {"plcc": texture_plcc, "srcc": texture_srcc, "krcc": texture_krcc, "rmse": texture_rmse},
        "Align": {"plcc": align_plcc, "srcc": align_srcc, "krcc": align_krcc, "rmse": align_rmse},
        "Overall": {"plcc": overall_plcc, "srcc": overall_srcc, "krcc": overall_krcc, "rmse": overall_rmse}
    }

    return geo_srcc + texture_srcc + align_srcc + overall_srcc, result


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/mate3d")
    parser.add_argument("--info_dir", type=str, default="data/csvfiles")
    parser.add_argument("--csv_file", type=str, default="12345")
    parser.add_argument("--msg", type=str, default="train")

    args = parser.parse_args()

    seed = args.seed
    seed_everything(seed)

    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    logger = Logger(
        f"logs/log_mate3d"
    )

    logger.log(f"CUDA_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}", verbos=False)
    logger.log(f"seed: {seed}", verbos=False)
    logger.log(f"Message: {args.msg}", verbos=False)

    results = []
    for csv_file in [int(i) for i in args.csv_file]:

        train_info = pd.read_csv(os.path.join(args.info_dir, f"train_{csv_file}.csv")).to_dict("records")
        test_info = pd.read_csv(os.path.join(args.info_dir, f"test_{csv_file}.csv")).to_dict("records")

        train_dataset = Mate3DData(train_info, data_dir=args.data_dir, transforms=transforms, shuffle=True)
        val_dataset = Mate3DData(test_info, data_dir=args.data_dir, transforms=transforms)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=6, collate_fn=QACollator())
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=6, collate_fn=QACollator())

        model = ViTWClip().cuda()

        encoder_params = []
        other_params = []
        # 遍历所有参数
        for name, p in model.named_parameters():
            if name.startswith("visual_encoder") or name.startswith("depth_encoder"):  # 假设编码器参数名前缀为 "encoder."
                encoder_params.append(p)
            elif p.requires_grad:
                other_params.append(p)

        optimizer = torch.optim.AdamW(params=[
            {"params": encoder_params,"lr": 2e-6,},
            {"params": other_params, "lr": 2e-4,},
        ], weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        logger.log(f"processing csv: {csv_file}")
        result = train(model, train_loader, val_loader, optimizer, lr_scheduler, max_epoch=30, logger=logger)
        results.append(result)


    for t in ["Geo", "Texture", "Align", "Overall"]:
        r = []
        for res in results:
            res_t = res[t]
            r.append([res_t["plcc"], res_t["srcc"], res_t["krcc"], res_t["rmse"]])

        r = np.asarray(r).mean(axis=0)

        logger.log(f"{t}: {r}")
