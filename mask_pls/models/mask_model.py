import mask_pls.utils.testing as testing
import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
from mask_pls.models.decoder import MaskedTransformerDecoder
from mask_pls.models.loss import MaskLoss, SemLoss
from mask_pls.models.mink import MinkEncoderDecoder
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator
from pytorch_lightning.core.lightning import LightningModule


class MaskPS(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(dict(hparams))
        self.cfg = hparams

        backbone = MinkEncoderDecoder(hparams.BACKBONE, hparams[hparams.MODEL.DATASET])
        self.backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(backbone)

        self.decoder = MaskedTransformerDecoder(
            hparams.DECODER,
            hparams.BACKBONE,
            hparams[hparams.MODEL.DATASET],
        )

        self.mask_loss = MaskLoss(hparams.LOSS, hparams[hparams.MODEL.DATASET])
        self.sem_loss = SemLoss(hparams.LOSS.SEM.WEIGHTS)

        self.evaluator = PanopticEvaluator(
            hparams[hparams.MODEL.DATASET], hparams.MODEL.DATASET
        )

    def forward(self, x):
        feats, coors, pad_masks, bb_logits = self.backbone(x)
        outputs, padding = self.decoder(feats, coors, pad_masks)
        return outputs, padding, bb_logits

    def getLoss(self, x, outputs, padding, bb_logits):
        targets = {"classes": x["masks_cls"], "masks": x["masks"]}
        loss_mask = self.mask_loss(outputs, targets, x["masks_ids"], x["pt_coord"])
        sem_labels = [
            torch.from_numpy(i).type(torch.LongTensor).cuda() for i in x["sem_label"]
        ]
        sem_labels = torch.cat([s.squeeze(1) for s in sem_labels], dim=0)
        bb_logits = bb_logits[~padding]
        loss_sem_bb = self.sem_loss(bb_logits, sem_labels)
        loss_mask.update(loss_sem_bb)

        return loss_mask

    def training_step(self, x: dict, idx):
        outputs, padding, bb_logits = self.forward(x)
        loss_dict = self.getLoss(x, outputs, padding, bb_logits)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        total_loss = sum(loss_dict.values())
        self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        torch.cuda.empty_cache()

        return total_loss

    def validation_step(self, x: dict, idx):
        if "EVALUATE" in self.cfg:
            self.evaluation_step(x, idx)
            return
        outputs, padding, bb_logits = self.forward(x)
        loss_dict = self.getLoss(x, outputs, padding, bb_logits)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        total_loss = sum(loss_dict.values())
        self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)

        sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
        self.evaluator.update(sem_pred, ins_pred, x)

        torch.cuda.empty_cache()
        return total_loss

    def validation_epoch_end(self, outputs):
        bs = self.cfg.TRAIN.BATCH_SIZE
        self.log("metrics/pq", self.evaluator.get_mean_pq(), batch_size=bs)
        self.log("metrics/iou", self.evaluator.get_mean_iou(), batch_size=bs)
        self.log("metrics/rq", self.evaluator.get_mean_rq(), batch_size=bs)
        if not "EVALUATE" in self.cfg:
            self.evaluator.reset()

    def evaluation_step(self, x: dict, idx):
        outputs, padding, bb_logits = self.forward(x)
        sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
        self.evaluator.update(sem_pred, ins_pred, x)

    def test_step(self, x: dict, idx):
        outputs, padding, bb_logits = self.forward(x)
        sem_pred, ins_pred = self.panoptic_inference(outputs, padding)

        if "RESULTS_DIR" in self.cfg:
            results_dir = self.cfg.RESULTS_DIR
            class_inv_lut = self.evaluator.get_class_inv_lut()
            dt = self.cfg.MODEL.DATASET
            testing.save_results(
                sem_pred, ins_pred, results_dir, x, class_inv_lut, x["token"], dt
            )
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.TRAIN.STEP, gamma=self.cfg.TRAIN.DECAY
        )
        return [optimizer], [scheduler]

    def semantic_inference(self, outputs, padding):
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        semseg = []
        for mask_cls, mask_pred, pad in zip(mask_cls, mask_pred, padding):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred[~pad].sigmoid()  # throw padding points
            pred = torch.einsum("qc,pq->pc", mask_cls, mask_pred)
            semseg.append(torch.argmax(pred, dim=1))
        return semseg

    def panoptic_inference(self, outputs, padding):
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        things_ids = self.trainer.datamodule.things_ids
        num_classes = self.cfg[self.cfg.MODEL.DATASET].NUM_CLASSES
        sem_pred = []
        ins_pred = []
        panoptic_output = []
        info = []
        for mask_cls, mask_pred, pad in zip(mask_cls, mask_pred, padding):
            scores, labels = mask_cls.max(-1)
            mask_pred = mask_pred[~pad].sigmoid()
            keep = labels.ne(num_classes)

            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[:, keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            # prob to belong to each of the `keep` masks for each point
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks

            panoptic_seg = torch.zeros(
                (cur_masks.shape[0]), dtype=torch.int32, device=cur_masks.device
            )
            sem = torch.zeros_like(panoptic_seg)
            ins = torch.zeros_like(panoptic_seg)
            segments_info = []
            masks = []
            segment_id = 0
            if cur_masks.shape[1] == 0:  # no masks detected
                panoptic_output.append(panoptic_seg)
                info.append(segments_info)
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
            else:
                # mask index for each point: between 0 and (`keep` - 1)
                cur_mask_ids = cur_prob_masks.argmax(1)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()  # current class
                    isthing = pred_class in things_ids
                    mask_area = (cur_mask_ids == k).sum().item()  # points in mask k
                    original_area = (cur_masks[:, k] >= 0.5).sum().item()  # binary mas
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.cfg.MODEL.OVERLAP_THRESHOLD:
                            continue  # binary mask occluded 80%
                        if not isthing:  # merge stuff regions
                            if int(pred_class) in stuff_memory_list.keys():
                                # in the list, asign id stored on the list for that class
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                # not in the list, class = cur_id + 1
                                stuff_memory_list[int(pred_class)] = segment_id + 1
                        segment_id += 1
                        panoptic_seg[mask] = segment_id
                        masks.append(mask)
                        # indice which class each segment id has
                        segments_info.append(
                            {
                                "id": segment_id,
                                "isthing": bool(isthing),
                                "sem_class": int(pred_class),
                            }
                        )
                panoptic_output.append(panoptic_seg)
                info.append(segments_info)
                for mask, inf in zip(masks, segments_info):
                    sem[mask] = inf["sem_class"]
                    if inf["isthing"]:
                        ins[mask] = inf["id"]
                    else:
                        ins[mask] = 0
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())

        return sem_pred, ins_pred
