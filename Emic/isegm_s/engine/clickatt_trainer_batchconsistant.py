import os
import random
import logging
from copy import deepcopy
from collections import defaultdict

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from isegm.data.points_sampler import generate_probs
from isegm.utils.log import get_root_logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points, add_tag
from isegm.utils.misc import save_checkpoint, get_labels_with_sizes
from isegm.utils.serialization import get_config_repr
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from .optimizer import get_optimizer
from torch.cuda.amp import autocast as autocast, GradScaler
try:
    import FastGeodis
    USE_FASTGEODIS=True
except:
    USE_FASTGEODIS=False

scaler = GradScaler()

class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer='adam',
                 optimizer_params=None,
                 image_dump_interval=200,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 net_inputs=('images', 'points'),
                 max_num_next_clicks=0,
                 click_models=None,
                 prev_mask_drop_prob=0.0,
                 sample_prob_sigma=0.8,
                 ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks

        self.click_models = click_models
        self.prev_mask_drop_prob = prev_mask_drop_prob

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None
        self.use_fp16 = cfg.fp16
        self.trainset = trainset
        self.valset = valset
        self.logger = get_root_logger()
        self.logger.info(f'Dataset of {trainset.get_samples_number()} samples was loaded for training.')
        self.logger.info(f'Dataset of {valset.get_samples_number()} samples was loaded for validation.')

        self.train_data = DataLoader(
            trainset, cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        self.val_data = DataLoader(
            valset, cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        self.optim = get_optimizer(model, optimizer, optimizer_params)
        #model = self._load_weights(model)

        if cfg.multi_gpu:
            if cfg.distributed:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = get_dp_wrapper(cfg.distributed)(model, device_ids=cfg.gpu_ids,
                                                    output_device=cfg.gpu_ids[0], find_unused_parameters=True)
            else:
                model = get_dp_wrapper(cfg.distributed)(model, device_ids=cfg.gpu_ids,
                                                    output_device=cfg.gpu_ids[0])
        if self.is_master:
            self.logger.info(model)
            #找不到
            #self.logger.info(get_config_repr(model._config))

        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(self.logger, level=logging.INFO)

        if self.click_models is not None:
            for click_model in self.click_models:
                for param in click_model.parameters():
                    param.requires_grad = False
                click_model.to(self.device)
                click_model.eval()

        self.sample_probs = generate_probs(self.max_num_next_clicks, sample_prob_sigma)


    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        self.logger.info(f'Starting Epoch: {start_epoch}')
        self.logger.info(f'Total Epochs: {num_epochs}')
        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            #if validation:
            #    self.validation(epoch)

    def training(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)\
            if self.is_master else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        #for m in self.net.feature_extractor.modules():
        #        m.eval()

        train_loss = 0.0
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i
            
            if self.use_fp16:
                self.optim.zero_grad()
                loss, losses_logging, splitted_batch_data, outputs = self.batch_forward(batch_data)
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
            else:
                loss, losses_logging, splitted_batch_data, outputs = self.batch_forward(batch_data)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            losses_logging['overall'] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                       value=loss_value.item(),
                                       global_step=global_step)

                for k, v in self.loss_cfg.items():
                    if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                        v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

                if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                    self.save_visualization(splitted_batch_data, outputs,global_step, prefix='train')

                self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                                   value=self.lr if not hasattr(self, 'lr_scheduler') else self.lr_scheduler.get_lr()[-1],
                                   global_step=global_step)

                tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.4f}')
                for metric in self.train_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for metric in self.train_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                                   value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                            epoch=None, multi_gpu=self.cfg.multi_gpu)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval

            if epoch % checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100) if self.is_master else self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            loss, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, validation=True)

            batch_losses_logging['overall'] = loss
            reduce_loss_dict(batch_losses_logging)
            for loss_name, loss_value in batch_losses_logging.items():
                losses_logging[loss_name].append(loss_value.item())

            val_loss += batch_losses_logging['overall'].item()

            if self.is_master:
                tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f}')
                for metric in self.val_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=epoch, disable_avg=True)

            for metric in self.val_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()} # 'images' 'instances(mask)' 'scri'
            image, gt_mask, scrib = batch_data['images'], batch_data['instances'], batch_data['scri']
            orig_image, orig_gt_mask, orig_scrib = image.clone(), gt_mask.clone(), scrib.clone() #把一張圖片的所有mask都當1合併成一張
            
            batch_size = image.shape[0]
            with torch.no_grad():
                num_iters = np.random.choice(np.arange(self.max_num_next_clicks), p=self.sample_probs)
                
                positive_click_num_array = np.zeros(batch_size, dtype=int)
                negative_click_num_array = np.zeros(batch_size, dtype=int)
                
                for click_indx in range(num_iters):

                    if not validation:
                        self.net.eval()

                    if self.click_models is None or click_indx >= len(self.click_models):
                        eval_model = self.net
                    else:
                        eval_model = self.click_models[click_indx]

                    #refined_output = eval_model(image, scrib)
                    
                    refined_output = eval_model(image, scrib)
                    prev_output = torch.sigmoid(refined_output['final_mask'])

                    ## 改
                    scrib = get_rough_next_scrib(prev_output, orig_gt_mask, scrib, positive_click_num_array, negative_click_num_array,click_indx)
                    
                    if not validation:
                        self.net.train()

            #batch_data['points'] = points
            #batch_data['prev_mask'] = prev_output
            
            refined_output = self.net(image, scrib)
            self.net.model_init()
            loss = 0.0
            loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                 lambda: (refined_output['adp_mask'], batch_data['instances']))
            
            # loss = self.add_loss('instance_click_loss', loss, losses_logging, validation,
            #                      lambda: (output['instances'], batch_data['instances'], output['click_map']))
                                 
            loss = self.add_loss('instance_aux_loss', loss, losses_logging, validation,
                                 lambda: (refined_output['instances_aux'], batch_data['instances']))
                
            loss = self.add_loss('instance_refine_loss', loss, losses_logging, validation,
                                 lambda: (refined_output['final_mask'], batch_data['instances']))

            if self.is_master:
                pass
                # with torch.no_grad():
                #     for m in metrics:
                #         m.update(*(output.get(x) for x in m.pred_outputs),
                #                  *(batch_data[x] for x in m.gt_outputs))
        return loss, losses_logging, batch_data, refined_output

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            losses_logging[loss_name] = loss
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        #points = splitted_batch_data['points'][0].detach().cpu().numpy()
        images = splitted_batch_data['images'][0].cpu().numpy().transpose((1, 2, 0)) * 255
        #image_with_points = draw_points(images, points[:self.max_interactive_points], (0, 255, 0))
        #image_with_points = draw_points(image_with_points, points[self.max_interactive_points:], (0, 0, 255))
        
        instance_masks = splitted_batch_data['instances'][0,0].detach().cpu().numpy()
        gt_mask = draw_probmap(instance_masks)
        gt_color_mask = np.zeros_like(images)
        gt_color_mask[:,:,0] = (instance_masks > 0.5) * 255
        gt_masked_image = 0.5 * images + 0.5 * gt_color_mask #vis

        # prev_masks = splitted_batch_data['prev_mask'][0,0].detach().cpu().numpy()
        # prev_masks = draw_probmap(prev_masks)

        
        predicted_instance_masks = torch.sigmoid(outputs['adp_mask'])[0,0].detach().cpu().numpy()
        pred_color_mask = np.zeros_like(images)
        pred_color_mask[:,:,0] = (predicted_instance_masks > 0.5) * 255
        #pred_color_mask[:,:,0] = (predicted_instance_masks) * 255
        pred_masked_image = 0.5 * images + 0.5 * pred_color_mask
        predicted_instance_masks = draw_probmap(predicted_instance_masks)

        predicted_aux_masks = torch.sigmoid(outputs['final_mask'])[0,0].detach().cpu().numpy()
        pred_color_mask_aux = np.zeros_like(images)
        #pred_color_mask_fdm[:,:,0] = (predicted_fdm_masks > 0.5) * 255
        pred_color_mask_aux[:,:,0] = (predicted_aux_masks) * 255
        pred_aux_image = 0.5 * images + 0.5 * pred_color_mask_aux
        predicted_aux_masks = draw_probmap(predicted_aux_masks)

        #image_with_points = add_tag(image_with_points,'Image with Points') 改
        image_with_points = add_tag(images,'Image') 
        gt_masked_image = add_tag(gt_masked_image,'Ground Truth Mask')
        pred_masked_image = add_tag(pred_masked_image,'Prediction')
        pred_aux_image = add_tag(pred_aux_image,'Aux Prediction')

        viz_image_full = np.hstack((image_with_points, gt_masked_image, pred_masked_image, pred_aux_image)).astype(np.uint8)
        #viz_image_full = np.hstack((image_with_points, gt_masked_image, pred_masked_image, pred_aux_image)).astype(np.uint8)
        viz_image = viz_image_full
        
        _save_image('iseg_baseline_vis', viz_image[:, :, ::-1])

    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            self.logger.info(f'Load checkpoint from path: {checkpoint_path}')
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return self.cfg.rank == 0

def get_rough_next_points(pred, gt, points, positive_click_num_array, negative_click_num_array, pred_thresh=0.49):
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8) #pad一圈0
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)
        if fn_max_dist + fp_max_dist == 0: #in case the pred is perfect
            continue
        else:
            is_positive = np.random.choice(np.array([0, 1]), p=np.array([fp_max_dist, fn_max_dist])/(fn_max_dist + fp_max_dist))
        
        if positive_click_num_array[bindx] >= num_points-1:
            is_positive = False
        if negative_click_num_array[bindx] >= num_points:
            is_positive = True
        if positive_click_num_array[bindx] >= num_points-1 and negative_click_num_array[bindx] >= num_points:
            continue

        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask) #傳回符合條件的row col 位置
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))] #隨機選擇滿足條件的一點
            if is_positive:
                positive_click_num_array[bindx] += 1
                points[bindx, num_points - positive_click_num_array[bindx], 0] = float(coords[0])
                points[bindx, num_points - positive_click_num_array[bindx], 1] = float(coords[1])
                points[bindx, num_points - positive_click_num_array[bindx], 2] = float(positive_click_num_array[bindx])
                
            else:
                negative_click_num_array[bindx] += 1
                points[bindx, 2 * num_points - negative_click_num_array[bindx], 0] = float(coords[0])
                points[bindx, 2 * num_points - negative_click_num_array[bindx], 1] = float(coords[1])
                points[bindx, 2 * num_points - negative_click_num_array[bindx], 2] = float(negative_click_num_array[bindx])
                

    return points

def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']

    new_keys = set(new_state_dict.keys())
    old_keys = set(current_state_dict.keys())
    print('='*10)
    print(' unexpected: ', new_keys - old_keys )
    print(' lacked: ', old_keys - new_keys )
    print('='*10)
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict, strict=False)
    
    
#################################################################### NEW ADD
def bound_scri(mask):
    temp=np.array(mask)
    kernel = np.ones((10, 10), np.uint8)
    eroded_mask = cv2.erode((temp*255).astype(np.uint8) , kernel, iterations=1)
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_inside_mask = np.zeros_like(temp)
    p=random.randint(1,3)
    boundary_inside_mask=cv2.drawContours(boundary_inside_mask, contours, -1, (255), thickness=p)
    
    return boundary_inside_mask

def rand_scri(mask):
    temp=np.zeros_like(mask)

    l=np.argwhere(mask)
    l=l[:, ::-1]
    len_l=len(l)
    start_point = tuple(l[random.randint(0,len_l-1)])
    end_point = tuple(l[random.randint(0,len_l-1)])
    p=random.randint(1,3)
    cv2.line(temp, start_point, end_point, 1, thickness=p)
    
    return temp

import random
def get_rough_next_scrib(pred, gt, scrib, positive_click_num_array, negative_click_num_array,click_indx,pred_thresh=0.49):
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8) #pad一圈0
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = int(click_indx+1)
    scrib = scrib.clone().detach().cpu()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)
        if fn_max_dist + fp_max_dist == 0: #in case the pred is perfect
            continue
        else:
            is_positive = np.random.choice(np.array([0, 1]), p=np.array([fp_max_dist, fn_max_dist])/(fn_max_dist + fp_max_dist))
        
        if positive_click_num_array[bindx] >= num_points-1:
            is_positive = False
        if negative_click_num_array[bindx] >= num_points:
            is_positive = True
        if positive_click_num_array[bindx] >= num_points-1 and negative_click_num_array[bindx] >= num_points:
            continue
        
        
        dt = fn_mask_dt if is_positive else fp_mask_dt #(H,W)
        inner_mask=np.zeros_like(dt)
        inner_mask[dt > max(fn_max_dist, fp_max_dist) / 2.0] = 1
        inner_mask=inner_mask.astype(np.uint8)
        if np.sum(inner_mask) > 0:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inner_mask, connectivity=8)
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

            largest_connected_region = np.zeros_like(labels, dtype=np.uint8)
            largest_connected_region[labels == largest_label] = 1
            largest_connected_region=largest_connected_region.astype(float)
            
            aa=random.randint(0,1)           
            if is_positive:
                positive_click_num_array[bindx] += 1
                if aa==0:
                    #scrib[bindx,0,:,:]+=bound_scri(largest_connected_region)
                    #scrib[scrib[bindx,0,:,:]>1]=1
                    scrib[bindx,0,:,:]=bound_scri(largest_connected_region)
                    scrib[scrib>1] = 1
                    
                elif aa==1:
                    #scrib[bindx,0,:,:]+=rand_scri(largest_connected_region)
                    #scrib[scrib[bindx,0,:,:]>1]=1
                    scrib[bindx,0,:,:]=rand_scri(largest_connected_region)
                    scrib[scrib>1] = 1
            else:
                negative_click_num_array[bindx] += 1
                if aa==0:
                    #scrib[bindx,1,:,:]+=bound_scri(largest_connected_region)
                    #scrib[scrib[bindx,1,:,:]>1]=1
                    scrib[bindx,1,:,:]=bound_scri(largest_connected_region)
                    scrib[scrib>1] = 1
                elif aa==1:
                    #scrib[bindx,1,:,:]+=rand_scri(largest_connected_region)
                    #scrib[scrib[bindx,1,:,:]>1]=1  
                    scrib[bindx,1,:,:]=rand_scri(largest_connected_region)
                    scrib[scrib>1] = 1              
    return scrib