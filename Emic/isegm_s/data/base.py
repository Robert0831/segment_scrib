import random
import pickle
import numpy as np
import torch
from torchvision import transforms
from .points_sampler import MultiPointSampler
from .sample import DSample
import cv2
from isegm.utils.crop_local import random_choose_target,get_bbox_from_mask,getLargestCC,expand_bbox, expand_bbox_with_bias
import skimage
import random

def bound_scri(mask):
    temp=np.array(mask[0])
    kernel = np.ones((10, 10), np.uint8)
    eroded_mask = cv2.erode((temp*255).astype(np.uint8) , kernel, iterations=1)
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_inside_mask = np.zeros_like(temp)
    p=random.randint(1,3)
    boundary_inside_mask=cv2.drawContours(boundary_inside_mask, contours, -1, (255), thickness=p)
    boundary_inside_mask=np.expand_dims(boundary_inside_mask,axis=0)

    return np.concatenate([boundary_inside_mask,np.zeros_like(boundary_inside_mask)],axis=0)

def rand_scri(mask):
    temp=np.zeros_like(mask[0])
    num_lines = random.randint(4,8)
    l=np.argwhere(mask[0])
    l=l[:, ::-1]
    len_l=len(l)
    for _ in range(num_lines):
        start_point = tuple(l[random.randint(0,len_l-1)])
        end_point = tuple(l[random.randint(0,len_l-1)])
        p=random.randint(1,3)
        cv2.line(temp, start_point, end_point, 1, thickness=p)
    temp=np.expand_dims(temp,axis=0)
    temp[temp > 1] = 1
    return np.concatenate([temp,np.zeros_like(temp)],axis=0)

    
class ISDataset(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 augmentator=None,
                 points_sampler=MultiPointSampler(max_num_points=12),
                 min_object_area=0,
                 keep_background_prob=0.0,
                 with_image_info=False,
                 samples_scores_path=None,
                 samples_scores_gamma=1.0,
                 epoch_len=-1,
                 with_refiner = True,
                 focus_size=192):
        super(ISDataset, self).__init__()
        self.epoch_len = epoch_len
        self.augmentator = augmentator
        self.min_object_area = min_object_area
        self.keep_background_prob = keep_background_prob
        self.points_sampler = points_sampler
        self.with_image_info = with_image_info
        self.samples_precomputed_scores = self._load_samples_scores(samples_scores_path, samples_scores_gamma)
        self.to_tensor = transforms.ToTensor()
        self.with_refiner = with_refiner
        self.dataset_samples = None
        if isinstance(focus_size, (tuple, list)):
            h, w = focus_size[0], focus_size[1]
        else:
            h, w = focus_size, focus_size
        self.focus_size = [h, w]

    def __getitem__(self, index):
        iter_num = 0
        while(1):
            try:
                if self.samples_precomputed_scores is not None:
                    index = np.random.choice(self.samples_precomputed_scores['indices'],
                                            p=self.samples_precomputed_scores['probs'])
                else:
                    if self.epoch_len > 0:
                        index = random.randrange(0, len(self.dataset_samples))
                sample = self.get_sample(index)
                sample = self.augment_sample(sample) #去除小區域 1
                sample.remove_small_objects(self.min_object_area)  #去除小區域 0,沒設
                if len(sample) == 0:
                    continue
                self.points_sampler.sample_object(sample)
                
                #### 捨棄
                #points = np.array(self.points_sampler.sample_points()) #取出正負點 (負的點都取-1)
                
                mask = self.points_sampler.selected_mask #取出mask ###################################我要用的 (1,320,480)
                        
                if not (hasattr(self, 'name') and (self.name == 'SBD')):
                    mask = self.remove_small_regions(mask)    
                     
                ######## 畫scribble ##################
                whi=random.randint(0,1)
                if whi==1:
                    scrib=bound_scri(mask.astype(np.uint8)).astype(float)
                else:
                    scrib=rand_scri(mask.astype(np.uint8)).astype(float)
                      
                image = sample.image
                mask_area = mask[0].shape[0] * mask[0].shape[1]
                #trimap = self.get_trimap(mask[0]) #把mask處理得更平滑
                if self.with_refiner:
                    '''
                    #trimap = self.get_trimap(mask[0]) 
                    #if mask[0].sum() < 3600: # 80 * 80
                    y1,x1,y2,x2 = self.sampling_roi_full_object(mask[0])
                    #else:
                    #    if np.random.rand() < 0.4:
                    #        y1,x1,y2,x2 = self.sampling_roi_on_boundary(mask[0])
                    #    else:
                    #        y1,x1,y2,x2 = self.sampling_roi_full_object(mask[0])
                    
                    min_y, max_y = points[:2, 0].min(), points[:2, 0].max()  #比前兩筆
                    min_x, max_x = points[:2, 1].min(), points[:2, 1].max()
                    x1 = min(max(min_x-1, 0), x1)
                    y1 = min(max(min_y-1, 0), y1)
                    x2 = max(max_x+1, x2)
                    y2 = max(max_y+1, y2)
                    
                    h, w = mask.shape[-2], mask.shape[-1]
                    roi = torch.tensor([x1, y1, x2, y2])
                    image_focus = image[y1:y2,x1:x2,:]
                    image_focus = cv2.resize(image_focus, (h,w)) #看連通區最大的那邊

                    mask_255 = (mask[0] * 255).astype(np.uint8)
                    mask_focus = mask_255[y1:y2,x1:x2]
                    mask_focus = cv2.resize(mask_focus, (h,w)) > 128
                    mask_focus = np.expand_dims(mask_focus,0).astype(np.float32)

                    trimap_255 = (trimap[0] * 255).astype(np.uint8)
                    trimap_focus = trimap_255[y1:y2,x1:x2]
                    trimap_focus = cv2.resize(trimap_focus, (h,w)) > 128
                    trimap_focus = np.expand_dims(trimap_focus,0).astype(np.float32)

                    hc,wc = y2-y1, x2-x1
                    ry,rx = h/hc, w/wc
                    bias = np.array([y1,x1,0])
                    ratio = np.array([ry,rx,1])
                    points_focus = (points - bias) * ratio

                    if mask.sum() > self.min_object_area and mask.sum() < mask_area * 0.85:
                        
                        output = {
                            'images': self.to_tensor(image),
                            'points': points.astype(np.float32),
                            'instances': mask,
                            'trimap':trimap,
                            'images_focus':self.to_tensor(image_focus),
                            'instances_focus':mask_focus,
                            'trimap_focus': trimap_focus,
                            'points_focus': points_focus.astype(np.float32),
                            'rois':roi.float()
                        }

                        if self.with_image_info:
                            output['image_info'] = sample.sample_id
                        return output
                    else:
                        index = np.random.randint(len(self.dataset_samples)-1)
                        '''
                    
                    if mask.sum() > self.min_object_area and mask.sum() < mask_area * 0.85:
                        
                        output = {
                            'images': self.to_tensor(image),
                            'instances': mask,
                            #'trimap':trimap,
                            'scri':scrib
                        }

                        if self.with_image_info:
                            output['image_info'] = sample.sample_id
                        return output
                    else:
                        index = np.random.randint(len(self.dataset_samples)-1)     
                                           
                else:
                    if mask.sum() > self.min_object_area and mask.sum() < mask_area * 0.85:
                        output = {
                            #'trimap': trimap,
                            'images': self.to_tensor(image),
                            #'points': points.astype(np.float32),
                            'scri':scrib,
                            'instances': mask,
                        }

                        if self.with_image_info:
                            output['image_info'] = sample.sample_id
                        return output
                    else:
                        index = np.random.randint(len(self.dataset_samples)-1)
            
            except Exception as e:
                index = np.random.randint(len(self.dataset_samples)-1)
                #print(e.args)
                #print(str(e))
                print('.............', e.args, ' ', str(e), ' ', repr(e))
                iter_num += 1
                #print(repr(e))
                #raise e'''
            if iter_num > 20:
                raise ValueError


    def remove_small_regions(self,mask):
        mask = mask[0] > 0.5
        mask = skimage.morphology.remove_small_objects(mask,min_size= 900)
        mask = np.expand_dims(mask,0).astype(np.float32)
        return mask


    def sampling_roi_full_object(self, gt_mask, min_size=32):
        max_mask = getLargestCC(gt_mask) #取出最大連通的mask
        y1,y2,x1,x2 = get_bbox_from_mask(max_mask) #box
        ratio = np.random.randint(11,17)/10
        y1,y2,x1,x2 = expand_bbox_with_bias(gt_mask,y1,y2,x1,x2,ratio,min_size,0.3) #增寬一點
        return y1,x1,y2,x2

    def sampling_roi_on_boundary(self,gt_mask):
        h,w = gt_mask.shape[0], gt_mask.shape[1]
        rh = np.random.randint(15,40)/10
        rw = np.random.randint(15,40)/10
        new_h,new_w = h/rh, w/rw
        crop_size = (int(new_h), int(new_w))

        alpha = gt_mask > 0.5
        alpha = alpha.astype(np.uint8)
        kernel = np.ones((5,5),np.uint8)
        dilate = cv2.dilate(alpha,kernel,iterations = 1)
        boundary = np.logical_and( dilate, np.logical_not(alpha))
        y1,x1,y2,x2 = random_choose_target(boundary,crop_size)
        return y1,x1,y2,x2


    def get_trimap(self, mask):
        h,w = mask.shape[0],mask.shape[1]
        hs,ws = h//8,w//8
        mask_255_big = (mask * 255).astype(np.uint8)
        mask_255_small = (cv2.resize(mask_255_big, (ws,hs)) > 128) * 255 #維持住 0 跟255 ,resize 可能數字可能會變
        mask_resized = cv2.resize(mask_255_small.astype(np.uint8),(w,h)) > 128
        diff_mask = np.logical_xor(mask, mask_resized).astype(np.uint8)

        kernel = np.ones((3, 3), dtype=np.uint8)
        diff_mask = cv2.dilate(diff_mask, kernel, iterations=2) 

        diff_mask = diff_mask.astype(np.float32)
        diff_mask = np.expand_dims(diff_mask,0)
        return diff_mask
        

    def augment_sample(self, sample) -> DSample:
        valid_augmentation = False
        while not valid_augmentation:
            sample.augment(self.augmentator)
            keep_sample = (self.keep_background_prob < 0.0 or
                           random.random() < self.keep_background_prob)
            valid_augmentation = len(sample) > 0 or keep_sample

        return sample

    #看外層dataset
    def get_sample(self, index) -> DSample:
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return self.get_samples_number()

    def get_samples_number(self):
        return len(self.dataset_samples)

    @staticmethod
    def _load_samples_scores(samples_scores_path, samples_scores_gamma):
        if samples_scores_path is None:
            return None

        with open(samples_scores_path, 'rb') as f:
            images_scores = pickle.load(f)

        probs = np.array([(1.0 - x[2]) ** samples_scores_gamma for x in images_scores])
        probs /= probs.sum()
        samples_scores = {
            'indices': [x[0] for x in images_scores],
            'probs': probs
        }
        print(f'Loaded {len(probs)} weights with gamma={samples_scores_gamma}')
        return samples_scores
