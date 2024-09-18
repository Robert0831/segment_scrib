from isegm_s.model.modeling.hrnet_ocr import HighResolutionNet
from isegm_s.model.modifiers import LRMult
from isegm_s.model.ops import ScaleLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
#scrubble 用2個channel表示
# if image size = 512 X 512
# pre_backbone_path='backbone.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class XConvBnRelu2(nn.Module):
    """
    Xception conv bn relu
    """
    def __init__(self, input_dims = 3, out_dims = 16,   **kwargs):
        super(XConvBnRelu2, self).__init__()
        self.conv3x3_1 = nn.Conv2d(input_dims,input_dims,3,1,1,groups=input_dims)
        self.norm1 = nn.BatchNorm2d(input_dims)
        self.conv3x3_2 = nn.Conv2d(input_dims,input_dims,3,1,1,groups=input_dims)
        self.conv1x1 = nn.Conv2d(input_dims,out_dims,1,1,0)
        self.norm2 = nn.BatchNorm2d(out_dims)
        self.activation = nn.ReLU()
    def forward(self,x):
        x = self.conv3x3_1(x)
        x = self.norm1(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x
class Haomodel(nn.Module):
    def __init__(self,cor_dim=16,mid_dims=96,pre_backbone_path=None):
        super(Haomodel, self).__init__()
        self.backbone=HighResolutionNet(width=18, ocr_width=48, small=True,num_classes=1, norm_layer=nn.BatchNorm2d) #(b,1,h/4,w/4) (b,1,h/4,w/4) (b,96,h/4,w/4) 
        self.backbone.apply(LRMult(0.1))
        if pre_backbone_path!=None:
            self.backbone.load_state_dict(torch.load(pre_backbone_path))
        self.prev_mask=None
        self.scribble_set=None
        self.first=True
        
        self.proto_fea=nn.Sequential(
        nn.Conv2d(in_channels=96, out_channels=192, kernel_size=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.BatchNorm2d(192),
        nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=2, padding=1),
        )
        
        self.gaussian_blur_transform = T.GaussianBlur(kernel_size=9)
        mt_layers = [
        nn.Conv2d(in_channels=2, out_channels=16, kernel_size=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.BatchNorm2d(16),
        nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
        ScaleLayer(init_value=0.05, lr_mult=1),  ##　乘上一個數字（可學的）
        nn.BatchNorm2d(64),
        ]
        self.maps_transform = nn.Sequential(*mt_layers)
        
        self.pos_cor=XConvBnRelu2(1,cor_dim)
        self.neg_cor=XConvBnRelu2(1,cor_dim)
        self.prim_fea=XConvBnRelu2(2*cor_dim,mid_dims)
        
        # for atten feature
        self.cross_attention = nn.MultiheadAttention(mid_dims, 4)
        self.adp_mask=nn.Sequential(
            nn.Conv2d(mid_dims, mid_dims,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_dims),
            nn.ReLU(),
            nn.Conv2d(mid_dims, 1,kernel_size=1, stride=1, padding=0, bias=True)
            )
        #第二階段
        self.bag_fea=nn.Sequential(
            nn.Conv2d(6,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(16,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64,mid_dims,1,1,0),
            )
        self.upfea=nn.Sequential(
            nn.Conv2d(mid_dims,mid_dims,3,1,1),
            nn.BatchNorm2d(mid_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(mid_dims,mid_dims,1),
        )
        self.P_b=nn.Parameter(torch.tensor(0.5))
        self.P_f=nn.Parameter(torch.tensor(0.5))
        self.comb=nn.Sequential(
            nn.Conv2d(mid_dims,mid_dims,3,1,1),
            nn.BatchNorm2d(mid_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(mid_dims,mid_dims,3,1,1),
            nn.BatchNorm2d(mid_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(mid_dims,mid_dims,1)
        )
        self.P_b2=nn.Parameter(torch.tensor(0.5))
        self.P_f2=nn.Parameter(torch.tensor(0.5))
        self.out=nn.Sequential(
            XConvBnRelu2(mid_dims,mid_dims),
            nn.Conv2d(mid_dims,1,3,1,1),
        )
    def forward(self,image,scribble):
        if self.scribble_set==None:
            self.scribble_set=scribble.detach().clone()
            temp_scri=self.scribble_set.detach().clone()
        else:
            #self.scribble_set=torch.clamp((self.scribble_set+scribble).detach().clone(), max=1)
            self.scribble_set=scribble.detach().clone()
            temp_scri=self.scribble_set.detach().clone()

        
        if self.first==True:
                                 
            _scri=self.gaussian_blur_transform(temp_scri*255)/255
            self.prev_mask=torch.zeros(_scri[:,:1,:,:].shape).to(device)
            
            scri_fea=self.maps_transform(_scri.float()) #不要前一個mask了
            #scri_fea=self.maps_transform(torch.cat((_scri.float(),self.prev_mask.float()),1)) # (b,3,h/4,w/4) ; pos neg prev_mask
            
            mask_aux, feature=self.backbone(image,scri_fea) #(b,1,h/4,w/4) (b,1,h/4,w/4) (b,96,h/4,w/4) 
            #outputs={'instances': mask, 'instances_aux':mask_aux, 'feature': feature}
            #outputs['instances_lr'] = outputs['instances'].detach() #prev_mask
            self.first=False
            #self.avg=nn.AvgPool2d(feature.shape[2:],stride=1)
        else:
            _scri=self.gaussian_blur_transform(temp_scri*255)/255
            scri_fea=self.maps_transform(_scri.float()) #不要前一個mask了
            #scri_fea=self.maps_transform(torch.cat((_scri.float(),self.prev_mask.float()),1))
            mask_aux, feature=self.backbone(image,scri_fea)
        
        #attention  
        proto_fea=self.proto_fea(feature)  #(b,384,h/8,w/8)  拿來做proto 的 feature map
        
        temp_scri_s=F.interpolate(temp_scri,(proto_fea.shape[2:])) # (b,2,h/8,w/8) 將scrabble 縮小
        self.pos_=proto_fea*temp_scri_s[:,:1,:,:] #(b,384,h/8,w/8)  #找出positive proto 位置
        self.neg_=proto_fea*temp_scri_s[:,1:,:,:] #(b,384,h/8,w/8)  #找出negtive proto 位置
        
        b_,c_,h_,w_=feature.shape
        self.pos_proto=torch.mean(self.pos_, dim=(2, 3), keepdim=True) #(b,384,1,1)->  positive proto
        self.neg_proto=torch.mean(self.neg_, dim=(2, 3), keepdim=True) #(b,384,1,1) negtive proto
        self.pos_fea=F.conv2d(proto_fea.reshape(1,-1,h_//2,w_//2).float(),self.pos_proto.float(),groups=b_).reshape(b_,-1,h_//2,w_//2) #(b,1,h/8,w/8)
        self.neg_fea=F.conv2d(proto_fea.reshape(1,-1,h_//2,w_//2).float(),self.neg_proto.float(),groups=b_).reshape(b_,-1,h_//2,w_//2) #(b,1,h/8,w/8)
        
        self.pos_fea_mask=self.pos_cor(self.pos_fea) #(b,16,h/8,w/8)
        self.neg_fea_mask=self.pos_cor(self.neg_fea) #(b,16,h/8,w/8)
        self.prim_mask=torch.cat((self.pos_fea_mask,self.neg_fea_mask),1) #(b,32,h/8,w/8)
        self.prim_mask=self.prim_fea(self.prim_mask) # (b,96,h/8,w/8)
        
        kv=self.prim_mask.permute(0, 2, 3, 1).reshape(b_,-1, c_).permute(1, 0, 2)
        atten=self.cross_attention(feature.permute(0, 2, 3, 1).reshape(b_, -1, c_).permute(1, 0, 2),kv,kv)[0].permute(1, 2, 0).reshape(b_,c_,h_,w_) #b,96,h/4,w/4
        atten=feature+atten #b,96,h/4,w/4
        atten=self.adp_mask(atten) #b,1,h/4,w/4
        
        #第二階段
        bag=torch.cat((F.interpolate(atten,image.shape[2:],mode='bilinear', align_corners=True),temp_scri,image),1) #b,6(1+2+3),h,w
        bag=self.bag_fea(bag.float()) #b,96,h/2,w/2
        feature=F.interpolate(self.upfea(feature),bag.shape[2:],mode='bilinear', align_corners=True) #b,96,h/2,w/2
        
        comb=self.P_b*bag+self.P_f*feature
        comb=self.comb(comb)
        comb=self.P_b2*comb+self.P_f2*(F.interpolate(atten,comb.shape[2:],mode='bilinear', align_corners=True))
        
        comb=self.out(comb)
        self.prev_mask=torch.sigmoid(F.interpolate(comb.detach().clone(),temp_scri.shape[2:],mode='bilinear', align_corners=True))# (b,1,h,w)
        
        mask_aux=F.interpolate(mask_aux,image.shape[2:],mode='bilinear', align_corners=True)
        atten=F.interpolate(atten,image.shape[2:],mode='bilinear', align_corners=True)
        comb=F.interpolate(comb,image.shape[2:],mode='bilinear', align_corners=True)
        outputs={'instances_aux':mask_aux,'adp_mask':atten,'final_mask': comb} # (H/4,W/4) (H/4,W/4) (H/2,W/2) ->H,W
        return outputs

            
    def model_init(self):
        self.prev_mask=None
        self.scribble_set=None
        self.first=True