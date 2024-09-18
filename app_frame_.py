from flask import Flask, render_template, request, jsonify, send_file
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QApplication
import sys
import os
import cv2
import json
import numpy as np
app = Flask(__name__, static_url_path='/static')


@app.route('/get_folder', methods=['get'])
def choose_fold():
    # if request.method == 'get':
    #     action = request.json.get('action')
    #     if action == 'abstract':
    qt_app = QApplication(sys.argv)
    widget = QtWidgets.QTableWidget()
    widget.setFixedHeight(300)
    widget.setFixedHeight(400)
    table_widget = QtWidgets.QTableWidget(widget)
    table_widget.setGeometry(50, 50, 400, 200)
    table_widget.setFixedHeight(300)
    table_widget.setFixedHeight(400)
    fname = QFileDialog.getExistingDirectory(widget, "Open Folder", "")
    table_widget.deleteLater()
    widget.deleteLater()
    qt_app.deleteLater()
    fname = str(fname)

    img_list = []
    for file in os.listdir(fname):
        if file.lower().endswith(('.png', '.jpg')) and os.path.isfile(os.path.join(fname, file)):
            img_list.append(fname + "/" + file)
            # img_list.append(os.path.join(fname, file))
    xx = {
        "fold_path": fname,
        "img_list": img_list
    }
    return jsonify(xx)
    # return jsonify({'status': "str(fname)"})


@app.route('/img_temp', methods=['POST'])
def get_image():
    temp = request.json.get('temp')    
    return send_file(temp)

@app.route('/json_temp', methods=['POST'])
def get_json():
    temp = request.json.get('temp')
    jsox_path=temp[:-3]+'json'
    all_point=[]
    all_cls=[]
    if os.path.exists(jsox_path):
        with open(jsox_path, 'r') as file:
            data = json.load(file)
        for obj in data["objects"]:
            #temp_obj=[]
            # for poi in obj['points']:
            #     temp_obj.append([poi[0], poi[1] ])
            #all_point.append(temp_obj)
            all_point.append(obj['points'])
            all_cls.append(obj['label'])
    return {"point":all_point,"label":all_cls}





@app.route('/json_', methods=['POST'])
def tojson():
    imgname = request.json.get('imgname')
    poi = request.json.get('poi')
    cls = request.json.get('cls')
    
    img_name=os.path.basename(imgname)
    img=cv2.imread(imgname)
    a={}
    a["objects"]=[]
    for i, poi_group in enumerate(poi):
        a["objects"].append({
        "points":poi_group ,
        "label":cls[i],
        })
    a["imagePath"]=img_name
    a["imageHeight"]=img.shape[0]
    a["imageWidth"] =img.shape[1]

    file_path=imgname[:-3]+'json'
    with open(file_path, 'w') as json_file:
        json.dump(a, json_file, indent=4)
        
    return '0'



import sys
import torch
from albumentations import *
sys.path.append("Emic")
from isegm_s.model.haomodel import Haomodel
from tapnet.func_ import predict_run,load_track_img
from tapnet.torch_ import tapir_model
def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict, strict=False)
    
def mask_to_polygon (mask): 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    largest_polygon = []

    for contour in contours:
        contour = contour.squeeze(axis=1)  # Remove redundant dimension
        polygon = contour.tolist()
        if len(polygon) >= 3:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_polygon = polygon

    return largest_polygon
def resize_polygon(poly,ori_w,ori_h,new_w,new_h):
    resized_polygons = []
    for point in poly:
        x = int(point[0] * new_w / ori_w)
        y = int(point[1] * new_h / ori_h)
        resized_polygons.append([x, y])
    return resized_polygons
################################### DeepLearn
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
_augmentator = Compose([
    Normalize(mean=mean, std=std)
], p=1.0)


device=torch.device('cpu')
seg_model= Haomodel().to(device)
load_weights(seg_model,'./Emic/no_mask.pth')
seg_model.eval()

model_track = tapir_model.TAPIR(pyramid_level=1)
model_track.load_state_dict(torch.load('./tapnet/bootstapir_checkpoint.pt'))
model_track = model_track.to(device)

@app.route('/get_scrib', methods=['POST'])
def scrib2seg():
    scrib_data = request.json
    width = scrib_data['width']
    height = scrib_data['height']
    
    data = scrib_data['data']
    data = np.array(data, dtype=np.uint8).reshape((height, width, 4))
    data = data[:, :, :3]
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    
    data = cv2.resize(data, (512, 512))  #換大小
    # cv2.imshow('mm',data)
    # cv2.waitKey(0)
    #pos
    pos = np.zeros((512,512))
    red_channel = data[:, :, 2]
    green_channel = data[:, :, 1]
    blue_channel = data[:, :, 0]
    red_mask = (red_channel == 255)
    green_mask = (green_channel == 0)
    blue_mask = (blue_channel == 0)
    final_mask = red_mask & green_mask & blue_mask
    pos[final_mask] = 1
    #neg
    neg = np.zeros((512,512))
    red_mask = (red_channel == 0)
    green_mask = (green_channel == 255)
    final_mask = red_mask & green_mask & blue_mask
    neg[final_mask] = 1
    #scrib
    mask=np.concatenate([[pos],[neg]], axis=0)
    scrib_t=torch.from_numpy(mask).unsqueeze(0)
    #img
    img_input=scrib_data['img_path']
    img_input=cv2.cvtColor(cv2.resize(cv2.imread(img_input),(512,512)),cv2.COLOR_RGB2BGR)
    img_input=_augmentator(image=img_input)['image']
    img_input=np.transpose(img_input,((2,0,1)))
    img_input=torch.from_numpy(img_input).unsqueeze(0)
    #run model
    output=seg_model(img_input,scrib_t)
    out=(torch.sigmoid(output['final_mask'])>=0.4).to(int)
    pred=out.numpy()
    if np.sum(pred)==0:
        print('no')
        return 'no_obj'
    
    pred=(pred[0,0,...]*255).astype(np.uint8)
    #polygons
    pred=mask_to_polygon(pred)
    pred=resize_polygon(pred,512,512,width,height)
    # mask=np.zeros((512, 512, 3), dtype=np.uint8)
    # mask[ss==1]=[0,255,255]
    # cv2.imshow('mm',mask)
    # cv2.waitKey(0)
    response = {
        'predict_point': pred,
    }

    return jsonify(response)


from isegm.inference import utils
def resize_polygon_c(poly,ori_w,ori_h,new_w,new_h):
    resized_polygons = []
    for point in poly:
        y = int(point[0] * new_w / ori_w)
        x = int(point[1] * new_h / ori_h)
        resized_polygons.append([x, y])
    return resized_polygons
path='./Emic/hr32.pth'
model_c = utils.load_is_model(path, 'cpu')



@app.route('/get_clicks', methods=['POST'])
def click2seg():
    _data = request.json
    width = _data['width']
    height = _data['height']
    poi = _data['pois']
    posfal = _data['posfal']
    img_input=_data['img_path']
    image = cv2.imread(img_input)
    image=cv2.resize(image,(512,512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=torch.tensor(image).permute(2,0,1).unsqueeze(0)
    prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
    inp=torch.cat((image/255, prev_output), dim=1) 
    
    resize_point = resize_polygon_c(poi,width,height,512,512)
    p_points = []
    f_points = []
    for i in range(len(resize_point)):
        if posfal[i]==1:
            p_points.append([resize_point[i][0],resize_point[i][1],i])
        else:
            f_points.append([resize_point[i][0],resize_point[i][1],i])
            
    cross = len(resize_point)-len(p_points)
    for i in range(cross):
        p_points.append([-1,-1,-1])   
    cross = len(resize_point)-len(f_points)      
    for i in range(cross):
        f_points.append([-1,-1,-1])
    tol_points =  torch.tensor(p_points + f_points).unsqueeze(0)
    
    
    temp_mask=None
    eval_outputs, refined_output = model_c(inp.to(device), tol_points.to(device), cached_instances_lr=temp_mask)
    temp_mask=refined_output['instances_refined']

    mask=torch.sigmoid(refined_output['instances_refined'])
    mask=(mask.cpu().numpy()[:, 0, :, :] > 0.49).astype(np.uint8)
    pred=(mask[0]*255).astype(np.uint8)
    # cv2.imshow('mm',mask)
    # cv2.waitKey(0)
    pred=mask_to_polygon(pred)
    pred=resize_polygon(pred,512,512,width,height)
    response = {
        'predict_point': pred,
    }

    return jsonify(response)

@app.route('/tracking', methods=['POST'])
def tracking_label():
    path=request.json.get('img_path')
    file_path=path[:-3]+'json'
    directory_path = os.path.dirname(path)
    img_list = []
    for file in os.listdir(directory_path):
        if file.lower().endswith(('.png', '.jpg')) and os.path.isfile(os.path.join(directory_path, file)):
            img_list.append(directory_path + "/" + file)
    temp_index = img_list.index(path)
    track_list = []
    for i,inx in enumerate(range(temp_index,len(img_list))):
        track_list.append(img_list[inx])
        if i==1:# 這裡只取一張
            break
    frames = load_track_img(track_list)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        width=data['imageWidth']
        height=data['imageHeight']
        for obj in data["objects"]:
            predict_run(seg_model,model_track,frames,height,width,obj['points'],obj['label'],track_list)
        print('doneeee')
    return '0'

@app.route('/')
def index():
    return render_template('frame_.html')


if __name__ == '__main__':
    app.run(debug=True)
