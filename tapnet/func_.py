import numpy as np
import torch.nn.functional as F
import chex
from typing import Sequence
import torch
import cv2
from albumentations import *
import json
import os
device=torch.device('cpu')
#from .torch_ import tapir_model
def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.float()
  frames = frames / 255 * 2 - 1
  return frames


def sample_random_points(frame_max_idx, height, width, num_points):
  """Sample random points with (time, height, width) order."""
  y = np.random.randint(0, height, (num_points, 1))
  x = np.random.randint(0, width, (num_points, 1))
  t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
  points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
  return points


def postprocess_occlusions(occlusions, expected_dist):
  visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) >= 0.4
  return visibles
def polygons_to_mask(polygons, height, width):
    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Fill polygons in the mask
    cv2.fillPoly(mask, polygons, 1)
    
    return mask

def inference(frames, query_points, model):
  # Preprocess video to match model inputs format
  frames = preprocess_frames(frames)
  query_points = query_points.float()
  frames, query_points = frames[None], query_points[None] #增加一維

  # Model inference
  
  outputs = model(frames, query_points)
  tracks, occlusions, expected_dist = outputs['tracks'][0], outputs['occlusion'][0], outputs['expected_dist'][0]
  # track : shape = (num_point,num_frams,2) , 2=[x,y]
  # occlusions expected_dist : shape = (num_point,num_frams)

  # Binarize occlusions
  visibles = postprocess_occlusions(occlusions, expected_dist) #看點是否還在畫面中吧
  return tracks, visibles
def convert_select_points_to_query_points(frame, points):
  """Convert select points to query points.

  Args:
    points: [num_points, 2], in [x, y]
  Returns:
    query_points: [num_points, 3], in [t, y, x]
  """
  points = np.stack(points)
  query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
  query_points[:, 0] = frame
  query_points[:, 1] = points[:, 1]
  query_points[:, 2] = points[:, 0]
  return query_points

def convert_grid_coordinates(
    coords: chex.Array,
    input_grid_size: Sequence[int],
    output_grid_size: Sequence[int],
    coordinate_format: str = 'xy',
) -> chex.Array:
  if isinstance(input_grid_size, tuple):
    input_grid_size = np.array(input_grid_size)
  if isinstance(output_grid_size, tuple):
    output_grid_size = np.array(output_grid_size)

  if coordinate_format == 'xy':
    if input_grid_size.shape[0] != 2 or output_grid_size.shape[0] != 2:
      raise ValueError(
          'If coordinate_format is xy, the shapes must be length 2.')
  elif coordinate_format == 'tyx':
    if input_grid_size.shape[0] != 3 or output_grid_size.shape[0] != 3:
      raise ValueError(
          'If coordinate_format is tyx, the shapes must be length 3.')
    if input_grid_size[0] != output_grid_size[0]:
      raise ValueError('converting frame count is not supported.')
  else:
    raise ValueError('Recognized coordinate formats are xy and tyx.')

  position_in_grid = coords
  position_in_grid = position_in_grid * output_grid_size / input_grid_size

  return position_in_grid
def load_track_img(track_list):
  frames = []
  for png_file in track_list:
    img=cv2.imread(png_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frames.append(cv2.resize(img, (256, 256)))
  frames=np.array(frames)  
  return frames
def mask_to_polygon (mask): 
    temp=np.array(mask.astype(np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(temp , kernel, iterations=1)
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
_augmentator = Compose([
    Normalize(mean=mean, std=std)
], p=1.0)
#################################### new
from shapely.geometry import Polygon
def shrink_polygon(polygon_coords, distance):
    polygon = Polygon(polygon_coords)
    shrunken_polygon = polygon.buffer(-distance)
    return list(shrunken_polygon.exterior.coords)
  
def small_region(polygons, height, width):
  polygons_new=shrink_polygon(polygons,7)
  if polygons_new ==[]:
    return polygons
  else:
    polygons=[]
    for i in range(len(polygons_new)):
      polygons.append([int(polygons_new[i][0]),int(polygons_new[i][1])])
    return polygons
  
def bound_scri(mask):
    temp=np.array(mask,np.float64)
    
    kernel = np.ones((10, 10), np.uint8)
    eroded_mask = cv2.erode((temp*255).astype(np.uint8) , kernel, iterations=1)
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_inside_mask = np.zeros_like(temp,np.float64)
    p=3
    boundary_inside_mask=cv2.drawContours(boundary_inside_mask, contours, -1, 1, thickness=p)
    boundary_inside_mask = boundary_inside_mask*temp
    if np.sum(boundary_inside_mask)>=5:
        return boundary_inside_mask
    else:
        contours, _ = cv2.findContours((temp*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary_inside_mask = np.zeros_like(mask)
        boundary_inside_mask=cv2.drawContours(boundary_inside_mask, contours, -1, 1, thickness=p)
        boundary_inside_mask = boundary_inside_mask*temp
        return boundary_inside_mask
      
      
def bezier_curve(t, points):
    points = np.array(points)
    n = len(points) - 1
    return sum(
        (np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))) *
        (1 - t)**(n - i) * t**i * points[i]
        for i in range(n + 1)
    )

def generate_random_bezier_curve(p0, p3,bonder,num_points=100):
    points=[p0]
    x_min = bonder[2]
    x_max = bonder[3]
    y_min = bonder[0]
    y_max = bonder[1]
    for i in range(28):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        points.append(np.array([y, x]))
    points.append(p3)
    
    t_values = np.linspace(0, 1, num_points)
    curve = np.array([bezier_curve(t, points) for t in t_values])
    
    return curve

def plot_bezier_curve(draw,curve):
    # Draw the Bezier curve
    p=3
    for i in range(len(curve) - 1):
        cv2.line(draw, tuple(curve[i].astype(int)), tuple(curve[i+1].astype(int)), 1, p)
    return draw
import random
def go_scrib(mask):
    scrib=np.zeros_like(mask)
    points = np.argwhere(mask == 1)
    points=points[:, ::-1]
    len_l=len(points)
    
    num_lines=13
    for _ in range(num_lines):
      start_point = tuple(points[random.randint(0,len_l-1)])
      end_point = tuple(points[random.randint(0,len_l-1)])
      
      left =min(start_point[1],end_point[1])
      right = max(start_point[1],end_point[1])
      top =min(start_point[0],end_point[0])
      bottom =max(start_point[0],end_point[0])
      bonder = [left, right,top,bottom] 
      
      curve= generate_random_bezier_curve(start_point, end_point,bonder)
      scrib =plot_bezier_curve(scrib,curve)

    return scrib   
  
def fish_draw(mask):
  scrib=np.zeros_like(mask)
  temp=(np.array(mask,np.float64)*255).astype(np.uint8)
  contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  polygon=contours[0]
  del contours
  del temp
  M = cv2.moments(polygon)
  if M["m00"] != 0:
      cx = int(M["m10"] / M["m00"])
      cy = int(M["m01"] / M["m00"])
  else:
      cx, cy = 0, 0
  for point in polygon:
      cv2.line(scrib, (cx, cy), (point[0][0], point[0][1]), 1, 3)
  return scrib
#################################################
def predict_run(seg_model,model_track,frames,height,width,points,label,track_list):
    if cv2.contourArea(np.array(points)) > height*width/20:
      points = small_region(points,height,width)
    if points ==[]:
      return 0
    query_points = convert_select_points_to_query_points(0, points)
    query_points = convert_grid_coordinates(query_points, (1, height, width), (1, 256, 256), coordinate_format='tyx')

    frames=torch.tensor(frames).to(device)
    query_points = torch.tensor(query_points).to(device)
    tracks, visibles = inference(frames, query_points, model_track)

    tracks = tracks.cpu().numpy()
    visibles = visibles.cpu().numpy()
    tracks=np.transpose(tracks,(1,0,2))
    visibles=np.transpose(visibles,(1,0))
    for i,tt in enumerate(tracks):
      if i==0:
        continue
      temp=tt[visibles[i]==True] #抓出還存在的點
      scrib=np.zeros((2,512, 512))
      polygon=[]
      for poi in temp:
          polygon.append([int(poi[0]),int(poi[1])])
      polygon= [np.array(polygon)]
      grid=polygons_to_mask(polygon,256,256) 
      scrib[0]=cv2.resize(grid,(512, 512), interpolation=cv2.INTER_NEAREST)

      all_sc = np.array(scrib[0])
      img=cv2.imread(track_list[i])
      img=cv2.resize(img,(512,512))
      img=_augmentator(image=img)['image']
      img=np.transpose(img,((2,0,1)))
      img=torch.from_numpy(img).unsqueeze(0)
      scrib=torch.from_numpy(scrib).unsqueeze(0)
      ## seg_model
      
      #全部
      out=seg_model(img,scrib)
      out=(torch.sigmoid(out['final_mask'])>=0.5).to(int)
      pred1=out.numpy()
      del out
      #邊界
      scrib_bou = bound_scri(all_sc)
      scrib_bou=torch.from_numpy(scrib_bou).unsqueeze(0)
      scrib[0,0,...]=scrib_bou
      out=seg_model(img,scrib)
      out=(torch.sigmoid(out['final_mask'])>=0.5).to(int)
      pred2=out.numpy()
      del scrib_bou
      del out
      #畫線
      # draw_scri = go_scrib(all_sc)
      # draw_scri=torch.from_numpy(draw_scri).unsqueeze(0)
      # scrib[0,0,...]=draw_scri
      # out=seg_model(img,scrib)
      # out=(torch.sigmoid(out['final_mask'])>=0.5).to(int)
      # pred3=out.numpy()
      # cv2.imshow('sss',(pred3[0,0,...]*255).astype(np.uint8))
      # cv2.waitKey(0)
      
      
      draw_scri = fish_draw(all_sc)
      draw_scri=torch.from_numpy(draw_scri).unsqueeze(0)
      scrib[0,0,...]=draw_scri
      out=seg_model(img,scrib)
      out=(torch.sigmoid(out['final_mask'])>=0.5).to(int)
      pred3=out.numpy()
      
      
      del draw_scri
      del out
      del scrib
    
      pred = pred1[0,0,...]+pred2[0,0,...]+pred3[0,0,...]
      pred[pred>1]=1
      del pred1
      del pred2
      del pred3
      
      
      

      pred=(pred*255).astype(np.uint8)
      #To polygons
      pred=mask_to_polygon(pred)
      if len(pred)<3:
        return 0
      pred=resize_polygon(pred,512,512,width,height)
      #save_json
      img_name=os.path.basename(track_list[i])
      temp_json=track_list[i][:-3]+'json'
      if os.path.exists(temp_json):
        with open(temp_json, 'r') as file:
            data = json.load(file)
      else:
          data={}
          data["objects"]=[]
          data["imagePath"]=img_name
          data["imageHeight"]=height
          data["imageWidth"] =width
      data["objects"].append({
        "points":pred ,
        "label":label,
        })
      with open(temp_json, 'w') as json_file:
        json.dump(data, json_file, indent=4)
      
      