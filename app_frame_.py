from flask import Flask, render_template, request, jsonify, send_file
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QApplication
import sys
import os
import cv2
import json

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
    # print(temp)
    return send_file(temp)


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
        poi_t = []
        for point in poi_group:
            poi_t.append([point['x'], point['y']])
        a["objects"].append({
        "label":poi_t ,
        "points":cls[i],
        })

    a["imagePath"]=img_name
    a["imageHeight"]=img.shape[0]
    a["imageWidth"] =img.shape[1]

    file_path=imgname[:-3]+'json'
    with open(file_path, 'w') as json_file:
        json.dump(a, json_file, indent=4)
        
    return '0'



@app.route('/')
def index():
    return render_template('frame_.html')


if __name__ == '__main__':
    app.run(debug=True)
