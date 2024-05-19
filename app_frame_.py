from flask import Flask, render_template, request, jsonify, send_file
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QApplication
import sys
import os

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
    print(temp)
    return send_file(temp)


@app.route('/json_', methods=['POST'])
def tojson():
    imgname = request.json.get('imgname')
    poi = request.json.get('poi')
    cls = request.json.get('cls')
    print(imgname)
    print(poi)
    print(cls)
    return '0'



@app.route('/')
def index():
    return render_template('frame_.html')


if __name__ == '__main__':
    app.run(debug=True)
