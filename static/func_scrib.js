var img_temp_name 
var img_temp_list

document.addEventListener("DOMContentLoaded", () => {
// 获取图片元素和div元素
var canvas = document.getElementById('myCanvas');
var context = canvas.getContext('2d');
var scribArea = document.getElementById('scrib_area_');
var image = document.getElementById('scrib_image');
var container = document.querySelector('.scrib_area_container');
var filt_name=document.getElementById('filt_name');
//poly
var canvas_poly = document.getElementById('canvas_poly');
var context_poly = canvas_poly.getContext('2d');
var canvas_bg = document.getElementById('canvas_poly_all');
var context_bg = canvas_bg.getContext('2d');



image.src="static/191.png"

// 监听滚轮事件
scribArea.addEventListener('wheel', function(event) {
  // 检查是否按下了Ctrl键
  if (event.ctrlKey) {
    event.preventDefault();
    let scrollRate = parseFloat(getComputedStyle(container).getPropertyValue("--scroll-rate"));
    if(isNaN(scrollRate)) scrollRate = 1;
    const scaleFactor = event.deltaY < 0
    // scrollRate += scaleFactor ? 0.1 : -0.1;
    scrollRate *= scaleFactor ? 1.1 : 0.9;
    // if(scrollRate < 0.5 || scrollRate > 2) return;
    if(scrollRate < 0.1) return;
    container.style.setProperty("--scroll-rate", scrollRate);

  }
});






// console.log(canvas.width)
// console.log(canvas.height)
canvas.style.position = 'absolute';
var isDrawing = false;
var isErasing = false;
isDrawing = false;
isErasing = false;

var lineWidth = 7;

// 定義一個函數來開始繪製
function startDrawing(e) {
    if (e.buttons == 1 & scrib_pn==1) { // 檢查是否是左鍵按下
        isDrawing = true;
        context.beginPath();
        // var x = e.clientX - canvas.offsetLeft;
        // var y = e.clientY - canvas.offsetTop;
        const x = e.offsetX
        const y= e.offsetY
        context.moveTo(x, y);
        context.strokeStyle = 'red';
        canvas.style.cursor = "url('static/pencil_scrib.png') 0 25, auto";
    }else if (e.buttons == 1 & scrib_pn==0) { // 檢查是否是左鍵按下
            isDrawing = true;
            context.beginPath();
            // var x = e.clientX - canvas.offsetLeft;
            // var y = e.clientY - canvas.offsetTop;
            const x = e.offsetX
            const y= e.offsetY
            context.moveTo(x, y);
            context.strokeStyle = 'green';
            canvas.style.cursor = "url('static/pencil_scrib.png') 0 25, auto";   
    }else if (e.buttons === 2) { // 檢查是否是右鍵按下
        isErasing = true;
        context.beginPath();
        const x = e.offsetX
        const y= e.offsetY
        context.moveTo(x, y);
        context.globalCompositeOperation = "destination-out"; // 使用擦子效果
        canvas.style.cursor = "url('static/eraser25.png') 5 25, auto";
    }
}

// 定義一個函數來繪製
function draw(e) {
    if (isDrawing) {
        // console.log(e)
        let scrollRate = parseFloat(getComputedStyle(container).getPropertyValue("--scroll-rate"));
        if(isNaN(scrollRate)) scrollRate = 1;

        const x = e.offsetX
        const y= e.offsetY
        // console.log(x, y)
        context.lineTo(x, y);
        context.stroke();
    } else if (isErasing) {
        const x = e.offsetX
        const y= e.offsetY
        context.lineWidth = lineWidth; 
        context.lineTo(x, y);
        context.stroke();
    }
}

// 定義一個函數來停止繪製
function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
    } else if (isErasing) {
        isErasing = false;
        context.globalCompositeOperation = "source-over"; // 恢復正常繪製模式
        context.lineWidth=1
        canvas.style.cursor = "url('static/pencil.png') 0 10, auto";
        // canvas.style.cursor = "";
    }
}

// 添加事件監聽器
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
canvas.addEventListener('contextmenu', event => event.preventDefault());



document.getElementById("brain").addEventListener("click", function() {
    var width = image.clientWidth;
    var height = image.clientHeight;
    canvas.width = width;
    canvas.height = height;
    canvas_poly.width=0;
    canvas_poly.height=0;
    //圖層移動
    canvas.style.zIndex = 100;
    canvas_poly.style.zIndex = 1;
    canvas_bg.style.zIndex = 2;
    context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height)
});
var scrib_pn = 1
document.getElementById("pos_scrib").addEventListener("click", function() {
    scrib_pn=1
});
document.getElementById("neg_scrib").addEventListener("click", function() {
    scrib_pn=0
});
document.getElementById("clear_scrib").addEventListener("click", function() {
    context.clearRect(0, 0, canvas.width, canvas.height); 
    //canvas.width = image.clientWidth;
    //canvas.height = image.clientHeight;
});

// get_folder
var get_fold_button=document.getElementById("get_fold_button")
var load_fold
get_fold_button.addEventListener('click',async function () {
    load_fold = await sendDataToBackend('abstract');
        //console.log("Message from Flask:", load_fold.fold_path);
        //console.log("Message from Flask:", load_fold.img_list);
        show_file_list(load_fold.img_list)
        });
    async function sendDataToBackend(action) {
        try{
            const url = '/get_folder' ;
            const response = await fetch(url, {method: 'GET'});
            const data = await response.json();
            console.log(data);
            return data;
        }catch (e){
            console.error('There was a problem with your fetch operation:', e);
            throw e
        }
    }
function show_file_list(files_){
    img_temp_list = files_
    for(let i =0; i<files_.length;i++){
        addParagraph(files_[i])
}}

// 顯示img list
var img_list_area=document.getElementById("img_list_area");

function addParagraph(img_name) {
    let paragraph = document.createElement("p");
    paragraph.textContent = img_name;
    const index = img_temp_list.indexOf(img_name);
    paragraph.id = 'path_'+ String(index);
    paragraph.setAttribute("onclick", "getPath(this)");
    img_list_area.appendChild(paragraph);
}

filt_name.addEventListener('input', function(event) {
    if(event.target.value!=""){
        clearPaths()
        show_file_list(load_fold.img_list.filter(path => path.includes(event.target.value)))
    }
    else{
        clearPaths()
        show_file_list(load_fold.img_list)
    }
});
function clearPaths() {
    img_list_area.innerHTML = ''; // 清空内容
}

// 前後張圖片按鈕


document.getElementById("right_button").addEventListener("click", function() {
    const index = img_temp_list.indexOf(img_temp_name) + 1 ;
    if (index < img_temp_list.length) {
        getPath(document.getElementById('path_'+ String(index)))

    }
});

document.getElementById("left_button").addEventListener("click", function() {
    const index = img_temp_list.indexOf(img_temp_name) - 1 ;
    if (index >= 0) {
        getPath(document.getElementById('path_'+ String(index)))
    }
});

////--------------------------------------------------------------------------polygon
///---------------------------------------------------------------------------
var points = []; // 單個多邊形的頂點
var total_points = [] //全部多邊形的頂點
var get_class = document.querySelector('.input-class');
var dif_class_value = [] //類別
var tol_get_class_value = [] //每個物件的cls
var ok_but = document.getElementById('ok_butt');
var can_but = document.getElementById('can_butt');
var diag= document.getElementById('myDialog');





function retain_total_point(){
    for (let i = 0; i < total_points.length; i++) {
        draw_done_point(total_points[i],context_bg);
        drawPolygon_bg(total_points[i],context_bg);
    }}
document.getElementById("poly").addEventListener("click", function() {
    var width = image.clientWidth;
    var height = image.clientHeight;
    canvas_poly.width = width;
    canvas_poly.height = height;
    canvas_bg.width = width;
    canvas_bg.height = height;

    canvas.width = 0;
    canvas.height = 0;
    //圖層移動
    canvas.style.zIndex = 1;
    canvas_poly.style.zIndex = 100;
    canvas_bg.style.zIndex = 2;
    context.clearRect(0, 0, canvas.width, canvas.height);
    console.log(total_points.length)
    if (total_points.length>0){
        console.log('sss')
        retain_total_point();
    }
    
});
function drawPolygon() {
    context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height);

    context_poly.fillStyle = 'rgba(255, 0, 0, 0.5)';
    context_poly.beginPath();
    context_poly.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        context_poly.lineTo(points[i].x, points[i].y);
    }
    context_poly.closePath();
    context_poly.fill();
}


function drawPreview(x, y) {
    context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height);

    context_poly.fillStyle = 'rgba(255, 0, 0, 0.5)';
    context_poly.beginPath();
    context_poly.moveTo(points[0].x, points[0].y);
    for (var i = 1; i < points.length; i++) {
        context_poly.lineTo(points[i].x, points[i].y);
    }
    context_poly.lineTo(x, y); // 在最後一個頂點和滑鼠位置之間畫一條連線
    context_poly.closePath();
    context_poly.fill();
    drawPoints();
    if (Math.sqrt(Math.pow(x - points[0].x, 2) + Math.pow(y - points[0].y, 2)) <= 10 && points.length >=3  ){
        context_poly.beginPath();
        context_poly.arc(points[0].x, points[0].y, 10, 0, Math.PI * 2);
        context_poly.fill();
    }
}
function drawPoints() {
  for (let i = 0; i < points.length; i++) {
    drawPoint(points[i].x, points[i].y);
  }
}


function drawPoint(x, y) {
    context_poly.fillStyle = 'blue';
    context_poly.beginPath();
    context_poly.arc(x, y, 3, 0, Math.PI * 2);
    context_poly.fill();
}

function draw_done_point(poi,ctx_x){
    for (var i = 0; i < poi.length; i++) {
        ctx_x.fillStyle = 'blue';
        ctx_x.beginPath();
        ctx_x.arc(poi[i].x, poi[i].y, 3, 0, Math.PI * 2);
        ctx_x.fill();
    }
}
function showDialogAtPosition(x, y) {
    diag.style.left = x + 'px';
    diag.style.top = y + 'px';

    diag.showModal();
}



function drawPolygon_bg(poi,ctx_x) {
    //ctx_x.clearRect(0, 0, canvas_bg.width, canvas_bg.height);

    ctx_x.fillStyle = 'rgba(255, 0, 0, 0.5)';
    ctx_x.beginPath();
    ctx_x.moveTo(poi[0].x, poi[0].y);
    for (let i = 1; i < poi.length; i++) {
        ctx_x.lineTo(poi[i].x, poi[i].y);
    }
    ctx_x.lineTo(poi[0].x, poi[0].y);
    ctx_x.closePath();
    ctx_x.fill();
}
canvas_poly.addEventListener('mousemove', function(event) {
    if (points.length > 0) {        
        let x = event.offsetX;
        let y = event.offsetY;
        drawPreview(x, y);
    }
});

canvas_poly.addEventListener('click', function(event) {
    let x = event.offsetX;
    let y = event.offsetY;
    if (points.length <= 2 ){
        points.push({ x: x, y: y });
        drawPolygon();
        drawPoints();       
    }
    else{
        if(Math.sqrt(Math.pow(x - points[0].x, 2) + Math.pow(y - points[0].y, 2)) <= 10 ){
            showDialogAtPosition(event.clientX,event.clientY)
        }
        else{
            points.push({ x: x, y: y });
            drawPolygon();
            drawPoints();   
        }
    }
});



ok_but.addEventListener('click', function(event) {
    context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height);
    //畫背景
    draw_done_point(points,context_bg);
    drawPolygon_bg(points,context_bg)
    total_points.push(points)
    tol_get_class_value.push(get_class.value)
    points = [] ;

    show_ok_can(1)
});
can_but.addEventListener('click', function(event) {
    drawPolygon();
    drawPoints();  

    show_ok_can(0)
});
function show_ok_can(ok_){
    if (ok_==1){
        let get_class_value=get_class.value;
        if (get_class_value!="" && !dif_class_value.includes(get_class_value)){
            dif_class_value.push(get_class_value)
            add_cls_list(get_class_value)

            ////測試回傳json
            jsonto_={
                imgname:img_temp_name,
                poi:total_points,
                cls:tol_get_class_value
                }
            json2thon (jsonto_)   
        }
    }
}


////------------------------------------------------------dialog 移動
const dialog = document.getElementById('myDialog');
const header = document.getElementById('dialogHeader');

let isDragging = false;
let offsetX, offsetY;

header.addEventListener('mousedown', (e) => {
    isDragging = true;
    offsetX = e.clientX - dialog.offsetLeft;
    offsetY = e.clientY - dialog.offsetTop;
});
header.addEventListener('mouseover', () => {
    header.style.cursor = 'grab';
});

header.addEventListener('mouseout', () => {
    header.style.cursor = '';
});
document.addEventListener('mousemove', (e) => {
    if (isDragging) {
    dialog.style.left = `${e.clientX - offsetX}px`;
    dialog.style.top = `${e.clientY - offsetY}px`;
    }
});
document.addEventListener('mouseup', () => {
    isDragging = false;
});

//-------------------dialog 裡面的calass
var class_list = document.getElementById('class_list');
function add_cls_list(cls_name) {
    let paragraph = document.createElement("p");
    paragraph.textContent = cls_name;
    paragraph.id = 'cls_'+ cls_name;
    paragraph.addEventListener('click', function() {
        get_cls(this)
    })
    class_list.appendChild(paragraph);
}
function get_cls(element){
    get_class.value = element.textContent
}

async function json2thon(ddata) {
    try{
        const url = '/json_' ;
        await fetch(url, {
            method: 'POST', 
            headers: {
                'Content-Type': 'application/json' 
            },
            body: JSON.stringify(ddata)    
        });
    }catch (e){
        console.error('There was a problem with your fetch operation:', e);
        throw e
    }
}






});


var selectedPathElement = null;
var image_t
async function getPath(element) {
    if (selectedPathElement) {
        selectedPathElement.classList.remove("selected"); // 移除之前选择的路径的选中效果
    }
    element.classList.add("selected"); // 添加新选择的路径的选中效果
    selectedPathElement = element;
    console.log("所选的路径是：" + element.textContent);
    image_t = document.getElementById('scrib_image');
    image_t.src = await change_img(element.textContent)
    img_temp_name = element.textContent
    async function change_img(temp_i) {
        try{
            const url = '/img_temp' ;
            const response = await fetch(url, {
                method: 'POST', 
                headers: {
                    'Content-Type': 'application/json' 
                },
                body: JSON.stringify({temp: temp_i })    
            });
            const blob = await response.blob();
            const reader = new FileReader();
            await new Promise((resolve, reject) => {
                reader.onload = resolve;
                reader.onerror = reject;
                reader.readAsDataURL(blob);
              });
              return reader.result
        }catch (e){
            console.error('There was a problem with your fetch operation:', e);
            throw e
        }
    }
    }

    