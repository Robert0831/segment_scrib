var img_temp_name 
var img_temp_list
var total_points = [] //全部多邊形的頂點v
var points = []; // 單個多邊形的頂點
var tol_get_class_value = [] //每個物件的cls
var dif_class_value = [] //類別
var now_act='None'
var tol_color = {}
var usedmodel=0
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
var canvas_none = document.getElementById('canvas_none');
var context_none = canvas_none.getContext('2d');
var label_col_list = document.getElementById('dif_col_list')
var obj_list = document.getElementById('obj_list')

var canvas_click = document.getElementById('canvas_click');
var context_click = canvas_click.getContext('2d');

image.src="static/init.png"

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
            //context.strokeStyle = 'green';
            context.strokeStyle = 'rgba(0,255,0,1)';
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
    canvas.style.cursor = "url('static/pencil_scrib.png') 0 25, auto";
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
        canvas.style.cursor = "url('static/eraser25.png') 5 25, auto";
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


//------------------------------------------------------------------------Deep learnign
document.getElementById("brain").addEventListener("click", function() {
    if (img_temp_name!=null){
        points=[]
        now_act = "brain"
        var width = image.clientWidth;
        var height = image.clientHeight;
        canvas.width = width;
        canvas.height = height;
        canvas_poly.width=width;
        canvas_poly.height=height;
        canvas_click.width=width;
        canvas_click.height=height;
        //圖層移動
        canvas.style.zIndex = 100;
        canvas_poly.style.zIndex = 1;
        canvas_bg.style.zIndex = 2;
        canvas_none.style.zIndex = -1
        canvas_click.style.zIndex = 3;
        context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height)
        context.clearRect(0, 0, canvas.width, canvas.height)
        context_click.clearRect(0, 0, canvas_click.width, canvas_click.height)
    }
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
    context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height); 
    points=[]
});
document.getElementById("sent_scrib").addEventListener("click", async function () {
    if (now_act== 'brain'){
        document.getElementById('remind').innerText = 'Please Wait!!!';
        const scrib_data = context.getImageData(0, 0, canvas.width, canvas.height);
        const dataToSend = {
            width: canvas.width,
            height: canvas.height,
            data: Array.from(scrib_data.data),
            img_path:img_temp_name
        };
        context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height)
        points=[]
        points = await sendscrib(dataToSend);
        //console.log(points)
        drawPolygon();
        //drawPoints();  
        //console.log(points)
        usedmodel=1
        document.getElementById('remind').innerText = 'Done!';
        };

    async function sendscrib(scrib_data) {
        try{
            const url = '/get_scrib' ;
            const response = await fetch(url,{
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(scrib_data)});
            const data = await response.json();
            return data.predict_point;
        }catch (e){
            console.error('There was a problem with your fetch operation:', e);
            throw e
        }
    }    


});
document.getElementById("finish_pred").addEventListener("click", function() {
    if (usedmodel==1){       
        //showDialogAtPosition(points[0][0],points[0][1])
        showDialogAtPosition(window.innerWidth*0.5,window.innerHeight*0.5)
        usedmodel=0
        context.clearRect(0, 0, canvas.width, canvas.height); 
        context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height); 
    }

});
document.getElementById("finish_click").addEventListener("click", function() {
    if (usedmodel==1){       
        //showDialogAtPosition(points[0][0],points[0][1])
        showDialogAtPosition(window.innerWidth*0.5,window.innerHeight*0.5)
        usedmodel=0
        context.clearRect(0, 0, canvas.width, canvas.height); 
        context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height); 
        context_click.clearRect(0, 0, canvas_click.width, canvas_click.height); 
    }

});
document.getElementById("tracking").addEventListener("click",async function() {
    document.getElementById('remind').innerText = 'Please Wait!!!';
    usedmodel = 0
        const dataToSend = {
            img_path:img_temp_name
        };
        context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height)
        context.clearRect(0, 0, canvas.width, canvas.height)
        await tracking_(dataToSend);
        console.log('doneeeee');

        async function tracking_(dataToSend) {
            try{
                const url = '/tracking' ;
                const response = await fetch(url,{
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(dataToSend)});
                const data = await response.json();
            }catch (e){
                console.error('There was a problem with your fetch operation:', e);
                throw e
            }
        }    
    document.getElementById('remind').innerText = 'Done!';      
});
//------------------------------------------------------------------------

//------------------------------------------------------------------------ get_folder
var get_fold_button=document.getElementById("get_fold_button")
var load_fold
var img_list_area=document.getElementById("img_list_area");
get_fold_button.addEventListener('click',async function () {
    usedmodel=0
    img_list_area.innerHTML=''
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
    }
    if (files_.length>0){
        getPath(document.getElementById('path_0'))
        canvas_none.width = image.clientWidth;
        canvas_none.height = image.clientHeight;
        canvas_bg.style.zIndex = 1;
        canvas_none.style.zIndex = 200;
    }
}

// 顯示img list


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
    if (img_temp_name!=null){
        const index = img_temp_list.indexOf(img_temp_name) + 1 ;
        if (index < img_temp_list.length) {
            getPath(document.getElementById('path_'+ String(index)))

        }
    }
    usedmodel=0
});

document.getElementById("left_button").addEventListener("click", function() {
    if (img_temp_name!=null){
        const index = img_temp_list.indexOf(img_temp_name) - 1 ;
        if (index >= 0) {
            getPath(document.getElementById('path_'+ String(index)))
        }
    }
    usedmodel=0
});

////--------------------------------------------------------------------------polygon
///---------------------------------------------------------------------------

var get_class = document.querySelector('.input-class');
var ok_but = document.getElementById('ok_butt');
var can_but = document.getElementById('can_butt');
var diag= document.getElementById('myDialog');

var get_class2 = document.querySelector('.input-class2');
var class_list2 = document.getElementById('class_list2');



function retain_total_point(pointer=null,ifc=0){
    //obj_list.innerHTML = '';
    context_bg.clearRect(0, 0, canvas_bg.width, canvas_bg.height);
    for (let i = 0; i < total_points.length; i++) {
        draw_done_point(total_points[i],context_bg);
        if (i==pointer){
            drawPolygon_bg(total_points[i],context_bg,tol_get_class_value[i],pointer,ifc);
        }
        else{
            drawPolygon_bg(total_points[i],context_bg,tol_get_class_value[i],null,ifc);
        }
    }}
document.getElementById("poly").addEventListener("click", function() {
    if (img_temp_name!=null){
        now_act = 'poly'
        usedmodel=0
        var width = image.clientWidth;
        var height = image.clientHeight;
        canvas_poly.width = width;
        canvas_poly.height = height;
        canvas_bg.width = width;
        canvas_bg.height = height;
        context.clearRect(0, 0, canvas.width, canvas.height);
        canvas.width = 0;
        canvas.height = 0;
        context_click.clearRect(0, 0, canvas_click.width, canvas_click.height);
        canvas_click.width = 0;
        canvas_click.height = 0;
        //圖層移動
        canvas.style.zIndex = 1;
        canvas_poly.style.zIndex = 100;
        canvas_bg.style.zIndex = 2;
        canvas_click.style.zIndex = 3;
        canvas_none.style.zIndex = -1
        if (total_points.length>0){
            retain_total_point();
        }
    }
});

var ispos=1
document.getElementById("clicks").addEventListener("click", function() {
    if (img_temp_name!=null){
        now_act = 'clicks'
        usedmodel=0
        var width = image.clientWidth;
        var height = image.clientHeight;
        context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height);
        canvas_poly.width = width;
        canvas_poly.height = height;
        canvas_click.width = width;
        canvas_click.height = height;
        canvas_bg.width = width;
        canvas_bg.height = height;
        context.clearRect(0, 0, canvas.width, canvas.height);
        canvas.width = 0;
        canvas.height = 0;
        //圖層移動
        canvas.style.zIndex = 1;
        canvas_poly.style.zIndex = 3;
        canvas_bg.style.zIndex = 2;
        canvas_none.style.zIndex = -1
        canvas_click.style.zIndex = 100;

        ispos=1

        if (total_points.length>0){
            retain_total_point();
        }
    }
});

function getRandomRGBAColor(alpha=0.5) {
    const r = Math.floor(Math.random() * 256);
    const g = Math.floor(Math.random() * 256);
    const b = Math.floor(Math.random() * 256);
    return `rgba(${r}, ${g}, ${b},${alpha})`;
}
function drawPolygon() {
    context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height);

    context_poly.fillStyle = 'rgba(255, 0, 0, 0.5)';
    context_poly.beginPath();
    context_poly.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
        context_poly.lineTo(points[i][0], points[i][1]);
    }
    context_poly.closePath();
    context_poly.fill();
}


function drawPreview(x, y) {
    context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height);

    context_poly.fillStyle = 'rgba(255, 0, 0, 0.5)';
    context_poly.beginPath();
    context_poly.moveTo(points[0][0], points[0][1]);
    for (var i = 1; i < points.length; i++) {
        context_poly.lineTo(points[i][0], points[i][1]);
    }
    context_poly.lineTo(x, y); // 在最後一個頂點和滑鼠位置之間畫一條連線
    context_poly.closePath();
    context_poly.fill();
    drawPoints();
    if (Math.sqrt(Math.pow(x - points[0][0], 2) + Math.pow(y - points[0][1], 2)) <= 10 && points.length >=3  ){
        context_poly.beginPath();
        context_poly.arc(points[0][0], points[0][1], 10, 0, Math.PI * 2);
        context_poly.fill();
    }
}
function drawPoints() {
  for (let i = 0; i < points.length; i++) {
    drawPoint(points[i][0], points[i][1]);
  }
}


function drawPoint(x, y) {
    context_poly.fillStyle = 'rgba(255,165,0,0.5)';
    context_poly.beginPath();
    context_poly.arc(x, y, 5, 0, Math.PI * 2);
    context_poly.fill();
}

function draw_done_point(poi,ctx_x){
    for (var i = 0; i < poi.length; i++) {
        ctx_x.fillStyle = 'rgba(255,165,0,0.5)';
        ctx_x.beginPath();
        ctx_x.arc(poi[i][0], poi[i][1], 5, 0, Math.PI * 2);
        ctx_x.fill();
    }
}
function showDialogAtPosition(x, y) {
    diag.style.left = x + 'px';
    diag.style.top = y + 'px';

    diag.showModal();
}



function drawPolygon_bg(poi,ctx_x,cls,pointer=null,ifc=0) {
    //ctx_x.clearRect(0, 0, canvas_bg.width, canvas_bg.height);
    if (cls in tol_color){
        if (pointer!=null){
            let now_alpha_color= tol_color[cls]
            now_alpha_color = now_alpha_color.replace(/(\d?\.?\d+)(?=\s*\))/, '0.7');
            ctx_x.fillStyle = now_alpha_color
        }
        else{
            ctx_x.fillStyle = tol_color[cls];
        }
    }
    else{
        let col = getRandomRGBAColor();
        tol_color[cls] = col;
        ctx_x.fillStyle = col;
        change_cls_col_list()
    }
    if (ifc==0) {each_cls_col_list()}
    
    ctx_x.beginPath();
    ctx_x.moveTo(poi[0][0], poi[0][1]);
    for (let i = 1; i < poi.length; i++) {
        ctx_x.lineTo(poi[i][0], poi[i][1]);
    }
    ctx_x.lineTo(poi[0][0], poi[0][1]);
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
        points.push([x, y]);
        drawPolygon();
        drawPoints();       
    }
    else{
        if(Math.sqrt(Math.pow(x - points[0][0], 2) + Math.pow(y - points[0][1], 2)) <= 10 ){
            showDialogAtPosition(event.clientX,event.clientY)
        }
        else{
            points.push([x, y]);
            drawPolygon();
            drawPoints();   
        }
    }
});
// #####################click model
var poi_click=[]
var posfalpoi=[]
document.getElementById("clear_click").addEventListener("click", function() {
    context_click.clearRect(0, 0, canvas_click.width, canvas_click.height); 
    context.clearRect(0, 0, canvas.width, canvas.height); 
    context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height); 
    poi_click=[]
    posfalpoi=[]
    points=[]
});
document.getElementById("pos_click").addEventListener("click", function() {
    ispos=1
});
document.getElementById("neg_click").addEventListener("click", function() {
    ispos=0
});
canvas_click.addEventListener('click', function(event) {
    let x = event.offsetX;
    let y = event.offsetY;
    poi_click.push([x, y]);
    posfalpoi.push(ispos)
    drawPoint_click(x, y)
});
function drawPoint_click(x, y) {
    if (ispos==1){
        context_click.fillStyle = 'rgba(255,0,0,0.5)';
    }
    else{
        context_click.fillStyle = 'rgba(0,255,0,0.5)';
    }
    context_click.beginPath();
    context_click.arc(x, y, 5, 0, Math.PI * 2);
    context_click.fill();
}
document.getElementById("sent_click").addEventListener("click", async function() {
    if (now_act== 'clicks'){
        document.getElementById('remind').innerText = 'Please Wait!!!';
        const dataToSend = {
            width: canvas_click.width,
            height: canvas_click.height,
            pois: poi_click,
            posfal : posfalpoi,
            img_path:img_temp_name
        };
        context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height)
        points=[]
        points = await sendclick(dataToSend);
        drawPolygon();
        usedmodel=1
        document.getElementById('remind').innerText = 'Done!';
        };
        async function sendclick(_data) {
            try{
                const url = '/get_clicks' ;
                const response = await fetch(url,{
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(_data)});
                const data = await response.json();
                return data.predict_point;
            }catch (e){
                console.error('There was a problem with your fetch operation:', e);
                throw e
            }
        }    
});


ok_but.addEventListener('click', function(event) {
    context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height);
    //畫背景
    draw_done_point(points,context_bg);
    drawPolygon_bg(points,context_bg,get_class.value)
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
        console.log(get_class_value)
        if (get_class_value!="" && !dif_class_value.includes(get_class_value)){
            dif_class_value.push(get_class_value)
            add_cls_list(get_class_value)
            add_cls_list_2(get_class_value)
            
            
        }
        jsonto_={
            imgname:img_temp_name,
            poi:total_points,
            cls:tol_get_class_value
            }
        json2thon (jsonto_)   
        each_cls_col_list()
    }
}
/// ----------------------        modeify bg_label
var draggedPoint = null;
var draggedObjIndex = null;
var hoverObjIndexv = null
var isDragging_bg = false;
var offsetX_bg, offsetY_bg;
document.getElementById("modeify").addEventListener("click", function() {
    if (img_temp_name!=null){
        now_act = 'modeify'
        usedmodel=0
        var width = image.clientWidth;
        var height = image.clientHeight;
        context.clearRect(0, 0, canvas.width, canvas.height);
        context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height);
        context_click.clearRect(0, 0, canvas_click.width, canvas_click.height);
        canvas_poly.width = 0;
        canvas_poly.height = 0;
        canvas.width = 0;
        canvas.height = 0;
        canvas_click.width = 0;
        canvas_click.height = 0;
        canvas_bg.width = width;
        canvas_bg.height = height;
        //圖層移動
        canvas.style.zIndex = 1;
        canvas_poly.style.zIndex = 2;
        canvas_bg.style.zIndex = 100;
        canvas_none.style.zIndex = -1;
        canvas_click.style.zIndex = 3;
        if (total_points.length>0){
            retain_total_point();
        }
    }
});

canvas_bg.addEventListener('mousedown', (e) => {
    offsetX_bg = e.offsetX;
    offsetY_bg = e.offsetY;
    draggedPoint = null;
    draggedObjIndex = null;
    isDragging_bg = false;
    // Check if click is on a vertex
    for (let i = 0; i < total_points.length; i++) {
        const obj_temp = total_points[i];
        draggedPoint = obj_temp.find(point => isPointClicked(point, offsetX_bg, offsetY_bg));
        if (draggedPoint) {
            draggedObjIndex = i;
            break;
        }
    }

    // Check if click is on an edge
    if (!draggedPoint) {
        for (let i = 0; i < total_points.length; i++) {
            const obj_temp = total_points[i];
            for (let j = 0; j < obj_temp.length; j++) {
                const startPoint = obj_temp[j];
                const endPoint = obj_temp[(j + 1) % obj_temp.length];
                if (isPointOnLine(startPoint, endPoint, offsetX_bg, offsetY_bg)) {
                    const newPoint = [offsetX_bg, offsetY_bg];
                    obj_temp.splice(j + 1, 0, newPoint);
                    draggedPoint = newPoint;
                    draggedObjIndex = i;
                    break;
                }
            }
            if (draggedPoint) break;
        }
    }
    if (draggedPoint) canvas_bg.style.cursor = "pointer";
    // Check if click is inside 
    if (!draggedPoint) {
        for (let i = 0; i < total_points.length; i++) {
            const obj_temp = total_points[i];
            if (isPointInPolygon([offsetX_bg, offsetY_bg], obj_temp)) {
                isDragging_bg = true;
                draggedObjIndex = i;
                break;
            }
        }
    }
    retain_total_point()
});

canvas_bg.addEventListener('mousemove', (e) => {
    //if (now_act=='modeify'){
        canvas_bg.style.cursor = "grab"
    //}
    const newOffsetX = e.offsetX;
    const newOffsetY = e.offsetY;

    if (draggedObjIndex!=null){
        hoverObjIndexv = draggedObjIndex
    }
    else{
        for (let i = 0; i < total_points.length; i++) {
            const obj_temp = total_points[i];
            if (isPointInPolygon([newOffsetX, newOffsetY], obj_temp)) {
                hoverObjIndexv = i;
                break;
            }
        }
    }

    if (draggedPoint) {
        draggedPoint[0] = newOffsetX;
        draggedPoint[1] = newOffsetY;
    } else if (isDragging_bg) {
        canvas_bg.style.cursor = "grab"
        const dx = newOffsetX - offsetX_bg;
        const dy = newOffsetY - offsetY_bg;
        total_points[draggedObjIndex].forEach(point => {
            point[0] += dx;
            point[1] += dy;
        });
        offsetX_bg = newOffsetX;
        offsetY_bg = newOffsetY;
    }
    
    retain_total_point(pointer=hoverObjIndexv);
    hoverObjIndexv=null
});

canvas_bg.addEventListener('mouseup', () => {
    draggedPoint = null;
    draggedObjIndex = null;
    isDragging_bg = false;
    canvas_bg.style.cursor = "";
    jsonto_={
        imgname:img_temp_name,
        poi:total_points,
        cls:tol_get_class_value
        }
    json2thon (jsonto_)  
});

canvas_bg.addEventListener('mouseout', () => {
    draggedPoint = null;
    draggedObjIndex = null;
    isDragging_bg = false;
    canvas_bg.style.cursor = "";
    jsonto_={
        imgname:img_temp_name,
        poi:total_points,
        cls:tol_get_class_value
        }
    json2thon (jsonto_)  
});

function isPointClicked(point, x, y) {
    const radius = 10;
    return Math.hypot(point[0] - x, point[1] - y) < radius;
}

function isPointOnLine(start, end, x, y) {
    const buffer = 0.5; // Click tolerance
    const distToStart = Math.hypot(start[0] - x, start[1] - y);
    const distToEnd = Math.hypot(end[0] - x, end[1] - y);
    const lineLength = Math.hypot(start[0] - end[0], start[1] - end[1]);
    return distToStart + distToEnd >= lineLength - buffer && distToStart + distToEnd <= lineLength + buffer;
}

function isPointInPolygon(point, polygon) {
    let x = point[0], y = point[1];
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        let xi = polygon[i][0], yi = polygon[i][1];
        let xj = polygon[j][0], yj = polygon[j][1];
        
        let intersect = ((yi > y) != (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
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

//-------------------dialog 裡面的class
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

//=====================================================右上 cols_color

function change_cls_col_list(){
    label_col_list.innerHTML = '';
    for (var cls in tol_color) {
        add_cls_col_list(cls,tol_color[cls])
    }
    function add_cls_col_list(cls_name,col) {
        let paragraph = document.createElement("p");
        paragraph.id = 'col_'+ cls_name;
        // Creating color dot
        let dot = document.createElement("span");
        dot.classList.add("dot");
        dot.style.backgroundColor = col;
        paragraph.appendChild(dot);
        paragraph.innerHTML += cls_name;
    
        label_col_list.appendChild(paragraph);
      }
}
//=====================================================右中 個別cls
var click_cls
function each_cls_col_list(){
    obj_list.innerHTML = '';
    for (let i=0 ;i< tol_get_class_value.length;i++) {
        add_cls_dif_list(tol_get_class_value[i],tol_color[tol_get_class_value[i]],i);
    }
    function add_cls_dif_list(cls_name,col,count) {
        let paragraph = document.createElement("p");
        paragraph.id = 'cls_'+ count;
        // Creating color dot
        let dot = document.createElement("span");
        dot.classList.add("dot");
        dot.style.backgroundColor = col;
        paragraph.appendChild(dot);
        paragraph.innerHTML += cls_name;
        obj_list.appendChild(paragraph);
        paragraph.addEventListener('mouseover', function() {
            get_classes(count)
                paragraph.style.background = '#a1aab3'; 
                paragraph.style.cursor = 'grab'; 
        })
        paragraph.addEventListener('click', function() {
            // let xx = total_points[count][0][0]
            // let yy = total_points[count][0][1]
            // let cls_ = tol_get_class_value[count]
            click_cls = count
            get_class2.value = tol_get_class_value[click_cls]
            showDialogAtPosition_2(window.innerWidth*0.5,window.innerHeight*0.5);
        })
      }
      function get_classes(count){
        retain_total_point(count,1)
      }
}
///--------------------------------------------更改標記好的dialog
var myDialog2 = document.getElementById('myDialog2')
function showDialogAtPosition_2(x, y) {
    myDialog2.style.left = x + 'px';
    myDialog2.style.top = y + 'px';
    myDialog2.showModal();
}
function add_cls_list_2(cls_name) {
    let paragraph = document.createElement("p");
    paragraph.textContent = cls_name;
    paragraph.id = 'cls2_'+ cls_name;
    paragraph.addEventListener('click', function() {
        
        get_cls2(this)
    })
    class_list2.appendChild(paragraph);
}
function get_cls2(element){
    get_class2.value = element.textContent
}
var ok_but2 = document.getElementById('ok_butt2');
var del_but = document.getElementById('del_butt');

ok_but2.addEventListener('click', function(event) {
    tol_get_class_value[click_cls] = get_class2.value
    jsonto_={
        imgname:img_temp_name,
        poi:total_points,
        cls:tol_get_class_value
        }
    json2thon(jsonto_)   
    retain_total_point()

});
del_but.addEventListener('click', function(event) {
    total_points.splice(click_cls, 1)
    tol_get_class_value.splice(click_cls, 1)

    jsonto_={
        imgname:img_temp_name,
        poi:total_points,
        cls:tol_get_class_value
        }
    json2thon(jsonto_)   
    retain_total_point()
});

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
                //total_points = response.point
                //tol_get_class_value = response.label
              return reader.result
        }catch (e){
            console.error('There was a problem with your fetch operation:', e);
            throw e
        }
    }
    /////////////////      判斷是否已有json

    var canvas = document.getElementById('myCanvas');
    var context = canvas.getContext('2d');
    var canvas_poly = document.getElementById('canvas_poly');
    var context_poly = canvas_poly.getContext('2d');
    var canvas_bg = document.getElementById('canvas_poly_all');
    var context_bg = canvas_bg.getContext('2d');
    var label_col_list = document.getElementById('dif_col_list')
    var obj_list = document.getElementById('obj_list')

    var get_class2 = document.querySelector('.input-class2');
    // var class_list2 = document.getElementById('class_list2');
    //let click_cls
    obj_list.innerHTML = '';
    await check_json(element.textContent)
    draw_label()
    change_canvas
    async function check_json(temp_i) {
        try{
            const url = '/json_temp' ;
            const response = await fetch(url, {
                method: 'POST', 
                headers: {
                    'Content-Type': 'application/json' 
                },
                body: JSON.stringify({temp: temp_i })    
            });
            const data = await response.json();
            total_points = data.point
            tol_get_class_value = data.label
            console.log(tol_get_class_value)
        }catch (e){
            console.error('There was a problem with your fetch operation:', e);
            throw e
        }
    }
    function draw_label(){
        const width = image_t.clientWidth;
        const height = image_t.clientHeight;
        canvas_bg.width = width;
        canvas_bg.height = height;
        if (total_points.length>0){
            retain_total_point();
        }
    }
        function retain_total_point(pointer=null,ifc=0){
            context_bg.clearRect(0, 0, canvas_bg.width, canvas_bg.height);
            for (let i = 0; i < total_points.length; i++) {
                draw_done_point(total_points[i],context_bg);
                if (i==pointer){
                    drawPolygon_bg(total_points[i],context_bg,tol_get_class_value[i],pointer,ifc);
                }
                else{
                    drawPolygon_bg(total_points[i],context_bg,tol_get_class_value[i],null,ifc);
                }
        
            }
        }
        function draw_done_point(poi,ctx_x){
            for (var i = 0; i < poi.length; i++) {
                ctx_x.fillStyle = 'rgba(255,165,0,0.5)';
                ctx_x.beginPath();
                ctx_x.arc(poi[i][0], poi[i][1], 5, 0, Math.PI * 2);
                ctx_x.fill();
            }
        }
        
        function drawPolygon_bg(poi,ctx_x,cls,pointer=null,ifc=0) {

            if (cls in tol_color){
                if (pointer!=null){
                    let now_alpha_color= tol_color[cls]
                    now_alpha_color = now_alpha_color.replace(/(\d?\.?\d+)(?=\s*\))/, '0.7');
                    ctx_x.fillStyle = now_alpha_color
                }
                else{
                    ctx_x.fillStyle = tol_color[cls];
                }
            }
            else{
                let col = getRandomRGBAColor();
                tol_color[cls] = col;
                ctx_x.fillStyle = col;
                change_cls_col_list()
            }
            if (ifc==0){each_cls_col_list()}
            ctx_x.beginPath();
            ctx_x.moveTo(poi[0][0], poi[0][1]);
            for (let i = 1; i < poi.length; i++) {
                ctx_x.lineTo(poi[i][0], poi[i][1]);
            }
            ctx_x.lineTo(poi[0][0], poi[0][1]);
            ctx_x.closePath();
            ctx_x.fill();
        }
        function getRandomRGBAColor(alpha=0.5) {
            const r = Math.floor(Math.random() * 256);
            const g = Math.floor(Math.random() * 256);
            const b = Math.floor(Math.random() * 256);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        }
        function change_cls_col_list(){
            label_col_list.innerHTML = '';
            for (var cls in tol_color) {
                add_cls_col_list(cls,tol_color[cls])
            }
            function add_cls_col_list(cls_name,col) {
                let paragraph = document.createElement("p");
                paragraph.id = 'col_'+ cls_name;
                // Creating color dot
                let dot = document.createElement("span");
                dot.classList.add("dot");
                dot.style.backgroundColor = col;
                paragraph.appendChild(dot);
                paragraph.innerHTML += cls_name;
            
                label_col_list.appendChild(paragraph);
              }
        }
        function each_cls_col_list(){
            obj_list.innerHTML = '';
            for (let i=0 ;i< tol_get_class_value.length;i++) {
                add_cls_dif_list(tol_get_class_value[i],tol_color[tol_get_class_value[i]],i);
            }
            function add_cls_dif_list(cls_name,col,count) {
                let paragraph = document.createElement("p");
                paragraph.id = 'cls_'+ count;
                // Creating color dot
                let dot = document.createElement("span");
                dot.classList.add("dot");
                dot.style.backgroundColor = col;
                paragraph.appendChild(dot);
                paragraph.innerHTML += cls_name;
                obj_list.appendChild(paragraph);
                // paragraph.addEventListener('mouseover', function() {
                //     get_classes(count)
                //         paragraph.style.background = '#a1aab3'; 
                //         paragraph.style.cursor = 'grab'; 
                // })
                // paragraph.addEventListener('click', function() {
                //     // let xx = total_points[count][0][0]
                //     // let yy = total_points[count][0][1]
                //     // let cls_ = tol_get_class_value[count]
                //     click_cls = count
                //     get_class2.value = tol_get_class_value[click_cls]
                //     showDialogAtPosition_2(window.innerWidth*0.5,window.innerHeight*0.5);
                // })
              }
              function get_classes(count){
                retain_total_point(count,1)
              }
        }
    // dialog
    var myDialog2 = document.getElementById('myDialog2')
    function showDialogAtPosition_2(x, y) {
        myDialog2.style.left = x + 'px';
        myDialog2.style.top = y + 'px';
        myDialog2.showModal();
    }
    function add_cls_list_2(cls_name) {
        let paragraph = document.createElement("p");
        paragraph.textContent = cls_name;
        paragraph.id = 'cls2_'+ cls_name;
        paragraph.addEventListener('click', function() {
            
            get_cls2(this)
        })
        class_list2.appendChild(paragraph);
    }
    function get_cls2(element){
        get_class2.value = element.textContent
    }
    // var ok_but2 = document.getElementById('ok_butt2');
    // var del_but = document.getElementById('del_butt');

    // ok_but2.addEventListener('click', function(event) {
    //     tol_get_class_value[click_cls] = get_class2.value
    //     jsonto_={
    //         imgname:img_temp_name,
    //         poi:total_points,
    //         cls:tol_get_class_value
    //         }
    //     json2thon(jsonto_)   
    //     retain_total_point()

    // });
    // del_but.addEventListener('click', function(event) {
    //     console.log(click_cls)
    //     total_points.splice(click_cls, 1)
    //     tol_get_class_value.splice(click_cls, 1)

    //     jsonto_={
    //         imgname:img_temp_name,
    //         poi:total_points,
    //         cls:tol_get_class_value
    //         }
    //     json2thon(jsonto_)   
    //     retain_total_point()
    // });
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
    
    ///////////////////////
    function change_canvas(){
        if (now_act == "brain"){
            const width = image_t.clientWidth;
            const height = image_t.clientHeight;
            canvas.width = width;
            canvas.height = height;
            context.clearRect(0, 0, canvas.width, canvas.height);
            context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height)
            canvas_poly.width=0;
            canvas_poly.height=0;
            //圖層移動
            canvas.style.zIndex = 100;
            canvas_poly.style.zIndex = 1;
            canvas_bg.style.zIndex = 2;
            canvas_click.style.zIndex = 3;
            canvas_none.style.zIndex = -1
            
        }
        if (now_act == "poly"){
            const width = image_t.clientWidth;
            const height = image_t.clientHeight;
            canvas_poly.width = width;
            canvas_poly.height = height;


            context.clearRect(0, 0, canvas.width, canvas.height);
            canvas.width = 0;
            canvas.height = 0;
            //圖層移動
            canvas.style.zIndex = 1;
            canvas_poly.style.zIndex = 100;
            canvas_bg.style.zIndex = 2;
            canvas_click.style.zIndex = 3;
            canvas_none.style.zIndex = -1
            
        }
        if (now_act == "clicks"){
            const width = image_t.clientWidth;
            const height = image_t.clientHeight;
            canvas_click.width = width;
            canvas_click.height = height;

            context.clearRect(0, 0, canvas.width, canvas.height);
            context.clearRect(0, 0, canvas.width, canvas.height);
            canvas.width = 0;
            canvas.height = 0;
            canvas_poly.width = 0;
            canvas_poly.height = 0;
            //圖層移動
            canvas.style.zIndex = 1;
            canvas_click.style.zIndex = 100;
            canvas_bg.style.zIndex = 2;
            canvas_poly.style.zIndex = 3;
            canvas_none.style.zIndex = -1
            
        }
        if (now_act == "modeify"){
            const width = image_t.clientWidth;
            const height = image_t.clientHeight;
            context.clearRect(0, 0, canvas.width, canvas.height);
            context_poly.clearRect(0, 0, canvas_poly.width, canvas_poly.height)
            canvas_none.width = width;
            canvas_none.height = height;
            canvas.width = 0;
            canvas.height = 0;
            canvas_poly.width = 0;
            canvas_poly.height = 0;
            //圖層移動
            canvas.style.zIndex = 1;
            canvas_poly.style.zIndex = 2;
            canvas_click.style.zIndex = 3;
            canvas_bg.style.zIndex = 100;
            canvas_none.style.zIndex = -1;
        }

    }
}
