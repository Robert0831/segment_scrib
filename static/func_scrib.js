var img_temp_name 
var img_temp_list

document.addEventListener("DOMContentLoaded", () => {
// 获取图片元素和div元素
var canvas = document.getElementById('myCanvas');
var context = canvas.getContext('2d');
var scribArea = document.getElementById('scrib_area_');
var image = document.getElementById('scrib_image');
var container = document.querySelector('.scrib_area_container');
//縮放圖片

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

function increaseLineWidth() {
    lineWidth += 1; // 增加线条粗细
}

// 减少粗细的函数
function decreaseLineWidth() {
    if (lineWidth > 1) { // 防止粗细小于1
        lineWidth -= 1; // 减少线条粗细
    }
}
// 添加事件監聽器
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
canvas.addEventListener('contextmenu', event => event.preventDefault());



document.getElementById("brain").addEventListener("click", function() {
    // var width = scribArea.clientWidth;
    // var height = scribArea.clientHeight;
    var width = image.clientWidth;
    var height = image.clientHeight;
    // console.log(width)
    // console.log(height)
    canvas.width = width;
    canvas.height = height;
});
var scrib_pn = 1
document.getElementById("pos_scrib").addEventListener("click", function() {
    scrib_pn=1
});
document.getElementById("neg_scrib").addEventListener("click", function() {
    scrib_pn=0
});
document.getElementById("clear_scrib").addEventListener("click", function() {
    canvas.width = image.clientWidth;
    canvas.height = image.clientHeight;
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

    