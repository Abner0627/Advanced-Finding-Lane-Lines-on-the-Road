# 自動駕駛實務 道路邊線偵測 AutonomousDriving_LaneLinesDetect

## Github

[<img src=https://i.imgur.com/3aZfqpy.png width=15%>](https://github.com/Abner0627/Finding-Lane-Lines-on-the-Road)

https://github.com/Abner0627/Finding-Lane-Lines-on-the-Road

## 作業目標

偵測車輛行走時，該車道之兩側邊線。

## 影片素材

1. [Udacity Test Video - solidWhiteRight](https://github.com/udacity/CarND-LaneLines-P1/blob/master/test_videos/solidWhiteRight.mp4)
2. [Udacity Test Video - solidYellowLeft](https://github.com/udacity/CarND-LaneLines-P1/blob/master/test_videos/solidYellowLeft.mp4)
3. [Udacity Test Video - challenge](https://github.com/udacity/CarND-LaneLines-P1/blob/master/test_videos/challenge.mp4)
4. [國道一號 中山高速公路 北向 高雄-基隆 374K-0K 全程 路程景National Highway No. 1 - Youtube ](https://www.youtube.com/watch?v=0crwED4yhBA)

## 成果

* solidWhiteRight (https://youtu.be/Rk7PWRhAPRc)
    ![white](/img/Lane_solidWhiteRight_Trim.png)
    
* solidYellowLeft (https://youtu.be/UKnnKqDSB44)
    ![yellow](/img/Lane_solidYellowLeft_Trim.png)

* challenge (https://youtu.be/hv0lQcNjJhU)
    ![ch](/img/Lane_challenge_Trim.png)

* tw_NH1 (https://youtu.be/1OO51ng19L4)
    ![NH1](/img/Lane_tw_NH1_Trim.png)

## 作法

>主要參考自下列專案：
>[OanaGaskey / Advanced-Lane-Detection - github](https://github.com/OanaGaskey/Advanced-Lane-Detection)

### Step 0 環境設定與套件安裝

1. 使用環境：  
    * Win 10 
    * python 3.8.5

2. 進入該專案之資料夾
`cd /d [<YOUR PATH>/Practices-of-Autonomous-Driving/Finding_Lane_Lines_on_the_Road]`

3. 安裝所需套件
`pip install -r requirements.txt`

4. 執行主要程式
`python main.py -V [DATA NAME]`

    * [DATA NAME]依處理的影片檔名而定，目前僅有存於`./data`中的影片可供使用。 
        *  challenge
        *  solidWhiteRight (defailt)
        *  solidYellowLeft
        *  tw_NH1
    * 範例：`python main.py -V solidWhiteRight`

5. 確認輸出
   * 目前預設輸出資料夾為`./output`，檔名為`Lane_`+`影片名稱`。

### 程式碼註解

### Step 1 載入影片

``` python
vc = cv2.VideoCapture(os.path.join(P, V))  # 載入影片，P為影片路徑；V為其檔名
fps = vc.get(cv2.CAP_PROP_FPS)  # 計算fps
frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))  # 計算影格數
video = []  # 建立list用以存放處理好的影格
```

### Step 2 將影格轉為鳥瞰圖

為了更精準擷取道路邊線，此處將車內視角轉作鳥瞰圖，  
以利後續偵測線段時皆以平面之線段為主。

```py
'''
img  為影片之每幀影格
src  與dst為透視變換之端點，M為變換矩陣，依不同影片視角有所不同，詳見下方說明
'''
img_size = (img.shape[1], img.shape[0])
M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)
img_warped = cv2.warpPerspective(img, M, img_size) 
```

<img src=https://i.imgur.com/tgfV7S6.png>

>src為下圖梯形區域（黃色區域）中四端點的座標；
>dst則為下圖目標投影區域（藍色框線）的四端點座標
><img src=https://i.imgur.com/00VzS1m.png width=60%>

### Step 3 將圖片二元化後進行邊緣偵測

```py
# 灰階化
gray_img =cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)  
# 使用Sobel濾波，過濾x方向紋路
sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)  
abs_sobelx = np.absolute(sobelx)
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
# 將Sobel濾波結果二元化
sx_binary = np.zeros_like(scaled_sobel)  
sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1
# 將灰階圖二元化
white_binary = np.zeros_like(gray_img)  
white_binary[(gray_img > 150) & (gray_img <= 255)] = 1
# 結合上述結果進行 OR 運算
binary_warped = cv2.bitwise_or(sx_binary, white_binary)  
```

<img src=https://i.imgur.com/S0fdLPo.png width=60%>

### Step 4 偵側車道

對上圖依y方向疊加後，利用峰值找出車道邊線x座標。

```py
histogram = np.sum(binary_warped, axis=0)

midpoint = int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])  # 找出左邊車道之峰值
rightx_base = np.argmax(histogram[midpoint:]) + midpoint  # 找出右邊車道之峰值
laneBase = [leftx_base, rightx_base]
```

<img src=https://i.imgur.com/ygMAndt.png width=60%>

接著由上而下，將圖片切割為數個區塊

```py
'''
nwindows  為區塊數量
margin    為區塊寬度
minpixel  為判斷該區塊之像素點數量是否用於擬合之閥值
'''
# 初始化區塊高度以及用於擬合之x、y變數
window_height = np.int32(binary_warped.shape[0]//nwindows)
laneLine_y = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
laneLine_x = np.ones([4, binary_warped.shape[0]]) * -10
# 初始化用於畫上擬合線的陣列
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
line_img = np.zeros_like(out_img)

for n_lane in range(len(laneBase)): 
    x_point = []
    y_point = []
    laneCurrent = laneBase[n_lane]  # 目前車道邊線之x座標
    for n_window in range(nwindows):
        # 判斷區塊之x方向範圍        
        x_range_L = laneCurrent - margin
        x_range_R = laneCurrent + margin
        if x_range_L < 0:
            x_range_L = 0
        if x_range_R >= binary_warped.shape[1]:
            x_range_R = binary_warped.shape[1] - 1
        # 判斷區塊之y方向範圍
        y_range_T = binary_warped.shape[0] - (n_window+1)*window_height
        y_range_B = binary_warped.shape[0] - n_window*window_height
        # 擷取區塊
        window = binary_warped[y_range_T:y_range_B, x_range_L:x_range_R]
        # 取得區塊中非像素點之座標
        y_Nz, x_Nz = np.nonzero(window)
        x_Nz = x_Nz + x_range_L
        y_Nz = y_Nz + y_range_B
        # 若像素點數量大於閥值則將其儲存
        if np.count_nonzero(window) > minpixel:
            x_point.extend(x_Nz)
            y_point.extend(y_Nz)
            # 更新目前車道邊線之x座標
            laneCurrent = np.mean(x_Nz, axis=0, dtype=np.int32
    # 進行擬合
    if len(y_point) > 0:     
        fit = np.polyfit(y_point, x_point, 2)
        laneLine_x[n_lane, :] = fit[0] * laneLine_y**2 + fit[1] * laneLine_y + fit[2]            
```

<img src=https://i.imgur.com/9VPKPC9.png width=60%>

### Step 5 劃出車道邊線

```py
'''
width     為邊線寬度
threshold 為可容許之最大邊線間隔
'''
for line_x in laneLine_x:
    if np.abs(line_x[-1]-line_x[0]) > threshold:
        continue
    if np.abs(line_x[-1] - line_x[len(line_x)//2]) > threshold:
        continue
    if np.abs(line_x[0] - line_x[len(line_x)//2]) > threshold:
        continue
    # 邊線左邊界
    lineWindow1 = np.expand_dims(np.vstack([line_x - width, laneLine_y]).T, axis=0)
    # 邊線右邊界
    lineWindow2 = np.expand_dims(np.flipud(np.vstack([line_x + width, laneLine_y]).T), axis=0)
    linePts = np.hstack((lineWindow1, lineWindow2))
    # 填上曲線間區域
    cv2.fillPoly(line_img, np.int32([linePts]), (255,0,0))
```

<img src=https://i.imgur.com/fr5X40q.png width=60%>

### Step 6 將車道線疊合

```py
# 將上圖曲線再轉移至原視角
weight = cv2.warpPerspective(line_img, M_inv, (img.shape[1], img.shape[0]))
# 調整RGB並疊圖
weight = weight[:,:,[2,1,0]]
result = cv2.addWeighted(img, 1, weight, 1, 0)
# 將結果加入list中
height, width, layers = result.shape
size = (width, height)
video.append(result)
vc.release()
# 使用OpenCV儲存影片
out = cv2.VideoWriter(os.path.join(sP, sV), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(video)):
    out.write(video[i])
out.release()
```

<img src=https://i.imgur.com/nPh888r.png>
