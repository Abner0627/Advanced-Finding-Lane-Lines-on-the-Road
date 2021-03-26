# 自動駕駛實務 道路邊線偵測 AutonomousDriving_LaneLinesDetect

## Github

[<img src=https://i.imgur.com/3aZfqpy.png width=15%>](https://github.com/Abner0627/Practices-of-Autonomous-Driving/tree/main/Finding_Lane_Lines_on_the_Road)

https://github.com/Abner0627/Practices-of-Autonomous-Driving/tree/main/Finding_Lane_Lines_on_the_Road

## 作業目標

偵測車輛行走時，該車道之兩側邊線。

## 影片素材

1. [Udacity Test Video - solidWhiteRight](https://github.com/udacity/CarND-LaneLines-P1/blob/master/test_videos/solidWhiteRight.mp4)
2. [Udacity Test Video - solidYellowLeft](https://github.com/udacity/CarND-LaneLines-P1/blob/master/test_videos/solidYellowLeft.mp4)
3. [Udacity Test Video - challenge](https://github.com/udacity/CarND-LaneLines-P1/blob/master/test_videos/challenge.mp4)
4. [國道一號 中山高速公路 北向 高雄-基隆 374K-0K 全程 路程景National Highway No. 1 - Youtube ](https://www.youtube.com/watch?v=0crwED4yhBA)

## 成果

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
img為影片之每幀影格
src與dst為透視變換之端點，M為變換矩陣，依不同影片視角有所不同，詳見下方說明
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
gray_img =cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)  # 灰階化
sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)  # 使用Sobel濾波，過濾x方向紋路
abs_sobelx = np.absolute(sobelx)
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

sx_binary = np.zeros_like(scaled_sobel)  # 將Sobel濾波結果二元化
sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1

white_binary = np.zeros_like(gray_img)  # 將灰階圖二元化
white_binary[(gray_img > 150) & (gray_img <= 255)] = 1

binary_warped = cv2.bitwise_or(sx_binary, white_binary)  
# 結合上述結果進行 OR 運算
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


