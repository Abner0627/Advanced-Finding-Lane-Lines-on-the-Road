# 自動駕駛實務 道路邊線偵測 AutonomousDriving_LaneLinesDetect

## Github

[<img src=https://i.imgur.com/3aZfqpy.png width=25% />](https://github.com/Abner0627/Practices-of-Autonomous-Driving/tree/main/Finding_Lane_Lines_on_the_Road)

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
vc = cv2.VideoCapture(os.path.join(P, V)) 
# 載入影片，P為影片路徑；V為其檔名
fps = vc.get(cv2.CAP_PROP_FPS)
# 計算fps
frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
# 計算影格數
video = []
# 建立list用以存放處理好的影格
```
