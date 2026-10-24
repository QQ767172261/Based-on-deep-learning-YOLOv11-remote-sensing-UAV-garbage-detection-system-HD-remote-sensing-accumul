
基于深度学习YOLOv11遥感无人机垃圾杂物检测系统 高清遥感堆积垃圾杂物检测数据集｜1996张，YOLO标注，YOLOv11n源码+PyQt界面   高清遥感图像堆积垃圾及杂物检测数据集，1996张，yolo，voc，coco三种标注方式
图像尺寸: 1202 * 2400
总类别数量: 1 类
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ac91a09b07884198b3b7f9dfb86e396c.png)
# 高清遥感图像堆积垃圾及杂物检测数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/076f69d7bfa243809407f0635fabff6d.png)




| 项目 | 详情 |
| --- | --- |
| 数据集名称 | 高清遥感图像堆积垃圾及杂物检测数据集 |
| 图像总量 | 1996张 |
| 图像分辨率 | 1202 × 2400 |
| 标注格式 | YOLO、VOC、COCO三种标注格式 |
| 类别总数 | 1类 |
| 类别名称 | DJZW - 堆积垃圾及杂物：训练图1197张，标签3345；验证图399张，标签1081；测试图400张，标签1136 |
| 数据集划分 | 训练集：1197张   验证集：399张   测试集：400张 |
| 训练模型 | YOLOv11n，训练轮次80epoch |
| 交付内容 | 训练权重、完整训练+测试源码、PyQt5可视化界面全套源码（.ui/.qrc/.py/图标） |
| 界面功能 | 图片检测、视频检测、摄像头实时检测；输出目标框、目标数量、置信度；检测结果导出保存，免命令行操作 |
| 运行环境 | Python3.8，opencv-python、PyQt5、torch；支持Windows、Linux，提供零基础CPU环境部署包 |

## 应用价值![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9ab6dab518d34007b5a2c7c782ccd624.png)


1. **遥感固废自动排查**：利用高分遥感影像自动识别露天堆积垃圾、杂物，助力非正规垃圾堆放点遥感筛查，替代人工目视解译。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4e0eacfbedbf425e800639b347464600.png)

2. **生态环保督查**：适用于环保、自然资源部门的固废排查、卫片执法、区域环境巡查，快速定位违规堆放点位。
3. **批量遥感影像自动化处理**：PyQt可视化工具降低算法使用门槛，一线工作人员无需代码基础即可批量处理遥感图片。
4. **算法验证与科研**：适合遥感小目标检测、地表异物识别相关课题，用于YOLO系列遥感目标检测算法调优实验。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6c7230a40b2a45e8a715fc380b2e3d5e.png)

## 适用群体

- 生态环境局、自然资源局、环卫巡查部门工作人员
- 遥感测绘、环保调查项目服务商、航测数据处理团队
- 固废智能监测、遥感目标检测算法研发人员
- 高校遥感、环境AI方向科研课题组

## 简易代码![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/877c0a16d9024c78808359dd64592e46.png)


### 1. 训练脚本 train.py

```
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.yaml")
    model.train(
        data="data.yaml",
        epochs=80,
        imgsz=(1202,2400),
        device="cpu",
        batch=4,
        name="garbage_detect"
    )
```

`data.yaml`

```
path: ./dataset
train: images/train
val: images/val
test: images/test

names:
  0: DJZW-堆积垃圾及杂物
```

### 2. 单图推理脚本 predict.py![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5b6db7bf12284cde80ebafefcc92b2d3.png)


```
from ultralytics import YOLO

model = YOLO("runs/detect/garbage_detect/weights/best.pt")
res = model.predict("test.jpg", save=True, conf=0.3)
for r in res:
    print(f"检测目标数量：{len(r.boxes)}")
```

### 3. PyQt简易推理界面 main.py![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e54e0b61348a48abb3c97423e7804d18.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cf3cae0067ec4756b7344f08b720abb9.png)

```
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from ultralytics import YOLO

class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("遥感堆积垃圾检测")
        self.resize(800,600)
        self.model = YOLO("best.pt")
        self.btn = QPushButton("选择图片检测",self)
        self.btn.clicked.connect(self.detect_img)
        self.label = QLabel("结果显示区",self)
        self.label.setGeometry(50,100,700,450)

    def detect_img(self):
        path,_ = QFileDialog.getOpenFileName()
        if not path:
            return
        res = self.model.predict(path,save=True)
        cnt = len(res[0].boxes)
        self.label.setText(f"识别完成，堆积垃圾杂物总数：{cnt}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWin()
    w.show()
    sys.exit(app.exec_())
```


摘要：
本高分遥感堆积垃圾杂物检测数据集共1996张1202×2400遥感影像，单类别标注堆积垃圾及杂物，三种标注格式齐全，划分训练验证测试集。采用YOLOv11n训练80轮，配套训练权重、全套源码以及PyQt可视化检测界面，支持图片、视频、摄像头实时检测，支持结果导出，适配Windows/Linux，零基础可部署。用于遥感固废排查、环保卫片执法、露天垃圾堆放点智能识别。


