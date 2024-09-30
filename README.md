##刚才新上传的`A_video_01_new.py`，友友们下载好文件，`把361行那个root_path路径改成自己的就ok了（下载里面有一个zhang_xue_liang的文件夹，文件夹里有一个mp4.就是这个文件夹的路径。尽量不要包含有中文），就可以直接运行出结果`

##A_video_01.py是旧的文件，A_video_01_Abcd.py是升级版的。`理论上把A_video_01_new.py的361-403行复制粘贴到A_video_01_Abcd.py的373-415位置，A_video_01_Abcd.py就可以直接运行`

##啊啊啊啊啊啊
* 这是我第一个上传到的github项目
* 欢迎大家下载，点赞，支持，可以的话帮忙点一下小星星🤩，谢谢
* 有点小问题请不要在意（比如把video写成了vedio   嘤嘤嘤）

## 日志
* 该项目首次完成是在2024.9.28晚上十一点，发布了一个简略的抖音
* 在2024.9.28晚上十一点半，改进了一下代码，在后续的半个小时内发布了一个抖音，在短短48小时收到了一百多万播放量，六万多的点赞，受宠若惊了有点
* 不少友友要代码，于是学习在github上创建项目，与2024.9.29中午一点上传了一个简略的项目，后续不断地往里填东西哈哈哈
* 现在是2024.9.30中午11:45！！！啊啊啊啊啊啊，昨晚有个新点子（实现效果见A_vedio_Abcd那个）。今早还有早八nnd舍友都回家了就剩我们两个大大冤种
* 时间来不及了，先把新项目上传，慢慢研究吧
* 提一嘴，主要需要下载的文件就是那几个jpg，png。还有那三个文件夹，以及py文件。mp4文件流量少的话就不用下载了
* 刚才新上传的`A_video_01_new.py`，友友们下载好文件，`把361行那个root_path路径改成自己的就ok了，就可以直接运行出结果`
* A_video_01.py是旧的文件，A_video_01_Abcd.py是升级版的。`理论上把A_video_01_new.py的361-403行复制粘贴到A_video_01_Abcd.py的373-415位置，A_video_01_Abcd.py就可以直接运行`
* A_video_01_Abcd.py的284和286中，white和black位置可以互换，友友们自行尝试
* 可以请我喝一瓶小甜水吗，我会特别开心的哈哈哈哈哈哈收款码在最底下😘
* 多余文件来不及删了，不说了，赶车去了，友友们加油！
  
  
## 有友友不会下载
* 点击右上角绿色的Code，再点击download下载压缩包
* 下载好解压就可以看到了嘿嘿🤤🤤🤤

## 运行代码
* 运行代码需要一定的基础，比如说pip install某一些库,配置文件路径之类的。
* 好好学习，多多益善。

## 视频主要制作流程为：
* ①视频提取出每一帧，保存
* ②对保存的的每一帧做灰度处理，保存
* ③对灰度图像进行边缘检测，保存。。。甚至可以进行膨胀处理
* ④对边缘图片进行裁剪（手机录屏缘故，上下有较多黑色区域需要去除）
* ⑤生成目标画幅的白色图像（与裁剪后的图片成比例）
* ⑥将图片0和图片1贴到白色图像上。对边缘图像进行识别，有白色，对应区域贴1.没白色（不是边缘区域）贴0
* ⑦将01图像连起来，生成视频

## 代码解释
* 主要运行代码在350-389行
* 修改根目录root_path---也是视频所在文件夹
* 修改vedio_name---视频名称
* 
* cut1和cut2。因为我是用手机录制的视频，然后上传到电脑，所以画幅是较长的，需要进行裁剪
* 裁剪位置就是cut1和cut2---比如视频画幅是592*1280，而视频主要部分上下都有较大黑色（手机录制缘故）
* 所以需要将cut1位置以上部分和cut2位置以下部分裁去。这两个参数在第二步！！使用。
* 
* create_picture_size和scale，指的是新生成的01视频的画幅，和裁剪后的照片到01视频画幅的缩放系数
* 比如裁剪后，照片画幅为592*1020，01图片的大小是10*15.希望生成的画幅为590*1020.
* 那么短边能放置590/10=59个01。长边能放置1020/15=68个01。
* 590/59=10.    1020/68=15.
* 那么缩放比例scale就是10*15
* 可以对create_picture_size和scale进行乘/除相同的系数，以生成更精细的视频
* （代码里的注释可能有误，多多包涵）
* 
## 运行流程
* 375-391行，就是运行流程。
* 先运行vedio_handle1()，
* 再运行find_up_down_edge()。其中系数，vedio_up_long为视频长边长度。vedio_find_edge_short，根据肉眼判断，找一个短边合适的值放进去
* 看运行结果。对照代码读懂（相信不难理解输出的东西。1234是对应图片，其他数字就是出现255的地方。找到最大值和最小值，进行适当取值，就是cut1和cut2）
* 
* 再运行vedio_handle2，一步到位，生成01贴图，连接成视频。
* picture_edge_to_01()和picture_to_veido()是调试用的

## 关于作者

```javascript
var ihubo = {
  抖音账号  : "火锅想喝雪碧",
  抖音账号 : "2203796229"
}
```
## 捐助开发者
* 在兴趣的驱动下,写一个`免费`的东西
* 上传到抖音的视频火了，我也高兴呢。有能力的可以请我喝一瓶`小甜水`吗？谢谢
* `收款码`在文件里，如果可以请一瓶小甜水的话，我会开心好久呢嘿嘿🤤🤤🤤![ecac975467493cd8520c0e4bdc99b6b](https://github.com/user-attachments/assets/c660f324-a13a-4aa9-8380-ec5b67505d63)
