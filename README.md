##啊啊啊啊啊啊
* 这是我第一个上传到的github项目
* 欢迎大家下载，点赞，支持
* 有点小问题请不要在意（比如把video写成了vedio   嘤嘤嘤）
  
##有友友不会下载
* 点击右上角绿色的Code，再点击download下载压缩包
* 下载好解压就可以看到了嘿嘿🤤🤤🤤

##捐助开发者
* 在兴趣的驱动下,写一个`免费`的东西
* 上传到抖音的视频火了，我也高兴呢。有能力的可以请我喝一瓶`小甜水`吗？谢谢
* `收款码`在文件里，如果可以请一瓶小甜水的话，我会开心好久呢嘿嘿🤤🤤🤤![可以的话希望支持一下_谢谢_不需要太多_就当是对本学生一点小小的鼓励](https://github.com/user-attachments/assets/bd29ea5a-8fa1-49e7-b519-aecf8227999e)
![可以的话希望支持一下_谢谢_不需要太多_就当是对本学生一点小小的鼓励](https://github.com/user-attachments/assets/d23e0273-f5a7-420e-bef4-4603f0000aff)

##感激

##运行代码
* 运行代码需要一定的基础，比如说pip install某一些库,配置文件路径之类的。
* 好好学习，多多益善。
  
##关于作者

```javascript
var ihubo = {
  抖音账号  : "火锅想喝雪碧",
  抖音账号 : "2203796229"
}
```

##视频主要制作流程为：
* ①视频提取出每一帧，保存
* ②对保存的的每一帧做灰度处理，保存
* ③对灰度图像进行边缘检测，保存。。。甚至可以进行膨胀处理
* ④对边缘图片进行裁剪（手机录屏缘故，上下有较多黑色区域需要去除）
* ⑤生成目标画幅的白色图像（与裁剪后的图片成比例）
* ⑥将图片0和图片1贴到白色图像上。对边缘图像进行识别，有白色，对应区域贴1.没白色（不是边缘区域）贴0
* ⑦将01图像连起来，生成视频

##代码解释
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
##运行流程
* 375-391行，就是运行流程。
* 先运行vedio_handle1()，
* 再运行find_up_down_edge()。其中系数，vedio_up_long为视频长边长度。vedio_find_edge_short，根据肉眼判断，找一个短边合适的值放进去
* 看运行结果。对照代码读懂（相信不难理解输出的东西。1234是对应图片，其他数字就是出现255的地方。找到最大值和最小值，进行适当取值，就是cut1和cut2）
* 
* 再运行vedio_handle2，一步到位，生成01贴图，连接成视频。
* picture_edge_to_01()和picture_to_veido()是调试用的

##关于作者

```javascript
var ihubo = {
  抖音账号  : "火锅想喝雪碧",
  抖音账号 : "2203796229"
}
```
