openvino环境配置。下载别下载最新版的2021，下载2020，3 稳定版
需要设置好多环境变量
运行C:\Program Files (x86)\IntelSWTools\openvino_2020.3.341\deployment_tools\model_optimizer\install_prerequisites 时有个path警告，也要放进去。搞了2天终于搞通了
视频教程 https://www.bilibili.com/video/BV1Av411676M?p=4

~~又有新的发现。好多库不支持3.5，用3.8，~~
numpy 1.19.4 有bug，要回退1.19.3

py版本众多，ml只能用64位。
~~py3.5是个不错的版本~~
把源换成国内的
如果报错，看下是否timeout。是的话重新安装
tf比较大，很容易timeout

~~一定要用32位的py3.7~~
然后安装opencv-python，官方版的不好用

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_table_of_contents_calib3d/py_table_of_contents_calib3d.html#py-table-of-content-calib


OPENVINO TensorRT libtorch tvm TFlite 移动端的有TVM、NCNN、MNN、TNN
