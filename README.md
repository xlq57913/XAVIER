# XAVIER
The Collection of some repo/codes worked on Xavier

# Version
|  环境   | 版本   |
|  ----  | ----  |
|ubuntu  | 18.04 |
|jetPack | 4.4   |
|cuda    | 10.2  |
|cuDNN   |  8.0  |
|tensorRT| 7.1   |
|openCV  | 4.1.1 |
|python3 | 3.6.9 |
|python2 | 2.7.17|
|pyTorch | 1.7.0 |
|torchVision| 2.2.0 |

# WIFI setup
run the following commands to enable the WIFI module (after restart), sometimes the first one will work.

```shell
sudo usb_modeswitch −KW −v 0bda −p 1a2b
sudo usb_modeswitch −KW −v 0bda −p b711
```
# Before Working
run the command below to turn on the fan 

```shell
sudo nvpmodel -m 0
sudo jetson_clocks
```
# Blog
* [刷机](https://blog.csdn.net/qq_38679413/article/details/109398853)
* [pytorch](https://www.jianshu.com/p/9e9c74834283)

# Repo
* [rtl8821cu](https://github.com/whitebatman2/rtl8821CU)
* [rtl8188gu](https://github.com/McMCCRU/rtl8188gu)
