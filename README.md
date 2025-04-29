```
pip install --upgrade pip
pip install setuptools typeguard jinja2 pyyaml
pip install opencv-python opencv-contrib-python matplotlib numpy
```

## Calibration
```
python camera_calibration.py --size 8x6 --square 0.025 --path "./frames" --output "./camera_params/camera_parameter.yaml"
```

`config.py`에서 설정값 변경

## UDP 송신
```
# 캘리브레이션 왜곡 보정 O
python run_udp_sender.py --target_ip <> --port <>

# 캘리브레이션 왜곡 보정 X
python run_udp_sender.py --no_undistort
```

## UDP 수신
```
python run_udp_receiver.py
```


## USB 카메라
```
# 캘리브레이션 왜곡 보정 O
python run_usb_streamer.py --camera 2

# 캘리브레이션 왜곡 보정 X
python run_usb_streamer.py --camera 2 --no_undistort
```
