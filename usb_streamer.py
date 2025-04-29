import os
import cv2
import time
import datetime
import config
from camera_handler import CameraHandler

class UsbStreamer:
    """
    USB 카메라를 사용하여 로컬 화면에 스트리밍하는 클래스.
    """
    def __init__(self, camera_index=config.USB_CAMERA_INDEX,
                 width=640, height=480, fps=30,
                 calibration_file=None, no_undistort=False,
                 window_name=None, save_dir="captured_frames"):
        """
        USB 스트리머 초기화.

        Args:
            camera_index (int): 카메라 인덱스.
            width (int): 카메라 프레임 너비.
            height (int): 카메라 프레임 높이.
            fps (int): 목표 초당 프레임 수.
            calibration_file (str, optional): 카메라 캘리브레이션 YAML 파일 경로.
            no_undistort (bool): True면 왜곡 보정 안 함.
            window_name (str, optional): OpenCV 창 제목. 기본값은 "USB Camera [index]".
        """
        self.fps = fps
        self.no_undistort = no_undistort
        self.window_name = window_name if window_name else f"USB Camera {camera_index}"
        self.running = False
        self.cam_handler = None
        self._display_fps = 0
        self._frame_count = 0
        self._start_time = time.time()

        self.save_dir = save_dir
        if self.save_dir and not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
                print(f"[정보] 프레임 저장 디렉토리 생성: {self.save_dir}")
            except OSError as e:
                print(f"[오류] 저장 디렉토리 생성 실패 ({self.save_dir}): {e}")
                self.save_dir = None
        elif not self.save_dir:
            self.save_dir = None

        try:
            # 카메라 핸들러 초기화
            calib_file = calibration_file if not no_undistort else None
            self.cam_handler = CameraHandler(
                index=camera_index,
                width=width, height=height, fps=fps, buffer_size=1,
                calibration_file=calib_file
            )
            # OpenCV 창 생성
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, width, height)

        except IOError as e:
            print(f"[ERROR] 카메라 초기화 실패: {e}")
            raise
        except Exception as e:
            print(f"[ERROR] 초기화 중 예기치 않은 ERROR: {e}")
            raise

    def _calculate_fps(self):
        """FPS 계산."""
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._start_time
        if elapsed >= 1.0:
            self._display_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._start_time = now

    def display_loop(self):
        """로컬 디스플레이 루프."""
        if not self.running:
             print("[ERROR] 디스플레이 루프 시작 불가: 실행 플래그 비활성.")
             return

        target_interval = 1.0 / self.fps if self.fps > 0 else 0
        last_capture_time = time.time()

        save_info = f"'{self.save_dir}'에 저장" if self.save_dir else "저장 비활성화됨"
        print(f"[INFO] 로컬 USB 카메라 ({self.cam_handler.cap.getBackendName()}) 스트리밍 시작.")
        print("[INFO] 종료하려면 'q' 키를 누르거나 창을 닫으세요.")
        print(f"[INFO] 원본 프레임 저장: 's' ({save_info})")

        while self.running:
            current_time = time.time()
            wait_time = target_interval - (current_time - last_capture_time)
            if target_interval > 0 and wait_time > 0.001:
                time.sleep(wait_time)
                last_capture_time = time.time()
            else:
                last_capture_time = current_time # 목표 시간 지났으면 시간 업데이트


            # 프레임 캡처
            frame = self.cam_handler.capture_frame(undistort=(not self.no_undistort))

            if frame is None:
                print("[WARN] 프레임 캡처 실패.")
                time.sleep(0.1)
                continue

            original_frame = frame.copy()

            # FPS 계산 및 표시
            self._calculate_fps()
            cv2.putText(frame, f"Display FPS: {self._display_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 화면 표시
            cv2.imshow(self.window_name, frame)

            # 키 입력 및 창 상태 확인
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] 'q' 키 입력 감지. 종료합니다.")
                self.running = False
                break
            elif key == ord('s'):  # 's' 키 처리 추가
                if original_frame is not None and self.save_dir:
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # 밀리초까지
                    filename = os.path.join(self.save_dir, f'usb_frame_{timestamp}.jpg')
                    try:
                        cv2.imwrite(filename, original_frame)  # 원본 프레임 저장
                        print(f"[정보] 원본 프레임 저장됨: {filename}")
                    except Exception as e:
                        print(f"[오류] 프레임 저장 실패 ({filename}): {e}")
                elif self.save_dir is None:
                    print("[경고] 저장 디렉토리가 설정되지 않아 저장할 수 없습니다.")
                else:
                    print("[경고] 저장할 원본 프레임이 없습니다 (캡처 실패 등).")
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] 디스플레이 창이 닫혔습니다. 종료합니다.")
                self.running = False
                break

        print("[INFO] 로컬 디스플레이 루프 종료됨.")
        self.stop_display()

    def start_display(self):
        """로컬 디스플레이 시작."""
        if self.running:
            print("[WARN] 이미 디스플레이가 실행 중입니다.")
            return
        if self.cam_handler is None:
            print("[ERROR] 카메라 핸들러가 초기화되지 않았습니다.")
            return

        self.running = True
        self.display_loop()

    def stop_display(self):
        """카메라 및 OpenCV 창 자원 해제."""
        print("[INFO] USB 스트리머 리소스 정리 시도...")
        if self.cam_handler:
            self.cam_handler.release_camera()
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except Exception as e:
            print(f"[WARN] OpenCV 창 닫기 중 ERROR: {e}")
        self.running = False
        print("[INFO] USB 스트리머 중지 완료.")

