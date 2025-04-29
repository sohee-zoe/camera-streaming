import cv2
import time
import numpy as np
from calibration_utils import load_calibration_from_yaml

class CameraHandler:
    def __init__(self, index, width, height, fps, buffer_size,
                 calibration_file=None):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise IOError(f"카메라 인덱스 {index}를 열 수 없습니다.")

        # --- 카메라 속성 설정 및 확인 ---
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_buffer = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)

        print(f"카메라 초기화 완료: 요청({width}x{height} @ {fps}FPS, 버퍼 {buffer_size}) -> "
              f"실제({self.actual_width}x{self.actual_height} @ {actual_fps}FPS, 버퍼 {actual_buffer})")
        # --- ---

        # --- 왜곡 보정 변수 초기화 ---
        self.K = None
        self.D = None
        self.new_K = None
        self.mapx = None
        self.mapy = None
        self.undistort_enabled = False
        self.calibration_image_width = None
        self.calibration_image_height = None
        # --- ---

        # --- 캘리브레이션 파일 처리 ---
        if calibration_file:
            # YAML 파일 로드 시도
            self.K, self.D, self.calibration_image_width, self.calibration_image_height = (
                load_calibration_from_yaml(calibration_file))

            # K, D 로드 성공 시
            if self.K is not None and self.D is not None:
                # 보정 맵 계산에 사용할 이미지 크기 결정
                img_width_for_calib = self.calibration_image_width if self.calibration_image_width else self.actual_width
                img_height_for_calib = self.calibration_image_height if self.calibration_image_height else self.actual_height

                img_size = (img_width_for_calib, img_height_for_calib)
                target_size = (self.actual_width, self.actual_height)

                print(f"[INFO] 왜곡 보정 매핑 계산 중 (원본 크기: {img_size}, 목표 크기: {target_size})")

                try:
                    # 최적의 새 카메라 매트릭스 및 보정 맵 계산
                    # alpha = 1 # 모든 픽셀 유지
                    alpha = 0  # crop
                    self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, img_size, alpha, target_size)
                    self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                        self.K, self.D, None, self.new_K, target_size, cv2.CV_32FC1)

                    # 맵 생성 성공 시 보정 활성화
                    if self.mapx is not None and self.mapy is not None:
                        self.undistort_enabled = True
                        print("[INFO] 카메라 왜곡 보정 활성화됨.")
                    else:
                        print("[WARN] 왜곡 보정 맵(mapx, mapy) 생성 실패. 보정 비활성화.")
                except Exception as e:
                    print(f"[ERROR] 왜곡 보정 맵 계산 중 오류 발생: {e}. 보정 비활성화.")
                    self.undistort_enabled = False # 오류 시 명시적 비활성화

            # K, D 로드 실패 시 (파일은 있었으나 내용이 문제)
            else:
                print(f"[WARN] YAML 캘리브레이션 데이터 로드 실패 ({calibration_file}). 왜곡 보정 비활성화.")
        # calibration_file 자체가 제공되지 않았을 때 (if calibration_file: 의 else)
        else:
            print("[INFO] 캘리브레이션 파일이 지정되지 않음. 왜곡 보정 비활성화.")
        # --- ---

    def capture_frame(self, undistort=True):
        """카메라에서 프레임을 캡처하여 반환합니다. 필요시 왜곡 보정을 수행합니다."""
        ret, frame = self.cap.read()
        if not ret:
            print("[WARN] 카메라 프레임 읽기 실패. 카메라 연결 상태 확인 필요.")
            time.sleep(0.1)
            return None

        # 왜곡 보정 수행
        if undistort and self.undistort_enabled:
            try:
                # remap 사용
                undistorted_frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
                return undistorted_frame
            except cv2.error as e: # OpenCV 관련 에러만 명시적으로 처리
                print(f"[ERROR] cv2.remap 중 OpenCV 오류 발생: {e}. 원본 프레임 반환.")
                return frame
            except Exception as e: # 그 외 예외
                print(f"[ERROR] cv2.remap 중 예기치 않은 오류 발생: {e}. 원본 프레임 반환.")
                return frame
        # 왜곡 보정 비활성화 또는 실패 시 원본 프레임 반환
        else:
            return frame

    def get_camera_parameters(self):
        """현재 적용된 카메라 파라미터 (보정 후 K 포함)를 반환합니다."""
        if self.undistort_enabled:
            # 보정된 프레임에 대한 파라미터 (new_K, D=0)
            corrected_D = np.zeros_like(self.D) if self.D is not None else None
            return self.new_K, corrected_D
        else:
            # 원본 파라미터 (로드 실패 시 None일 수 있음)
            return self.K, self.D

    def release_camera(self):
        """카메라 장치를 해제합니다."""
        if self.cap.isOpened():
            self.cap.release()
            print("[INFO] 카메라 리소스 해제 완료.") # 로그 레벨 INFO로 변경