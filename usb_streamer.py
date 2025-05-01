# 파일명: usb_streamer.py
import os
import cv2
import cv2.aruco as aruco # drawDetectedMarkers 사용 위해 유지
import time
import datetime
import numpy as np # Numpy import
import config
from camera_handler import CameraHandler # 변경 없음

class UsbStreamer:
    """
    CameraHandler에서 프레임 및 ArUco 데이터를 받아 화면에 표시하고,
    키 입력에 따라 원본 또는 화면 프레임을 저장합니다.
    """
    def __init__(self, camera_index=config.USB_CAMERA_INDEX,
                 width=640, height=480, fps=30,
                 calibration_file=None, no_undistort=False,
                 detect_aruco=True,
                 window_name=None, save_dir="captured_frames"):
        self.fps = fps
        self.window_name = window_name if window_name else f"USB Camera {camera_index}"
        self.running = False
        self.cam_handler = None
        self._display_fps = 0
        self._frame_count = 0
        self._start_time = time.time()
        self.save_dir = save_dir

        # 저장 디렉토리 생성
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                try:
                    os.makedirs(self.save_dir)
                    print(f"[INFO] 저장 디렉토리 생성됨: {self.save_dir}")
                except OSError as e:
                    print(f"[오류] 저장 디렉토리 생성 실패 ({self.save_dir}): {e}")
                    self.save_dir = None # 저장 비활성화
        else:
             print("[INFO] 프레임 저장이 비활성화되었습니다 (저장 디렉토리 미지정).")

        # CameraHandler 초기화 및 연결
        try:
            self.cam_handler = CameraHandler(
                index=camera_index,
                width=width, height=height, fps=fps, buffer_size=1,
                calibration_file=calibration_file,
                no_undistort=no_undistort,
                detect_aruco=detect_aruco
            )
            if not self.cam_handler.connect():
                self.cam_handler = None
                raise IOError(f"CameraHandler 연결 실패 (인덱스: {camera_index})")

            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.cam_handler.actual_width, self.cam_handler.actual_height)

        except (IOError, Exception) as e:
            print(f"[오류] USB 스트리머 초기화/연결 중 오류: {e}")
            if self.cam_handler: self.cam_handler.disconnect()
            self.cam_handler = None
            raise

    def _calculate_fps(self):
        """디스플레이 FPS 계산."""
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._start_time
        if elapsed >= 1.0:
            self._display_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._start_time = now

    def _save_frame(self, frame_to_save, filename_prefix="frame"):
        """주어진 프레임을 저장 디렉토리에 저장."""
        if frame_to_save is not None and self.save_dir:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3] # 밀리초까지
            filename = os.path.join(self.save_dir, f'{filename_prefix}_{timestamp}.jpg')
            try:
                cv2.imwrite(filename, frame_to_save)
                print(f"[INFO] 프레임 저장됨: {filename}")
            except Exception as e:
                print(f"[오류] 프레임 저장 실패 ({filename}): {e}")
        elif self.save_dir is None:
            print("[경고] 프레임을 저장할 수 없습니다: 저장 디렉토리가 설정되지 않았습니다.")
        else:
            print("[경고] 프레임을 저장할 수 없습니다: 프레임 데이터가 없습니다.")

    def display_loop(self):
        """메인 디스플레이 루프. 키 입력 처리 포함."""
        if not self.running: return
        if not self.cam_handler or not self.cam_handler.is_connected: return

        target_interval = 1.0 / self.fps if self.fps > 0 else 0
        last_capture_time = time.time()
        save_info = f"'{self.save_dir}'" if self.save_dir else "비활성화됨"

        print(f"[INFO] 로컬 USB 카메라 스트리밍 시작.")
        if self.cam_handler.perform_aruco_detection: print("[INFO] ArUco 감지 활성화됨 (CameraHandler 처리).")
        if self.cam_handler.perform_undistortion and self.cam_handler.undistortion_ready: print("[INFO] 왜곡 보정 활성화됨 (CameraHandler 처리).")
        if self.cam_handler.use_one_euro_filter: print("[INFO] OneEuroFilter 스무딩 활성화됨 (CameraHandler 처리).")
        print("[INFO] 'q': 종료")
        print(f"[INFO] 's': 원본 프레임 저장 (저장: {save_info})")
        print(f"[INFO] 'd': 화면 프레임 저장 (저장: {save_info})")

        while self.running:
            current_time = time.time()
            wait_time = target_interval - (current_time - last_capture_time)
            if target_interval > 0 and wait_time > 0.001: time.sleep(wait_time)
            last_capture_time = time.time()

            # CameraHandler로부터 프레임 및 데이터 받기 (원본 프레임 포함)
            ret, original_frame, processed_frame, aruco_data = self.cam_handler.get_frame()

            if not ret or processed_frame is None: # processed_frame 기준으로 확인
                time.sleep(0.1)
                continue

            # 화면에 표시하고 오버레이를 그릴 프레임 복사
            display_frame = processed_frame.copy()

            # --- ArUco 데이터 시각화 (display_frame에 그림) ---
            if aruco_data is not None:
                # 마커 테두리 그리기
                all_corners = [marker['corners'] for marker in aruco_data]
                all_ids = np.array([[marker['id']] for marker in aruco_data])
                if all_corners and all_ids.size > 0:
                    if hasattr(cv2, 'aruco') and hasattr(cv2.aruco, 'drawDetectedMarkers'):
                        cv2.aruco.drawDetectedMarkers(display_frame, all_corners, all_ids)
                    # else: 경고는 init에서 처리

                marker_info_text = []
                cam_matrix, dist_coeffs_orig = self.cam_handler.get_effective_camera_parameters()

                if cam_matrix is not None:
                    # 왜곡 계수 처리 (1D 또는 0벡터)
                    dist_coeffs_for_draw = dist_coeffs_orig if dist_coeffs_orig is not None else np.zeros((5,), dtype=np.float32)

                    for marker in aruco_data:
                        # 좌표축 그리기
                        try:
                            # 필터링된 rvec, tvec 사용
                            cv2.drawFrameAxes(display_frame, cam_matrix, dist_coeffs_for_draw,
                                           marker['rvec'], marker['tvec'],
                                           self.cam_handler.marker_length * 0.5)
                        except Exception as e:
                            print(f"[오류] cv2.drawFrameAxes 오류: {e}")

                        # 텍스트 정보 준비
                        try:
                            marker_id = marker.get('id', 'N/A')
                            tvec = marker.get('tvec')
                            rot_z = marker.get('rotation_z')

                            if tvec is not None and tvec.shape == (3, 1):
                                x, y, z = tvec[0, 0], tvec[1, 0], tvec[2, 0]
                            else: x, y, z = float('nan'), float('nan'), float('nan')

                            r_val = rot_z if rot_z is not None else float('nan')
                            info_str = f"ID {marker_id}: X={x:.2f} Y={y:.2f} Z={z:.2f} R={r_val:.1f}deg"
                            marker_info_text.append(info_str)
                        except Exception as e_text:
                             print(f"[오류] ID {marker_id} 텍스트 정보 처리 오류: {e_text}")

                    # 텍스트 정보 화면에 그리기
                    y_offset = 60
                    for text_line in marker_info_text:
                        cv2.putText(display_frame, text_line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                        y_offset += 20
                # else: cam_matrix None 경고는 init에서 처리

            # --- ArUco 시각화 끝 ---

            # FPS 계산 및 표시 (display_frame에 그림)
            self._calculate_fps()
            cv2.putText(display_frame, f"Display FPS: {self._display_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 최종 화면 표시
            cv2.imshow(self.window_name, display_frame)

            # --- 키 입력 처리 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] 'q' 키 눌림. 종료합니다.")
                self.running = False
                break
            elif key == ord('s'):
                # 원본 프레임 저장
                self._save_frame(original_frame, filename_prefix="original_frame")
            elif key == ord('d'):
                # 화면에 표시된 프레임 저장
                self._save_frame(display_frame, filename_prefix="display_frame")
            # --- ---

            # 창 닫기 버튼 처리
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] 디스플레이 창 닫힘. 종료합니다.")
                    self.running = False
                    break
            except cv2.error: pass # 오류 무시

        # --- 루프 종료 ---
        print("[INFO] 디스플레이 루프 종료됨.")

    def start_display(self):
        """디스플레이 시작 및 정리 처리."""
        if self.running: return
        if not self.cam_handler or not self.cam_handler.is_connected:
            print("[오류] 디스플레이 시작 불가: CameraHandler 준비 안 됨.")
            return

        self.running = True
        self._start_time = time.time()
        self._frame_count = 0
        try:
            self.display_loop()
        except Exception as e:
             print(f"[오류] display_loop 중 예외 발생: {e}")
             import traceback
             traceback.print_exc()
        finally:
             self.stop_display() # 항상 정리 수행

    def stop_display(self):
        """리소스 정리 (카메라, OpenCV 창)."""
        print("[INFO] USB 스트리머 리소스 정리 시도...")
        self.running = False

        if self.cam_handler:
            self.cam_handler.disconnect()
            self.cam_handler = None

        try:
            # 창 존재 여부 확인 후 파괴
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE) >= 0:
                 cv2.destroyWindow(self.window_name)
                 cv2.waitKey(1)
        except cv2.error: pass
        except Exception as e: print(f"[경고] OpenCV 창 닫기 중 오류: {e}")

        print("[INFO] USB 스트리머 중지됨.")

