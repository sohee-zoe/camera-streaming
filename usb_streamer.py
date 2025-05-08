# usb_streamer.py

import cv2
import numpy as np
import config
from camera_handler import CameraHandler
from stream_displayer import StreamDisplayer
import signal
import sys
from aruco_processor import ArucoProcessor


class UsbStreamer:
    def __init__(self, camera_index, width, height, fps, calibration_file=None, no_undistort=False):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.calibration_file = calibration_file
        self.no_undistort = no_undistort
        self.running = False
        self.cam_handler = None

        try:
            self.cam_handler = CameraHandler(
                index=self.camera_index,
                width=self.width,
                height=self.height,
                fps=self.fps,
                buffer_size=1,
                calibration_file=self.calibration_file
            )

        except IOError as e:
            print(f"[ERROR] 카메라 초기화 실패: {e}")
            raise
        except Exception as e:
            print(f"[ERROR] CameraHandler 초기화 중 예기치 않은 오류: {e}")
            raise

        # ArucoProcessor 초기화
        camera_matrix, dist_coeffs = self.cam_handler.get_camera_parameters()
        self.aruco_processor = ArucoProcessor(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            marker_length=config.DEFAULT_ARUCO_LENGTH
        )

        self.displayer = StreamDisplayer(frame_queue=None, window_name="USB Camera Stream")

    def start_display(self):
        self.running = True
        self.display_loop()

    def stop_display(self):
        self.running = False
        self.displayer.stop_display()
        if self.cam_handler:
            self.cam_handler.release_camera()

    def display_loop(self):
        if not self.running:
            print("[ERROR] 디스플레이 루프 시작 불가: 실행 플래그 비활성.")
            return

        print("[INFO] 화면 표시 루프 시작.")

        while self.running:
            original_frame = self.cam_handler.capture_frame(undistort=True)
            if original_frame is None:
                print("[WARN] 카메라 프레임 캡처 실패.")
                continue

            # # 왜곡 보정 적용
            # undistorted_frame = None
            # if self.cam_handler.undistort_enabled and self.cam_handler.mapx is not None and self.cam_handler.mapy is not None:
            #     try:
            #         undistorted_frame = cv2.remap(original_frame, self.cam_handler.mapx, self.cam_handler.mapy,
            #                                       cv2.INTER_LINEAR)
            #     except Exception as e:
            #         print(f"[ERROR] 왜곡 보정 중 오류: {e}")
            #         undistorted_frame = original_frame
            # else:
            #     undistorted_frame = original_frame
            #
            # display_frame = undistorted_frame.copy()  # 표시용 프레임 복사

            display_frame = original_frame.copy()  # 표시용 프레임 복사

            # ArUco 마커 검출 및 처리 (왜곡 보정된 프레임 사용)
            corners, ids, rvecs, tvecs = self.aruco_processor.detect_markers(display_frame)
            aruco_data = self.aruco_processor.get_pose_data(corners, ids, rvecs, tvecs)

            # 마커 테두리 및 좌표축 그리기
            self.aruco_processor.draw_detected_markers(display_frame, corners, ids)
            if rvecs is not None and tvecs is not None:
                for i in range(len(ids)):
                    self.aruco_processor.draw_axes(display_frame, rvecs[i], tvecs[i])

            # ArUco 마커 pos, rot 값 콘솔 출력 및 화면 표시용 텍스트 준비
            print("\n--- 감지된 ArUco 마커 정보 ---")  # 콘솔 출력 시작
            marker_info_text_for_display = []  # 화면 표시용 리스트

            cam_matrix, dist_coeffs_for_draw = self.cam_handler.get_effective_camera_parameters()

            if cam_matrix is not None:
                dist_coeffs_for_draw = dist_coeffs_for_draw if dist_coeffs_for_draw is not None else np.zeros((5,),
                                                                                                              dtype=np.float32)

                if aruco_data is not None:
                    for marker in aruco_data:
                        marker_id = marker.get('id', 'N/A')
                        tvec = marker.get('tvec')  # 변환 벡터 (카메라 좌표계 기준 위치)
                        rvec = marker.get('rvec')  # 회전 벡터 (카메라 좌표계 기준 회전)

                        # 콘솔에 pos (tvec) 와 rot (rvec) 값 출력
                        if tvec is not None:
                            pos_str_console = f" ID {marker_id}: Pos (tvec) = [X:{tvec[0, 0]:.3f}, Y:{tvec[1, 0]:.3f}, Z:{tvec[2, 0]:.3f}] (단위: m, 마커 크기 설정에 따라 다름)"
                            print(pos_str_console)

                        if rvec is not None:
                            rot_str_console = f" ID {marker_id}: Rot (rvec) = [{rvec[0, 0]:.3f}, {rvec[1, 0]:.3f}, {rvec[2, 0]:.3f}] (축-각 표현)"
                            print(rot_str_console)

                        # Z축 회전각 (이미 계산되어 있다면)
                        rotation_z_deg = marker.get('rotation_z')
                        if rotation_z_deg is not None:
                            print(f" ID {marker_id}: Z축 회전각 = {rotation_z_deg:.1f}도")

                        try:
                            x, y, z = (tvec[0, 0], tvec[1, 0], tvec[2, 0]) if tvec is not None else (float('nan'),
                                                                                                     float('nan'),
                                                                                                     float('nan'))
                            r_val = marker.get('rotation_z', float('nan'))
                            info_str_display = f"ID {marker_id}: X={x:.2f} Y={y:.2f} Z={z:.2f} R={r_val:.1f}deg"
                            marker_info_text_for_display.append(info_str_display)
                        except Exception as e_draw:
                            print(f"[오류] ID {marker_id} 시각화 정보 처리 오류: {e_draw}")

            print("------------------------------\n")  # 콘솔 출력 끝

            # 화면에 텍스트 정보 그리기
            y_offset = 60
            for text_line in marker_info_text_for_display:
                cv2.putText(display_frame, text_line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
                            cv2.LINE_AA)
                y_offset += 20

            # FPS 텍스트 화면에 표시
            cv2.putText(display_frame, f"Display FPS: {self.displayer._display_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("USB Camera Stream", display_frame)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] 'q' 키 입력 감지. 종료합니다.")
                self.running = False
                break

            if cv2.getWindowProperty("USB Camera Stream", cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] 디스플레이 창이 닫혔습니다. 종료합니다.")
                self.running = False
                break

        print("[INFO] 화면 표시 루프 종료됨.")
        self.stop_display()
