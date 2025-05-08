# usb_aruco_detector.py

import cv2
import argparse
import sys
import time
import numpy as np

# 로컬 모듈 임포트
import config
from camera_handler import CameraHandler
from aruco_processor import ArucoProcessor
from calibration_utils import load_calibration_from_yaml


def main():
    parser = argparse.ArgumentParser(description="USB 카메라 ArUco 마커 탐지기")
    parser.add_argument("--camera", type=int, default=config.USB_CAMERA_INDEX,
                        help=f"카메라 인덱스 (기본값: {config.USB_CAMERA_INDEX})")
    parser.add_argument("--width", type=int, default=config.CAMERA_DEFAULT_WIDTH,
                        help=f"카메라 프레임 너비 (기본값: {config.CAMERA_DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=config.CAMERA_DEFAULT_HEIGHT,
                        help=f"카메라 프레임 높이 (기본값: {config.CAMERA_DEFAULT_HEIGHT})")
    parser.add_argument("--fps", type=int, default=30,
                        help="카메라 목표 FPS (기본값: 30)")
    parser.add_argument("--calib", type=str, default=config.GLOBAL_CALIBRATION_FILE,
                        help="카메라 캘리브레이션 YAML 파일 경로")
    parser.add_argument("--aruco_dict", type=str, default=config.DEFAULT_ARUCO_TYPE,
                        help=f"ArUco 사전 타입 (기본값: {config.DEFAULT_ARUCO_TYPE})")
    parser.add_argument("--marker_length", type=float, default=config.USB_ARUCO_LENGTH,
                        help=f"마커 크기(미터) (기본값: {config.USB_ARUCO_LENGTH}m)")
    parser.add_argument("--save_dir", type=str, default="captured_frames",
                        help="프레임 저장 디렉토리 (기본값: captured_frames)")

    args = parser.parse_args()

    try:
        # 카메라 핸들러 초기화
        cam_handler = CameraHandler(
            index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            buffer_size=1,
            calibration_file=args.calib
        )

        # 카메라 매트릭스와 왜곡 계수 가져오기
        K, D = cam_handler.get_effective_camera_parameters()

        if K is None or D is None:
            print("[ERROR] 카메라 파라미터를 가져올 수 없습니다. 캘리브레이션 파일이 올바른지 확인하세요.")
            return

        # ArUco 프로세서 초기화
        aruco_dict_type = config.ARUCO_DICT.get(args.aruco_dict, cv2.aruco.DICT_6X6_250)
        aruco_processor = ArucoProcessor(
            camera_matrix=K,
            dist_coeffs=D,
            aruco_dict_type=aruco_dict_type,
            marker_length=args.marker_length
        )

        # 창 생성
        window_name = f"USB 카메라 ArUco 탐지 (카메라 {args.camera})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

        print(f"[INFO] 'q'키: 종료, 's'키: 프레임 저장 ({args.save_dir})")

        frame_count = 0
        start_time = time.time()
        display_fps = 0

        while True:
            # 왜곡 보정된 프레임 캡처
            frame = cam_handler.capture_frame(undistort=True)

            if frame is None:
                print("[WARN] 프레임 캡처 실패, 다시 시도합니다.")
                time.sleep(0.1)
                continue

            # ArUco 마커 검출
            corners, ids, rejected, rvecs, tvecs = aruco_processor.detect_markers(frame)

            # 마커 정보 처리 및 시각화
            aruco_data_list = aruco_processor.get_pose_data(corners, ids, rvecs, tvecs)

            if aruco_data_list:
                print("\n--- 감지된 ArUco 마커 정보 ---")
                for marker_data in aruco_data_list:
                    marker_id = marker_data.get('id', 'N/A')
                    tvec_m = marker_data.get('tvec')
                    rot_z_deg = marker_data.get('rotation_z')
                    print(f" ID {marker_id}: "
                          f"위치 = [{tvec_m[0, 0]:.3f}, {tvec_m[1, 0]:.3f}, {tvec_m[2, 0]:.3f}] | "
                          f"Z축 회전 = {rot_z_deg:.1f}°")
                print("----------------------------------\n")

            # 마커 시각화
            aruco_processor.draw_detected_markers(frame, corners, ids, None)
            if rvecs is not None and tvecs is not None and ids is not None:
                for i in range(len(ids)):
                    aruco_processor.draw_axes(frame, rvecs[i], tvecs[i])

            # FPS 계산
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # FPS 표시
            cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 화면 표시
            cv2.imshow(window_name, frame)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF

            # 's' 키: 프레임 저장
            if key == ord('s'):
                import os
                import datetime

                if not os.path.exists(args.save_dir):
                    try:
                        os.makedirs(args.save_dir)
                        print(f"[INFO] 디렉토리 생성: {args.save_dir}")
                    except OSError as e:
                        print(f"[ERROR] 디렉토리 생성 실패: {e}")
                        continue

                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = os.path.join(args.save_dir, f'frame_{timestamp}.jpg')

                try:
                    cv2.imwrite(filename, frame)
                    print(f"[INFO] 프레임 저장됨: {filename}")
                except Exception as e:
                    print(f"[ERROR] 프레임 저장 실패: {e}")

            # 'q' 키: 종료
            elif key == ord('q'):
                print("[INFO] 'q' 키 입력 감지. 종료합니다.")
                break

            # 창이 닫혔는지 확인
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] 창이 닫혔습니다. 종료합니다.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] 키보드 인터럽트 감지. 종료합니다.")
    except Exception as e:
        print(f"[ERROR] 예기치 않은 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 리소스 정리
        if 'cam_handler' in locals():
            cam_handler.release_camera()
        cv2.destroyAllWindows()
        print("[INFO] 리소스 정리 완료. 프로그램 종료.")


if __name__ == "__main__":
    main()
