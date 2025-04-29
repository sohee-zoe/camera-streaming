import argparse
import signal
import sys

import config
from usb_streamer import UsbStreamer

# --- 종료 처리를 위한 전역 변수 및 핸들러 ---
streamer_instance = None
def signal_handler(sig, frame):
    print("\n[정보] Ctrl+C 감지. USB 스트리머 종료 중...")
    if streamer_instance:
        # display_loop가 메인 스레드에 있으므로 running 플래그만 설정
        streamer_instance.running = False
signal.signal(signal.SIGINT, signal_handler)
# --- ---

def main():
    global streamer_instance
    parser = argparse.ArgumentParser(description="USB 카메라 로컬 스트리머 (클래스 기반)")
    parser.add_argument("--camera", type=int, default=config.UDP_CAMERA_INDEX, help=f"카메라 인덱스 (기본값: {config.UDP_CAMERA_INDEX})")
    parser.add_argument("--width", type=int, default=640, help="카메라 프레임 너비")
    parser.add_argument("--height", type=int, default=480, help="카메라 프레임 높이")
    parser.add_argument("--fps", type=int, default=30, help="카메라 목표 FPS")
    parser.add_argument("--calib", type=str, default=config.GLOBAL_CALIBRATION_FILE, help="카메라 캘리브레이션 YAML 파일 경로")
    parser.add_argument("--no_undistort", action='store_true', help="캘리브레이션 파일 있어도 왜곡 보정 안 함")

    args = parser.parse_args()

    try:
        streamer_instance = UsbStreamer(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            calibration_file=args.calib,
            no_undistort=args.no_undistort
        )

        # 로컬 디스플레이 시작 (현재 스레드에서 실행, blocking)
        streamer_instance.start_display()

        # --- start_display() 종료 후 ---
        print("[정보] USB 스트리머 루프 정상 종료됨.")

    except (IOError, Exception) as e:
        print(f"[치명적 오류] USB 스트리머 초기화 또는 실행 중 오류 발생: {e}")
    finally:
        # 최종 정리
        if streamer_instance and streamer_instance.running:
            streamer_instance.stop_display()
        print("[정보] USB 스트리머 애플리케이션 종료.")


if __name__ == "__main__":
    main()

