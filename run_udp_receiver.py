# run_udp_receiver.py
import argparse
import signal
import socket
import sys
import threading
import config
from udp_receiver import UdpStreamReceiver
from stream_displayer import StreamDisplayer

receiver_instance = None
displayer_instance = None

def signal_handler(sig, frame):
    print("\n[INFO] Ctrl+C 감지. 수신기 및 디스플레이어 종료 중...")
    if displayer_instance:
        displayer_instance.running = False # StreamDisplayer의 루프 종료 플래그
    if receiver_instance:
        receiver_instance.stop_receiving() # UdpStreamReceiver의 스레드 종료 요청

signal.signal(signal.SIGINT, signal_handler)

def main():
    global receiver_instance, displayer_instance
    parser = argparse.ArgumentParser(description="UDP 비디오 스트림 수신기 (클래스 기반, ArUco 검출)")
    parser.add_argument("--listen_ip", type=str, default=config.CLIENT_IP, help=f"수신 대기 IP 주소 (기본값: {config.CLIENT_IP})")
    parser.add_argument("--port", type=int, default=config.PORT, help=f"UDP 포트 번호 (기본값: {config.PORT})")
    parser.add_argument("--buffer", type=int, default=config.CLIENT_RECV_BUFFER, help=f"UDP 수신 버퍼 크기 (기본값: {config.CLIENT_RECV_BUFFER})")
    parser.add_argument("--qsize", type=int, default=config.DEFAULT_QUEUE_MAX_SIZE, help=f"내부 프레임 큐 크기 (기본값: {config.DEFAULT_QUEUE_MAX_SIZE})")
    parser.add_argument("--skip", action='store_true', help="큐에 쌓인 오래된 프레임 건너뛰기 (StreamDisplayer용)")
    parser.add_argument("--calib", type=str, default=config.GLOBAL_CALIBRATION_FILE, # config에 기본 캘리브레이션 파일 경로 추가 필요
                        help="카메라 캘리브레이션 YAML 파일 경로 (ArUco 검출용)")
    args = parser.parse_args()

    try:
        receiver_instance = UdpStreamReceiver(
            listen_ip=args.listen_ip,
            listen_port=args.port,
            buffer_size=args.buffer,
            queue_size=args.qsize,
            calibration_file=args.calib # 캘리브레이션 파일 경로 전달
        )

        frame_queue = receiver_instance.get_frame_queue()
        displayer_instance = StreamDisplayer(
            frame_queue=frame_queue,
            skip_frames=args.skip,
            window_name=f"UDP 수신 ({args.listen_ip}:{args.port}) - ArUco"
        )

        receiver_instance.start_receiving() # 백그라운드 스레드로 수신 시작
        displayer_instance.start_display()  # 메인 스레드에서 화면 표시 루프 실행 (blocking)

        print("[INFO] 디스플레이 루프 정상 종료됨. 자원 정리 중...")

    except (IOError, socket.error, Exception) as e:
        print(f"[ERROR] 수신기 초기화 또는 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if receiver_instance and receiver_instance.running_flag.is_set():
            receiver_instance.stop_receiving()
        if displayer_instance and displayer_instance.running:
            # StreamDisplayer는 자체 루프에서 running 플래그를 확인하므로,
            # signal_handler에서 이미 False로 설정되었다면 여기서 추가 호출은 불필요할 수 있음
            # 하지만 명시적으로 stop_display 호출하여 창 정리.
            displayer_instance.stop_display()
        print("[INFO] 수신기 애플리케이션 종료.")

if __name__ == "__main__":
    main()
