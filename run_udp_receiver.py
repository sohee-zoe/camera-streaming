import argparse
import signal
import socket
import sys
import threading # Receiver 스레드 관리를 위해

# 로컬 모듈 임포트
import config
from udp_receiver import UdpStreamReceiver
from stream_displayer import StreamDisplayer

# --- 종료 처리를 위한 전역 변수 및 핸들러 ---
receiver_instance = None
displayer_instance = None
def signal_handler(sig, frame):
    print("\n[INFO] Ctrl+C 감지. 수신기 및 디스플레이어 종료 중...")
    if displayer_instance:
         # 디스플레이 루프가 메인 스레드에 있으므로 running 플래그만 설정
         displayer_instance.running = False
    if receiver_instance:
        receiver_instance.stop_receiving() # 수신 스레드 중지 요청
    # 메인 스레드의 display_loop가 종료되면 프로그램이 멈춤
signal.signal(signal.SIGINT, signal_handler)
# --- ---

def main():
    global receiver_instance, displayer_instance
    parser = argparse.ArgumentParser(description="UDP 비디오 스트림 수신기 (클래스 기반)")
    parser.add_argument("--listen_ip", type=str, default=config.CLIENT_IP, help=f"수신 대기 IP 주소 (기본값: {config.CLIENT_IP})")
    parser.add_argument("--port", type=int, default=config.PORT, help=f"UDP 포트 번호 (기본값: {config.PORT})")
    parser.add_argument("--buffer", type=int, default=config.CLIENT_RECV_BUFFER, help=f"UDP 수신 버퍼 크기 (기본값: {config.CLIENT_RECV_BUFFER})")
    parser.add_argument("--qsize", type=int, default=config.DEFAULT_QUEUE_MAX_SIZE, help=f"내부 프레임 큐 크기 (기본값: {config.DEFAULT_QUEUE_MAX_SIZE})")
    parser.add_argument("--skip", action='store_true', help="큐에 쌓인 오래된 프레임 건너뛰기")

    args = parser.parse_args()

    try:
        # 수신기 인스턴스 생성
        receiver_instance = UdpStreamReceiver(
            listen_ip=args.listen_ip,
            listen_port=args.port,
            buffer_size=args.buffer,
            queue_size=args.qsize
        )

        # 디스플레이어 인스턴스 생성 (수신기의 큐 사용)
        frame_queue = receiver_instance.get_frame_queue()
        displayer_instance = StreamDisplayer(
            frame_queue=frame_queue,
            skip_frames=args.skip,
            window_name=f"UDP 수신 ({args.listen_ip}:{args.port})"
        )

        # 수신 스레드 시작 (백그라운드 실행)
        receiver_instance.start_receiving()

        # 디스플레이 루프 시작 (메인 스레드에서 실행 - GUI 호환성)
        displayer_instance.start_display()

        # --- start_display()가 blocking 함수이므로, 여기 도달하면 종료된 것 ---
        print("[INFO] 디스플레이 루프 정상 종료됨. 자원 정리 중...")

    except (IOError, socket.error, Exception) as e:
        print(f"[ERROR] 수신기 초기화 또는 실행 중 오류 발생: {e}")
    finally:
        # 프로그램 종료 시 최종 정리 (signal handler에서 호출되지 않았을 경우 대비)
        if receiver_instance and receiver_instance.running_flag.is_set():
            receiver_instance.stop_receiving()
        if displayer_instance and displayer_instance.running:
             displayer_instance.stop_display()
        print("[INFO] 수신기 애플리케이션 종료.")


if __name__ == "__main__":
    main()

