import cv2
import socket
import queue
import threading
import time
import struct
import numpy as np
import config

class UdpStreamReceiver:
    def __init__(self, listen_ip=config.CLIENT_IP, listen_port=config.PORT,
                 buffer_size=config.CLIENT_RECV_BUFFER,
                 queue_size=config.DEFAULT_QUEUE_MAX_SIZE):
        """
        수신기 초기화.

        Args:
            listen_ip (str): 수신 대기할 IP 주소 ("0.0.0.0"은 모든 인터페이스).
            listen_port (int): 수신 대기할 UDP 포트.
            buffer_size (int): UDP 소켓 수신 버퍼 크기.
            queue_size (int): 재조립된 프레임을 저장할 큐의 최대 크기.
        """
        self.listen_address = (listen_ip, listen_port)
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.running_flag = threading.Event()
        self.thread = None
        self.sock = None
        self.reassembly_buffer = {} # 프레임 재조립 버퍼
        self._init_socket()

    def _init_socket(self):
        """UDP 리스닝 소켓 생성 및 설정."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(1.0) # 루프 탈출 및 상태 확인을 위한 타임아웃
            self.sock.bind(self.listen_address)
            # 수신 버퍼 크기 설정 시도
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)
            actual_buf_size = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            print(f"[INFO] UDP 수신 소켓 생성됨 (수신 대기: {self.listen_address[0]}:{self.listen_address[1]}). 수신 버퍼: {actual_buf_size}")
        except socket.error as e:
            print(f"[ERROR] UDP 소켓 바인딩/설정 실패: {e}")
            self.sock = None
            raise
        except Exception as e:
             print(f"[ERROR] 소켓 초기화 중 예외 발생: {e}")
             self.sock = None
             raise

    def _cleanup_stale_frames(self, current_time):
        """타임아웃된 미완성 프레임 정리."""
        stale_ids = [fid for fid, data in self.reassembly_buffer.items()
                     if current_time - data['last_seen'] > config.FRAME_REASSEMBLY_TIMEOUT]
        if stale_ids:
            print(f"[WARN] 타임아웃 프레임 정리: ID {stale_ids}")
            for frame_id in stale_ids:
                del self.reassembly_buffer[frame_id]

    def _receiver_loop(self):
        """UDP 패킷 수신 및 프레임 재조립 루프 (스레드에서 실행)."""
        if self.sock is None:
            print("[ERROR] 수신 루프 시작 불가: 소켓이 초기화되지 않음.")
            return

        last_cleanup_time = time.time()
        received_packets = 0
        processed_frames = 0
        last_processed_frame_id = -1

        while self.running_flag.is_set():
            try:
                msg, address = self.sock.recvfrom(self.buffer_size)
                current_time = time.time()
                received_packets += 1

                if len(msg) < config.HEADER_SIZE: continue # 너무 작은 패킷 무시

                # 헤더 파싱
                frame_id, total_chunks, chunk_id = struct.unpack(config.HEADER_FORMAT, msg[:config.HEADER_SIZE])
                payload = msg[config.HEADER_SIZE:]

                # 매우 오래된 프레임 ID 무시 (간단한 랩어라운드 처리)
                if last_processed_frame_id > 1000 and frame_id < (last_processed_frame_id - 1000):
                    continue

                # 재조립 버퍼 업데이트
                if frame_id not in self.reassembly_buffer:
                    # 버퍼 크기 제한 (간단히)
                    if len(self.reassembly_buffer) > 50:
                         print("[WARN] 재조립 버퍼가 너무 큽니다. 새 프레임 패킷 무시.")
                         self._cleanup_stale_frames(current_time) # 정리 시도
                         continue
                    self.reassembly_buffer[frame_id] = {'total': total_chunks, 'chunks': {}, 'last_seen': current_time}

                frame_entry = self.reassembly_buffer[frame_id]
                if chunk_id not in frame_entry['chunks']:
                    frame_entry['chunks'][chunk_id] = payload
                    frame_entry['last_seen'] = current_time
                    if frame_entry['total'] != total_chunks: # 헤더 INFO가 다르면 최신 값 사용
                         frame_entry['total'] = total_chunks

                # 프레임 완성 확인
                if len(frame_entry['chunks']) == frame_entry['total']:
                    # print(f"[디버그] 프레임 {frame_id} 완성 ({frame_entry['total']} 청크).")
                    sorted_chunks = [frame_entry['chunks'][i] for i in range(frame_entry['total'])]
                    full_frame_data = b"".join(sorted_chunks)

                    # 완성된 프레임 큐에 넣기 (큐 full 시 오래된 것 제거)
                    try:
                        if self.frame_queue.full():
                            try: self.frame_queue.get_nowait()
                            except queue.Empty: pass
                        self.frame_queue.put(full_frame_data, block=False)
                        processed_frames += 1
                        last_processed_frame_id = frame_id
                    except queue.Full:
                         print("[WARN] 프레임 큐가 계속 꽉 차 있습니다. 현재 프레임 버림.")
                         pass

                    del self.reassembly_buffer[frame_id] # 버퍼에서 제거

            except socket.timeout:
                # 타임아웃은 정상, 주기적 정리 수행
                current_time = time.time()
                if current_time - last_cleanup_time > config.FRAME_REASSEMBLY_TIMEOUT / 2:
                    self._cleanup_stale_frames(current_time)
                    last_cleanup_time = current_time
                continue
            except struct.error as e:
                print(f"[WARN] 헤더 언패킹 실패: {e}")
                continue
            except Exception as e:
                print(f"[ERROR] UDP 수신 루프 ERROR: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1) # ERROR 발생 시 잠시 대기

        print(f"[INFO] UDP 수신 스레드 종료됨. 총 수신 패킷: {received_packets}, 처리된 프레임: {processed_frames}")
        if self.sock:
            self.sock.close()
            print("[INFO] UDP 수신 소켓 닫힘.")

    def start_receiving(self):
        """수신 스레드 시작."""
        if self.thread is not None and self.thread.is_alive():
            print("[WARN] 이미 수신 스레드가 실행 중입니다.")
            return

        if self.sock is None:
             print("[ERROR] 수신 시작 불가: 소켓이 초기화되지 않음.")
             return

        self.running_flag.set()
        self.thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self.thread.start()
        print("[INFO] UDP 수신 스레드가 백그라운드에서 시작되었습니다.")

    def stop_receiving(self):
        """수신 스레드 중지 요청 및 대기."""
        if not self.running_flag.is_set():
            return

        print("[INFO] 수신 스레드 중지 요청 중...")
        self.running_flag.clear() # 스레드 루프 종료 신호

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0) # 스레드 종료 대기 (최대 2초)
            if self.thread.is_alive():
                print("[WARN] 수신 스레드가 시간 내에 깔끔하게 종료되지 않았습니다.")
                if self.sock:
                    print("[INFO] 수신 소켓 강제 닫기 시도.")
                    self.sock.close() # 스레드가 블록되어 있을 수 있으므로 소켓 닫기

        print("[INFO] 수신기 중지 완료.")

    def get_frame_queue(self) -> queue.Queue:
        """프레임 큐 객체 반환."""
        return self.frame_queue
