import os
import sys
import cv2
import numpy as np
import socket
import struct
import threading
import time
from camera_handler import CameraHandler
import config

class UdpStreamSender:
    def __init__(self, target_ip, target_port, camera_index=config.UDP_CAMERA_INDEX,
                 width=640, height=480, fps=30,
                 quality=config.DEFAULT_JPEG_QUALITY,
                 calibration_file=None, no_undistort=False,
                 send_buffer_size=config.SERVER_SEND_BUFFER):
        """
        송신기 초기화.

        Args:
            target_ip (str): 수신측 IP 주소.
            target_port (int): 수신측 UDP 포트.
            camera_index (int): 사용할 카메라 인덱스.
            width (int): 카메라 프레임 너비.
            height (int): 카메라 프레임 높이.
            fps (int): 목표 초당 프레임 수.
            quality (int): JPEG 압축 품질 (0-100).
            calibration_file (str, optional): 카메라 캘리브레이션 YAML 파일 경로.
            no_undistort (bool): True면 캘리브레이션 파일 있어도 왜곡 보정 안 함.
            send_buffer_size (int): UDP 소켓 송신 버퍼 크기.
        """
        self.target_address = (target_ip, target_port)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self.no_undistort = no_undistort
        self.send_buffer_size = send_buffer_size

        self.running = False
        self.socket = None
        self.thread = None
        self.frame_id = 0
        self.cam_handler = None

        try:
            calibration_file = calibration_file if not no_undistort else None
            self.cam_handler = CameraHandler(
                self.camera_index, self.width, self.height, self.fps,
                buffer_size=1, calibration_file=calibration_file
            )
        except IOError as e:
            print(f"[ERROR] 카메라 초기화 실패: {e}")
            raise
        except Exception as e:
            print(f"[ERROR] CameraHandler 초기화 중 예기치 않은 ERROR: {e}")
            raise

        self._init_socket()

    def _init_socket(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.send_buffer_size)
            actual_buf_size = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            print(f"[INFO] UDP 송신 소켓 생성됨. 송신 버퍼 크기: {actual_buf_size}")
        except socket.error as e:
            print(f"[ERROR] UDP 소켓 생성 실패: {e}")
            self.sock = None
            raise
        except Exception as e:
            print(f"[ERROR] 소켓 설정 중 예외 발생: {e}")
            self.sock = None
            raise

    def _send_frame(self, frame: np.ndarray):
        """단일 프레임을 인코딩하고 청크로 나누어 전송."""
        if self.sock is None or frame is None:
            return False

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        result, encoded_bytes = cv2.imencode('.jpg', frame, encode_param)

        if not result:
            print("[WARN] 프레임 JPEG 인코딩 실패.")
            return False

        data = encoded_bytes.tobytes()
        data_size = len(data)
        total_chunks = (data_size + config.CHUNK_PAYLOAD_SIZE - 1) // config.CHUNK_PAYLOAD_SIZE
        if total_chunks == 0: total_chunks = 1

        # print(f"[DEBUG] 프레임 {self.frame_id} 전송: 크기={data_size}, 청크 수={total_chunks}")

        success = True
        for chunk_id in range(total_chunks):
            start = chunk_id * config.CHUNK_PAYLOAD_SIZE
            end = min(start + config.CHUNK_PAYLOAD_SIZE, data_size)
            payload = data[start:end]

            try:
                header = struct.pack(config.HEADER_FORMAT, self.frame_id, total_chunks, chunk_id)
                message = header + payload
                self.sock.sendto(message, self.target_address)
                # time.sleep(0.0001) # 매우 짧은 지연 (필요시)
            except socket.error as e:
                print(f"[ERROR] UDP 전송 실패 (프레임 {self.frame_id}, 청크 {chunk_id}): {e}")
                success = False
                break  # 현재 프레임 전송 중단
            except struct.error as e:
                print(f"[ERROR] 헤더 패킹 실패: {e}")
                success = False
                break
            except Exception as e:
                print(f"[ERROR] 예기치 않은 전송 ERROR: {e}")
                success = False
                break

        if success:
            self.frame_id = (self.frame_id + 1) % (2 ** 32)  # 프레임 ID 증가 (랩어라운드)

        return success

    def _streaming_loop(self):
        target_interval = 1.0 / self.fps if self.fps > 0 else 0
        last_capture_time = time.time()

        while self.running:
            current_time = time.time()
            elapsed = current_time - last_capture_time

            # 목표 FPS에 맞춰 대기 시간 계산
            wait_time = target_interval - elapsed
            if target_interval > 0 and wait_time > 0.001:  # 너무 짧은 sleep 방지
                time.sleep(wait_time)
                last_capture_time = time.time()  # sleep 후 시간 다시 측정
            elif elapsed < target_interval:
                # sleep 없이 CPU 양보 시도 (짧은 시간 남았을 때)
                time.sleep(0.0001)
                last_capture_time = time.time()
            else:
                # 목표 시간 지났으면 바로 진행
                last_capture_time = current_time  # 다음 루프 위해 시간 업데이트

            # 프레임 캡처 (왜곡 보정은 핸들러에서 처리)
            frame = self.cam_handler.capture_frame(undistort=(not self.no_undistort))

            if frame is None:
                print("[WARN] 프레임 캡처 실패, 건너<0xEB><0><0x8A>.")
                time.sleep(0.1)  # 캡처 실패 시 잠시 대기
                continue

            # 프레임 전송
            if not self._send_frame(frame):
                print("[WARN] 프레임 전송 실패, 계속 시도...")
                time.sleep(0.05)  # 전송 실패 시 잠시 대기

        print("[INFO] 스트리밍 루프 종료됨.")

    def start_streaming(self, run_in_thread=False):
        """스트리밍 시작."""
        if self.running:
            print("[WARN] 이미 스트리밍이 실행 중입니다.")
            return

        if self.sock is None or self.cam_handler is None:
            print("[ERROR] 소켓 또는 카메라 핸들러가 초기화되지 않았습니다.")
            return

        self.running = True
        print(f"[INFO] UDP 스트리밍 시작 -> {self.target_address[0]}:{self.target_address[1]}")

        if run_in_thread:
            # 별도 스레드에서 실행
            self.thread = threading.Thread(target=self._streaming_loop, daemon=True)
            self.thread.start()
            print("[INFO] 송신 스레드가 백그라운드에서 시작되었습니다.")
        else:
            # 현재 스레드에서 실행 (blocking)
            self._streaming_loop()

    def stop_streaming(self):
        """스트리밍 중지 및 자원 해제."""
        if not self.running:
            return

        print("[INFO] 스트리밍 중지 요청 중...")
        self.running = False

        # 스레드로 실행된 경우 스레드 종료 기다림
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("[WARN] 송신 스레드가 시간 내에 깔끔하게 종료되지 않았습니다.")

        # 자원 해제
        if self.cam_handler:
            self.cam_handler.release_camera()
        if self.sock:
            self.sock.close()
            print("[INFO] UDP 소켓 닫힘.")

        print("[INFO] 송신기 중지 완료.")




