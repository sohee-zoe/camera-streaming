# udp_receiver.py
import cv2
import socket
import queue
import threading
import time
import struct
import numpy as np
import config
from aruco_processor import ArucoProcessor  # ArucoProcessor 임포트
from calibration_utils import load_calibration_from_yaml  # YAML 로딩 함수 임포트


class UdpStreamReceiver:
    def __init__(self, listen_ip=config.CLIENT_IP, listen_port=config.PORT,
                 buffer_size=config.CLIENT_RECV_BUFFER,
                 queue_size=config.DEFAULT_QUEUE_MAX_SIZE,
                 calibration_file=None):  # 캘리브레이션 파일 인자 추가
        self.listen_address = (listen_ip, listen_port)
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=queue_size)  # StreamDisplayer가 사용할 큐
        self.running_flag = threading.Event()
        self.thread = None
        self.sock = None
        self.reassembly_buffer = {}
        self.aruco_processor = None
        self.K_for_aruco = None  # ArUco 처리에 사용할 카메라 매트릭스
        self.D_for_aruco = None  # ArUco 처리에 사용할 왜곡 계수 (대부분 None 또는 0)

        # 캘리브레이션 파일 로드 및 ArUcoProcessor 초기화
        if calibration_file:
            print(f"[INFO] UdpReceiver: 캘리브레이션 파일 로드 시도: {calibration_file}")
            # 송신측 카메라의 원본 K, D 및 이미지 크기 로드
            temp_K, temp_D, calib_img_width, calib_img_height = load_calibration_from_yaml(calibration_file)

            if temp_K is not None and temp_D is not None:
                # 송신측 CameraHandler가 왜곡 보정을 했다고 가정.
                # 수신된 프레임은 new_K에 해당하며, 왜곡은 제거된 상태.
                # 따라서 ArucoProcessor에는 new_K와 D=None을 사용해야 함.
                # 송신측에서 사용한 것과 동일한 파라미터로 new_K를 계산해야 함.
                # (주의: 실제 수신 이미지 크기를 알아야 정확한 new_K 계산 가능. 여기서는 캘리브레이션 파일의 크기를 우선 사용)
                img_w = calib_img_width if calib_img_width else config.CAMERA_DEFAULT_WIDTH  # 예시: config에서 기본 너비 사용
                img_h = calib_img_height if calib_img_height else config.CAMERA_DEFAULT_HEIGHT  # 예시: config에서 기본 높이 사용

                # 실제 수신 프레임의 크기를 알아야 new_K를 정확히 계산할 수 있으나,
                # 우선 캘리브레이션 파일에 명시된 크기나 기본값으로 new_K 계산 시도.
                # 첫 프레임 수신 후 실제 크기로 재계산하는 로직이 더 정확할 수 있음.
                # 여기서는 단순화를 위해 캘리브레이션 파일의 K, D와 명시된 이미지 크기로 new_K를 구함.
                # 송신측 CameraHandler의 alpha값과 동일하게 사용 (여기서는 alpha=0 가정)
                try:
                    # 수신할 프레임의 예상 크기 (첫 프레임 수신 전에는 알 수 없으므로, 캘리브레이션 당시 크기나 기본값 사용)
                    # 보다 정확하려면, 첫 프레임 수신 후 해당 프레임의 크기로 new_K를 다시 계산해야 할 수 있음.
                    # 여기서는 캘리브레이션 파일에 명시된 이미지 크기 또는 기본 너비/높이로 가정.
                    # target_size_for_new_K = (img_w, img_h) # 이 부분이 수신 프레임 실제 크기와 다를 수 있음.
                    # 송신측에서 설정한 width, height를 사용하는 것이 더 적절할 수 있음.
                    # 하지만 송신측 설정값을 수신측이 미리 알기 어려움.
                    # 여기서는 캘리브레이션 파일의 K를 new_K로 간주하고 D=None으로 단순화.
                    # 또는, 송신측 CameraHandler가 사용한 new_K 값을 어떻게든 전달받아야 함.

                    # 가장 실용적인 접근: 캘리브레이션 파일의 K를 사용하고, D는 0으로 가정 (송신측에서 왜곡 보정했으므로)
                    # 만약 송신측이 new_K를 사용했다면, 수신측도 해당 new_K를 알아야 함.
                    # 여기서는 편의상 원본 K를 new_K 대신 사용한다고 가정하고, D는 0으로.
                    # 정확한 포즈 추정을 위해서는 송신측의 `new_K` 값을 알아야 함.
                    # 여기서는 캘리브레이션 파일의 K를 그대로 사용하고 D는 0으로 설정.
                    # (실제로는 송신측 CameraHandler가 사용한 new_K를 사용하는 것이 이상적)
                    alpha = 0  # 송신측 CameraHandler의 alpha 값과 동일하게 설정 가정
                    # 송신측에서 이미 왜곡보정을 한 프레임을 보내므로,
                    # 수신측에서는 보정된 K (new_K) 와 D=None (또는 0)을 사용해야 함.
                    # calibration_file에서 읽은 K,D로 new_K를 다시 계산
                    # 주의: target_size는 실제 수신되는 프레임의 크기여야 함. (UDP 헤더에 포함시키거나, 첫 프레임에서 알아내야 함)
                    # 우선, 캘리브레이션 파일의 이미지 크기로 new_K 계산.
                    # 실제로는 UDP로 전송되는 이미지의 해상도를 알아야 함.
                    # 이 예제에서는 캘리브레이션 파일의 K를 사용하고 D는 0으로 단순화.
                    # 실제 환경에서는 송신측에서 사용한 new_K 값을 수신측이 알 수 있도록 하는 메커니즘 필요.
                    # 예를 들어, config.py에 해당 값을 저장하거나, 통신 초기에 교환.
                    # 여기서는 로드한 K를 그대로 사용하고 D는 0으로 설정하여 진행.
                    # 즉, 송신측이 왜곡보정 시 사용한 new_K와 동일한 값을 수신측이 안다고 가정.

                    # 송신측에서 왜곡 보정을 했다면, 수신측에서는 보정된 new_K와 왜곡계수 0을 사용해야 한다.
                    # 캘리브레이션 파일에는 원본 K, D가 있으므로, 수신측에서도 new_K를 계산해야 한다.
                    # 이 계산에는 "수신되는 프레임의 실제 크기"가 필요하다.
                    # 이 크기를 UDP 패킷으로 전달받거나, config에 고정값으로 설정해야 한다.
                    # 여기서는 우선 config.py의 기본 카메라 크기를 사용한다고 가정.
                    # target_size_for_new_K = (config.CAMERA_DEFAULT_WIDTH, config.CAMERA_DEFAULT_HEIGHT)
                    # self.K_for_aruco, _ = cv2.getOptimalNewCameraMatrix(temp_K, temp_D, (calib_img_width, calib_img_height), alpha, target_size_for_new_K)

                    # 단순화: 캘리브레이션 파일의 K를 사용하고, D는 0으로 간주.
                    # 이는 송신측이 왜곡보정 후 보낸 프레임의 카메라 매트릭스가 원본 K와 유사하고 왜곡만 제거된 경우에 해당.
                    # 더 정확하게는 송신측 CameraHandler가 사용한 new_K 값을 여기에 사용해야 함.
                    # config.py에 `SENDER_CAMERA_NEW_K` 와 같은 항목을 두고 사용하거나,
                    # 송신측에서 전송하는 이미지와 함께 이 정보를 보내는 방법도 있음.
                    # 여기서는 캘리브레이션 파일의 K를 사용하고 D는 0으로 하여 ArucoProcessor를 초기화.
                    self.K_for_aruco = temp_K
                    self.D_for_aruco = np.zeros((5,), dtype=np.float32)  # 왜곡 보정된 이미지를 받으므로 왜곡 계수는 0

                    self.aruco_processor = ArucoProcessor(
                        camera_matrix=self.K_for_aruco,
                        dist_coeffs=self.D_for_aruco,  # 왜곡 보정된 이미지를 받으므로 None 또는 0
                        aruco_dict_type=config.ARUCO_DICT.get(config.DEFAULT_ARUCO_TYPE, cv2.aruco.DICT_6X6_250),
                        marker_length=config.DEFAULT_ARUCO_LENGTH
                    )
                    print("[INFO] UdpReceiver: ArucoProcessor 초기화 완료.")
                except Exception as e:
                    print(f"[ERROR] UdpReceiver: new_K 계산 또는 ArucoProcessor 초기화 중 오류: {e}")
                    self.K_for_aruco = None  # 오류 시 None으로 설정
                    self.aruco_processor = None
            else:
                print("[WARN] UdpReceiver: 캘리브레이션 파일에서 K 또는 D를 로드하지 못했습니다.")
        else:
            print("[WARN] UdpReceiver: 캘리브레이션 파일이 제공되지 않아 ArUco 처리가 제한될 수 있습니다.")

        self._init_socket()

    def _init_socket(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(1.0)
            self.sock.bind(self.listen_address)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)
            actual_buf_size = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            print(
                f"[INFO] UDP 수신 소켓 생성됨 (수신 대기: {self.listen_address[0]}:{self.listen_address[1]}). 수신 버퍼: {actual_buf_size}")
        except (socket.error, Exception) as e:
            print(f"[ERROR] UDP 소켓 바인딩/설정 실패 또는 초기화 중 예외: {e}")
            self.sock = None
            raise

    def _cleanup_stale_frames(self, current_time):
        stale_ids = [fid for fid, data in self.reassembly_buffer.items()
                     if current_time - data['last_seen'] > config.FRAME_REASSEMBLY_TIMEOUT]
        if stale_ids:
            print(f"[WARN] 타임아웃 프레임 정리: ID {stale_ids}")
            for frame_id in stale_ids:
                del self.reassembly_buffer[frame_id]

    def _receiver_loop(self):
        if self.sock is None:
            print("[ERROR] 수신 루프 시작 불가: 소켓이 초기화되지 않음.")
            return
        last_cleanup_time = time.time()
        received_packets = 0
        processed_frames_count = 0  # 변수명 변경
        last_processed_frame_id = -1

        while self.running_flag.is_set():
            try:
                msg, address = self.sock.recvfrom(self.buffer_size)
                current_time = time.time()
                received_packets += 1
                if len(msg) < config.HEADER_SIZE: continue

                frame_id, total_chunks, chunk_id = struct.unpack(config.HEADER_FORMAT, msg[:config.HEADER_SIZE])
                payload = msg[config.HEADER_SIZE:]

                # (간단한 랩어라운드 처리 및 재조립 버퍼 로직은 원본 유지)
                if last_processed_frame_id > 1000 and frame_id < (last_processed_frame_id - 1000): continue
                if frame_id not in self.reassembly_buffer:
                    if len(self.reassembly_buffer) > 50:
                        self._cleanup_stale_frames(current_time)
                        continue
                    self.reassembly_buffer[frame_id] = {'total': total_chunks, 'chunks': {}, 'last_seen': current_time}

                frame_entry = self.reassembly_buffer[frame_id]
                if chunk_id not in frame_entry['chunks']:
                    frame_entry['chunks'][chunk_id] = payload
                frame_entry['last_seen'] = current_time
                if frame_entry['total'] != total_chunks: frame_entry['total'] = total_chunks

                if len(frame_entry['chunks']) == frame_entry['total']:
                    sorted_chunks = [frame_entry['chunks'][i] for i in range(frame_entry['total'])]
                    full_frame_data_bytes = b"".join(sorted_chunks)  # JPEG 바이트 데이터

                    # JPEG 바이트를 OpenCV 프레임으로 디코딩
                    cv_frame = cv2.imdecode(np.frombuffer(full_frame_data_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if cv_frame is None:
                        print(f"[WARN] 프레임 {frame_id} 디코딩 실패.")
                        del self.reassembly_buffer[frame_id]
                        continue

                    # --- ArUco 마커 검출 및 처리 ---
                    if self.aruco_processor and self.K_for_aruco is not None:
                        # detect_markers는 이미 K와 D=None (또는 0)으로 초기화된 ArucoProcessor 내부에서 처리
                        corners, ids, rvecs, tvecs = self.aruco_processor.detect_markers(cv_frame)
                        aruco_data_list = self.aruco_processor.get_pose_data(corners, ids, rvecs, tvecs)  # 변수명 변경

                        if aruco_data_list:  # 검출된 마커가 있다면
                            print(f"\n--- UDP 수신 (프레임 {frame_id}): 감지된 ArUco 마커 정보 ---")
                            for marker_data in aruco_data_list:
                                marker_id = marker_data.get('id', 'N/A')
                                tvec_m = marker_data.get('tvec')
                                rvec_m = marker_data.get('rvec')
                                rot_z_deg = marker_data.get('rotation_z')

                                print(f"  ID {marker_id}: "
                                      f"Pos(tvec) = [{tvec_m[0, 0]:.3f}, {tvec_m[1, 0]:.3f}, {tvec_m[2, 0]:.3f}] | "
                                      f"Rot(rvec) = [{rvec_m[0, 0]:.3f}, {rvec_m[1, 0]:.3f}, {rvec_m[2, 0]:.3f}] | "
                                      f"Z-Rot = {rot_z_deg:.1f}deg")
                            print("--------------------------------------------------\n")

                            # 프레임에 마커 정보 그리기 (StreamDisplayer로 보내기 전)
                            self.aruco_processor.draw_detected_markers(cv_frame, corners, ids)
                            if rvecs is not None and tvecs is not None and ids is not None:
                                for i in range(len(ids)):
                                    self.aruco_processor.draw_axes(cv_frame, rvecs[i], tvecs[i])
                    # --- ArUco 처리 끝 ---

                    # 시각화 정보가 포함된 프레임을 다시 JPEG으로 인코딩하여 큐에 넣음
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), config.DEFAULT_JPEG_QUALITY]  # 송신측과 동일 품질 사용 가능
                    result, final_encoded_bytes = cv2.imencode('.jpg', cv_frame, encode_param)

                    if result:
                        try:
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.frame_queue.put(final_encoded_bytes.tobytes(), block=False)
                            processed_frames_count += 1
                            last_processed_frame_id = frame_id
                        except queue.Full:
                            print("[WARN] 프레임 큐가 계속 꽉 차 있습니다. 현재 프레임 버림.")
                            pass
                    else:
                        print(f"[WARN] 프레임 {frame_id} 재인코딩 실패.")

                    del self.reassembly_buffer[frame_id]

            except socket.timeout:
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
                time.sleep(0.1)

        print(f"[INFO] UDP 수신 스레드 종료됨. 총 수신 패킷: {received_packets}, 처리된 프레임: {processed_frames_count}")
        if self.sock:
            self.sock.close()
            print("[INFO] UDP 수신 소켓 닫힘.")

    def start_receiving(self):
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
        if not self.running_flag.is_set(): return
        print("[INFO] 수신 스레드 중지 요청 중...")
        self.running_flag.clear()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("[WARN] 수신 스레드가 시간 내에 깔끔하게 종료되지 않았습니다.")
        # 소켓 닫기는 루프 종료 후 자동으로 처리됨
        print("[INFO] 수신기 중지 완료.")

    def get_frame_queue(self) -> queue.Queue:
        return self.frame_queue
