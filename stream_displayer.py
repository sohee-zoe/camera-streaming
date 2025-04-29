import os
import datetime
import cv2
import queue
import time
import numpy as np

class StreamDisplayer:
    """
    큐에서 프레임 바이트를 가져와 디코딩하고 화면에 표시하는 클래스.
    """

    def __init__(self, frame_queue: queue.Queue, skip_frames=False, window_name="Stream Display", save_dir="captured_frames"):
        """
        디스플레이어 초기화.

        Args:
            frame_queue (queue.Queue): 프레임 바이트를 가져올 큐.
            skip_frames (bool): True면 큐에 쌓인 오래된 프레임 건너뛰기.
            window_name (str): OpenCV 창 제목.
            save_dir (str): 프레임 저장 디렉토리 경로. # 추가
        """
        self.frame_queue = frame_queue
        self.skip_frames = skip_frames
        self.window_name = window_name
        self.running = False
        self._display_fps = 0
        self._frame_count = 0
        self._start_time = time.time()

        # --- 이미지 저장 관련 초기화 ---
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
                print(f"[INFO] 프레임 저장 디렉토리 생성: {self.save_dir}")
            except OSError as e:
                print(f"[ERROR] 저장 디렉토리 생성 실패 ({self.save_dir}): {e}")
                self.save_dir = None # 저장 비활성화
        # --- 추가 끝 ---

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 480)
        except Exception as e:
            print(f"[ERROR] OpenCV 창 생성 실패: {e}. GUI 환경이 아닐 수 있습니다.")


    def _calculate_fps(self):
        """FPS 계산."""
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._start_time
        if elapsed >= 1.0:
            self._display_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._start_time = now

    def display_loop(self):
        """메인 디스플레이 루프. 'q' 종료, 's' 저장.""" # 설명 수정
        if not self.running:
            print("[ERROR] 디스플레이 루프 시작 불가: 실행 플래그 비활성.")
            return

        print(f"[INFO] 화면 표시 루프 시작. 'q': 종료, 's': 프레임 저장 ({self.save_dir if self.save_dir else '저장 비활성화됨'})") # 안내 메시지 수정

        while self.running:
            processed_this_loop = False
            original_frame = None # 원본 프레임 저장 변수 # 추가

            try:
                # 프레임 건너뛰기 로직
                if self.skip_frames and self.frame_queue.qsize() > 1:
                    try: self.frame_queue.get_nowait()
                    except queue.Empty: pass

                # 큐에서 프레임 가져오기
                frame_bytes = self.frame_queue.get(block=True, timeout=0.5)
                processed_this_loop = True

                # 디코딩
                frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is None:
                    print("[WARN] 프레임 디코딩 실패.")
                    continue

                original_frame = frame.copy() # 텍스트 추가 전 원본 복사 # 추가

                # FPS 계산
                self._calculate_fps()

                # FPS 텍스트 화면에 표시 (이 부분은 원본이 아닌 frame 변수에 적용됨)
                cv2.putText(frame, f"Display FPS: {self._display_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 화면 표시
                cv2.imshow(self.window_name, frame)

                # 키 입력 및 창 상태 확인
                key = cv2.waitKey(1) & 0xFF

                # --- 's' 키 눌렀을 때 원본 프레임 저장 --- #
                if key == ord('s'):
                    if original_frame is not None and self.save_dir:
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        filename = os.path.join(self.save_dir, f'frame_{timestamp}.jpg')
                        try:
                            cv2.imwrite(filename, original_frame)
                            print(f"[INFO] 원본 프레임 저장됨: {filename}")
                        except Exception as e:
                            print(f"[ERROR] 프레임 저장 실패 ({filename}): {e}")
                    elif self.save_dir is None:
                         print("[WARN] 저장 디렉토리가 설정되지 않아 저장할 수 없습니다.")
                    else:
                         print("[WARN] 저장할 원본 프레임이 없습니다.")

                elif key == ord('q'): # 기존 'q' 처리 부분
                    print("[INFO] 'q' 키 입력 감지. 종료합니다.")
                    self.running = False
                    break

                # 창이 닫혔는지 확인 (기존과 동일)
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] 디스플레이 창이 닫혔습니다. 종료합니다.")
                    self.running = False
                    break

            except queue.Empty:
                if not processed_this_loop:
                    time.sleep(0.01)
                continue
            except cv2.error as e:
                print(f"[ERROR] OpenCV ERROR 발생: {e}")
                time.sleep(0.1)
            except Exception as e:
                print(f"[ERROR] 디스플레이 루프 중 예기치 않은 ERROR 발생: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

        print("[INFO] 화면 표시 루프 종료됨.")
        self.stop_display()

    def start_display(self):
        """디스플레이 루프 시작."""
        if self.running:
            print("[WARN] 이미 디스플레이가 실행 중입니다.")
            return
        self.running = True
        self.display_loop()

    def stop_display(self):
        """OpenCV 창 닫기."""
        print("[INFO] OpenCV 창 리소스 정리 시도...")
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except Exception as e:
            print(f"[WARN] OpenCV 창 닫기 중 ERROR: {e}")
        self.running = False
