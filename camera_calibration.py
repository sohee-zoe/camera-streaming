import os
import cv2
import numpy as np
import glob
import argparse
from typing import Tuple, List, Optional
from calibration_utils import save_calibration_to_yaml, load_calibration_from_yaml


def setup_criteria() -> Tuple[int, int, float]:
    """코너 서브픽셀 검출 기준 설정"""
    # TERM_CRITERIA_EPS: 지정된 정확도에 도달하면 반복 중지
    # TERM_CRITERIA_MAX_ITER: 지정된 최대 반복 횟수에 도달하면 중지
    return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def setup_object_points(
        num_corners_x: int,
        num_corners_y: int,
        square_size: float
) -> np.ndarray:
    """3D 월드 좌표계 포인트 생성"""
    objp = np.zeros((num_corners_x * num_corners_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1, 2)
    return objp * square_size


def process_images(
        path: str,
        pattern_size: Tuple[int, int],
        criteria: Tuple[int, int, float],
        objp: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    globs = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    images = []
    for ext in globs:
        images.extend(glob.glob(os.path.join(path, ext)))
        
    if not images:
        raise FileNotFoundError(f"No JPG images found in {path}")
    print(f"총 {len(images)}개의 이미지 파일 검색됨.")

    object_points = [] # 3D 월드 좌표 (모든 이미지에 대해 동일 objp 추가)
    image_points = [] # 각 이미지에서 검출된 2D 코너 좌표
    pattern_detected = 0
    img_shape: Optional[Tuple[int, int]] = None

    for i, fname in enumerate(images):
        try:
            img = cv2.imread(fname)
            if img is None:
                print(f"[WARN] 이미지 로드 실패: {fname}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 첫 이미지에서 shape 결정 시도
            if img_shape is None:
                img_shape = gray.shape[::-1]  # (width, height)

            # 체커보드 코너 찾기
            # flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            success, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)

            print(f"이미지 처리 중 [{i + 1}/{len(images)}]: {os.path.basename(fname)} - 패턴 검출: {'성공' if success else '실패'}")

            # 코너 검출 성공 시
            if success:
                pattern_detected += 1
                object_points.append(objp)  # 성공한 이미지에 대해서만 objp 추가

                # 서브픽셀 단위로 코너 위치 정교화
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                image_points.append(corners_refined)

                # 시각화: 검출된 코너 그리기
                cv2.drawChessboardCorners(img, pattern_size, corners_refined, success)
                cv2.imshow("Detected Pattern", img)
                cv2.waitKey(100)  # 0.1초 대기

        except Exception as e:
            print(f"[ERROR] 이미지 처리 중 오류 발생 ({fname}): {e}")
            continue  # 다음 이미지로 넘어감

    cv2.destroyWindow("Detected Pattern")  # 시각화 창 닫기

    if pattern_detected == 0:
        raise ValueError("어떤 이미지에서도 체커보드 패턴을 검출하지 못했습니다.")
    if img_shape is None:
        raise ValueError("이미지 크기를 결정할 수 없습니다. 유효한 이미지가 없습니다.")

    print(f"\n패턴 검출 성공: {pattern_detected}/{len(images)}개 이미지")
    return object_points, image_points, img_shape


def calibrate(
        num_corners_x: int,
        num_corners_y: int,
        square_size: float,
        path: str,
        output_file: str = "camera_param.yaml" # 출력 파일 이름 인자 추가
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """카메라 캘리브레이션을 수행하고 결과를 YAML 파일에 저장합니다."""
    try:
        criteria = setup_criteria()
        objp = setup_object_points(num_corners_x, num_corners_y, square_size)
        pattern_size = (num_corners_x, num_corners_y)

        print("이미지 처리 및 패턴 검출 시작...")
        object_points, image_points, img_shape = process_images(
            path, pattern_size, criteria, objp
        )

        if not object_points or not image_points or img_shape is None:
            print("[ERROR] 캘리브레이션에 필요한 데이터를 얻지 못했습니다.")
            return None

        print(f"\n캘리브레이션 수행 시작 (이미지 크기: {img_shape[0]}x{img_shape[1]})...")
        # 캘리브레이션 수행
        # mtx: 카메라 매트릭스 (K)
        # dist: 왜곡 계수 (D)
        # rvecs, tvecs: 각 이미지별 회전 및 이동 벡터
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, img_shape, None, None
        )

        if not ret:
            print("[ERROR] 카메라 캘리브레이션 실패.")
            return None

        print(f"캘리브레이션 완료. RMS 재투영 오류(Reprojection Error): {ret:.4f}")
        if ret > 1.0:
             print("[WARN] RMS 오류가 1.0보다 큽니다. 캘리브레이션 품질이 낮을 수 있습니다.")

        # 결과 저장 (이미지 크기 포함)
        save_calibration_to_yaml(output_file, mtx, dist,
                                 image_width=img_shape[0], image_height=img_shape[1])

        # 저장된 파일 다시 로드하여 검증
        verify_mtx, verify_dist, verify_width, verify_height = load_calibration_from_yaml(output_file)

        if verify_mtx is None or verify_dist is None:
             print("[ERROR] 저장된 YAML 파일 로드/검증 실패.")
             return None

        print("\n--- 캘리브레이션 결과 (YAML 파일 검증) ---")
        print(f"이미지 크기: {verify_width} x {verify_height}")
        print(f"카메라 매트릭스 (K):\n{verify_mtx}")
        print(f"왜곡 계수 (D):\n{verify_dist}")
        print("-" * (30 + len(" 캘리브레이션 결과 (YAML 파일 검증) ")))
        return mtx, dist

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return None
    except ValueError as e:
        print(f"[ERROR] {e}")
        return None
    except Exception as e:
        print(f"[ERROR] 캘리브레이션 중 예기치 않은 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add_argument("--size",
                        type=lambda s: tuple(map(int, s.split('x'))),
                        default=(8, 6),
                        help="체커보드 코너 개수 (가로x세로), 예: 8x6")
    parser.add_argument("--square",
                        type=float,
                        default=0.025, # 25 mm
                        help="체커보드 한 칸의 실제 크기(미터 단위)")
    parser.add_argument("--path",
                        type=str,
                        default="./frames",
                        help="이미지 저장 경로")
    parser.add_argument("--output",
                        type=str,
                        default="camera_param.yaml",  # 기본 출력 파일 이름
                        help="캘리브레이션 결과 저장 파일 이름 (YAML)")

    args = parser.parse_args()

    if len(args.size) != 2 or args.size[0] <= 0 or args.size[1] <= 0:
        parser.error("--size 인자는 '가로x세로' 형태여야 하며, 양수 값이어야 합니다 (예: 8x6).")


    calibration_result = calibrate(
        num_corners_x=args.size[0],
        num_corners_y=args.size[1],
        square_size=args.square,
        path=args.path,
        output_file=args.output
    )

    if calibration_result:
        print("\n캘리브레이션 성공!")
    else:
        print("\n캘리브레이션 실패.")


if __name__ == "__main__":
    main()
