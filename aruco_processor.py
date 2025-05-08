# aruco_processor.py

import cv2
import numpy as np


class ArucoProcessor:
    def __init__(self, camera_matrix, dist_coeffs, aruco_dict_type=cv2.aruco.DICT_ARUCO_ORIGINAL, marker_length=0.05):
        """
        ArUco 마커 검출 및 포즈 추정을 처리하는 클래스입니다.
        Args:
            camera_matrix (np.ndarray): 카메라 캘리브레이션 매트릭스 (K).
            dist_coeffs (np.ndarray): 카메라 왜곡 계수 (D).
            aruco_dict_type (int): 사용할 ArUco 사전 타입 (cv2.aruco.DICT_XXX).
            marker_length (float): ArUco 마커 한 변의 길이 (미터 단위).
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.marker_length = marker_length

    def detect_markers(self, image):
        """
        이미지에서 ArUco 마커를 검출하고, 마커 코너, ID, 회전 벡터(rvecs), 변환 벡터(tvecs)를 반환합니다.
        Args:
            image (np.ndarray): 입력 이미지 (grayscale 또는 컬러).
        Returns:
            tuple: (corners, ids, rvecs, tvecs) 튜플. 마커가 검출되지 않으면 (None, None, None, None)을 반환합니다.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 and image.shape[2] > 1 else image

        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is not None:
            rvecs = []
            tvecs = []

            for i in range(len(corners)):
                # 3D 모델 좌표 (마커의 네 모서리)
                objPoints = np.array([
                    [-self.marker_length / 2, self.marker_length / 2, 0],
                    [self.marker_length / 2, self.marker_length / 2, 0],
                    [self.marker_length / 2, -self.marker_length / 2, 0],
                    [-self.marker_length / 2, -self.marker_length / 2, 0]
                ], dtype=np.float32)

                # solvePnP로 포즈 추정
                retval, rvec, tvec = cv2.solvePnP(
                    objPoints,
                    corners[i],
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                rvecs.append(rvec)
                tvecs.append(tvec)

            return corners, ids, np.array(rvecs), np.array(tvecs)
        else:
            return None, None, None, None

    def draw_detected_markers(self, image, corners, ids):
        """
        이미지에 검출된 ArUco 마커를 그리고, ID를 표시합니다.
        Args:
            image (np.ndarray): 입력 이미지.
            corners (list): 마커 코너 좌표 리스트.
            ids (np.ndarray): 마커 ID 배열.
        """
        if corners is not None and ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)

    def draw_axes(self, image, rvec, tvec):
        """
        이미지에 ArUco 마커 좌표축을 그립니다.
        Args:
            image (np.ndarray): 입력 이미지.
            rvec (np.ndarray): 회전 벡터.
            tvec (np.ndarray): 변환 벡터.
        """
        if rvec is not None and tvec is not None:
            # 수정된 부분: drawAxis 대신 drawFrameAxes 사용
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)

    def get_pose_data(self, corners, ids, rvecs, tvecs):
        """
        ArUco 마커의 포즈 정보를 담은 리스트를 반환합니다.
        쿼터니언 변환과 각도 연속성 유지 로직, LPF 필터링이 적용되었습니다.
        """
        from scipy.spatial.transform import Rotation

        aruco_data = []
        if ids is not None and corners is not None and rvecs is not None and tvecs is not None:
            for i, marker_id in enumerate(ids.flatten()):
                marker_corners = corners[i]
                rvec = rvecs[i]
                tvec = tvecs[i]

                # 회전 벡터를 쿼터니언으로 변환
                rotation = Rotation.from_rotvec(rvec.flatten())

                # 쿼터니언 정규화
                quat = rotation.as_quat()  # [x, y, z, w] 형식
                quat_norm = np.sqrt(np.sum(quat ** 2))
                if quat_norm > 0:
                    quat = quat / quat_norm

                # 정규화된 쿼터니언으로 새 회전 객체 생성
                rotation = Rotation.from_quat(quat)

                # 오일러 각으로 변환 (ZYX 순서)
                euler_angles = rotation.as_euler('zyx', degrees=True)
                rotation_z_deg = euler_angles[0]  # Z축 회전각

                # 각도 연속성 유지
                marker_attr = f'prev_rotation_z_{marker_id}'
                if hasattr(self, marker_attr):
                    prev_z = getattr(self, marker_attr)
                    if abs(rotation_z_deg - prev_z) > 170:
                        if rotation_z_deg > 0 and prev_z < 0:
                            rotation_z_deg -= 360
                        elif rotation_z_deg < 0 and prev_z > 0:
                            rotation_z_deg += 360

                # LPF 적용 (1차 저주파 통과 필터)
                lpf_attr = f'lpf_rotation_z_{marker_id}'
                tau = 0.1  # 필터 시정수 (작을수록 더 많은 필터링)
                Ts = 0.033  # 샘플링 시간 (30fps 기준)

                if not hasattr(self, lpf_attr):
                    # 첫 프레임은 필터링 없이 초기값으로 설정
                    setattr(self, lpf_attr, rotation_z_deg)
                    filtered_z = rotation_z_deg
                else:
                    # LPF 공식: y_k = (tau * y_k-1 + Ts * x_k) / (tau + Ts)
                    prev_filtered_z = getattr(self, lpf_attr)
                    filtered_z = (tau * prev_filtered_z + Ts * rotation_z_deg) / (tau + Ts)
                    setattr(self, lpf_attr, filtered_z)

                # 현재 원시 각도 저장 (연속성 유지용)
                setattr(self, marker_attr, rotation_z_deg)

                # 위치 정보에도 LPF 적용
                tvec_filtered = np.zeros_like(tvec)
                for j in range(3):  # x, y, z 각각에 대해
                    pos_attr = f'lpf_pos_{marker_id}_{j}'
                    if not hasattr(self, pos_attr):
                        setattr(self, pos_attr, tvec[j, 0])
                        tvec_filtered[j, 0] = tvec[j, 0]
                    else:
                        prev_filtered_pos = getattr(self, pos_attr)
                        filtered_pos = (tau * prev_filtered_pos + Ts * tvec[j, 0]) / (tau + Ts)
                        setattr(self, pos_attr, filtered_pos)
                        tvec_filtered[j, 0] = filtered_pos

                aruco_data.append({
                    'id': marker_id,
                    'corners': marker_corners,
                    'rvec': rvec,
                    'tvec': tvec_filtered,
                    'quaternion': quat,
                    'euler_angles': euler_angles,
                    'rotation_z': filtered_z  # 필터링된 Z축 회전각
                })
        return aruco_data
