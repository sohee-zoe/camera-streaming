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
        """
        aruco_data = []

        if ids is not None and corners is not None and rvecs is not None and tvecs is not None:
            for i, marker_id in enumerate(ids.flatten()):
                marker_corners = corners[i]
                rvec = rvecs[i]
                tvec = tvecs[i]

                # Z축 회전각 계산 (예시)
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                z_rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                rotation_z_deg = np.degrees(z_rotation_angle)

                aruco_data.append({
                    'id': marker_id,
                    'corners': marker_corners,
                    'rvec': rvec,
                    'tvec': tvec,
                    'rotation_z': rotation_z_deg
                })

        return aruco_data
