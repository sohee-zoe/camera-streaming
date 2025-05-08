# aruco_processor.py
import cv2
import numpy as np


class ArucoProcessor:
    def __init__(self, camera_matrix, dist_coeffs, aruco_dict_type=cv2.aruco.DICT_ARUCO_ORIGINAL, marker_length=0.05):
        """
        ArUco 마커 검출 및 포즈 추정을 처리하는 클래스입니다.
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
        """
        if corners is not None and ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)

    def draw_axes(self, image, rvec, tvec):
        """
        이미지에 ArUco 마커 좌표축을 그립니다.
        """
        if rvec is not None and tvec is not None:
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)
