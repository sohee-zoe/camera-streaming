# advanced_aruco_detector.py

import cv2
import numpy as np
import argparse
import time
import os
import datetime
import threading
import queue
from scipy.spatial.transform import Rotation
from collections import deque
from enum import Enum

# 로컬 모듈 임포트
import config
from camera_handler import CameraHandler
from calibration_utils import load_calibration_from_yaml

"""
# LPF 필터 사용 (기본값)
python aruco_detector.py --camera 2 --filter lpf --lpf_tau 0.3

# EKF 필터 사용
python aruco_detector.py --camera 2 --filter ekf --ekf_process_noise 0.01

# UKF 필터 사용
python aruco_detector.py --camera 2 --filter ukf --ukf_alpha 0.1 --ukf_beta 2.0

# 파티클 필터 사용
python aruco_detector.py --camera 2 --filter particle --particle_count 200

# 다중 마커 평균화 사용
python aruco_detector.py --camera 2 --filter multi_avg

# 원시 데이터도 함께 표시
python aruco_detector.py --camera 2 --filter lpf --show_raw
"""

"""
실시간 처리가 중요하고 비선형성이 낮은 경우: LPF
일반적인 비선형 시스템: EKF
비선형성이 강하고 계산 자원이 충분한 경우: UKF
최고의 정확도가 필요하고 계산 비용이 문제되지 않는 경우: 파티클 필터
"""

class FilterType(Enum):
    NONE = 0
    LPF = 1
    EKF = 2
    UKF = 3
    PARTICLE = 4
    MULTI_MARKER_AVG = 5


class MarkerState:
    """마커의 상태를 추적하기 위한 클래스"""

    def __init__(self, marker_id, initial_tvec=None, initial_rvec=None):
        self.marker_id = marker_id

        # 위치 및 회전 상태
        self.tvec = initial_tvec.copy() if initial_tvec is not None else np.zeros((3, 1), dtype=np.float32)
        self.rvec = initial_rvec.copy() if initial_rvec is not None else np.zeros((3, 1), dtype=np.float32)

        # 속도 상태 (EKF용)
        self.vel = np.zeros((3, 1), dtype=np.float32)
        self.rot_vel = np.zeros((3, 1), dtype=np.float32)

        # 쿼터니언 및 오일러 각
        self.quat = np.array([0, 0, 0, 1], dtype=np.float32)  # [x, y, z, w]
        self.euler_angles = np.zeros(3, dtype=np.float32)  # [z, y, x]

        # 상태 공분산 행렬 (EKF용)
        self.P = np.eye(12, dtype=np.float32) * 1000  # 초기 불확실성 높게 설정

        # 마지막 업데이트 시간
        self.last_update_time = time.time()

        # 필터링된 값 저장
        self.filtered_tvec = self.tvec.copy()
        self.filtered_rvec = self.rvec.copy()
        self.filtered_quat = self.quat.copy()
        self.filtered_euler = self.euler_angles.copy()

        # 파티클 필터용 파티클
        self.particles_pos = None
        self.particles_rot = None
        self.particles_weights = None

        # 히스토리 저장 (디버깅 및 스무딩용)
        self.history_tvec = deque(maxlen=10)
        self.history_rvec = deque(maxlen=10)


class AdvancedArucoProcessor:
    """향상된 ArUco 마커 처리 클래스 (다양한 필터링 옵션 지원)"""

    def __init__(self, camera_matrix, dist_coeffs, aruco_dict_type=cv2.aruco.DICT_6X6_250,
                 marker_length=0.05, filter_type=FilterType.LPF, filter_params=None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_length = marker_length

        # 필터 설정
        self.filter_type = filter_type
        self.filter_params = filter_params if filter_params else {}

        # LPF 파라미터
        self.lpf_tau = self.filter_params.get('lpf_tau', 0.3)  # 시정수 (높을수록 더 강한 필터링)
        self.lpf_Ts = self.filter_params.get('lpf_Ts', 0.033)  # 샘플링 시간 (30fps 기준)

        # EKF 파라미터
        self.ekf_process_noise = self.filter_params.get('ekf_process_noise', 0.01)
        self.ekf_measurement_noise_pos = self.filter_params.get('ekf_measurement_noise_pos', 0.1)
        self.ekf_measurement_noise_rot = self.filter_params.get('ekf_measurement_noise_rot', 0.2)

        # UKF 파라미터
        self.ukf_alpha = self.filter_params.get('ukf_alpha', 0.1)
        self.ukf_beta = self.filter_params.get('ukf_beta', 2.0)
        self.ukf_kappa = self.filter_params.get('ukf_kappa', 0.0)
        self.ukf_process_noise = self.filter_params.get('ukf_process_noise', 0.01)
        self.ukf_measurement_noise = self.filter_params.get('ukf_measurement_noise', 0.1)

        # 파티클 필터 파라미터
        self.particle_count = self.filter_params.get('particle_count', 100)
        self.particle_noise_pos = self.filter_params.get('particle_noise_pos', 0.01)
        self.particle_noise_rot = self.filter_params.get('particle_noise_rot', 0.05)

        # 다중 마커 평균화 파라미터
        self.use_marker_average = self.filter_type == FilterType.MULTI_MARKER_AVG

        # 마커 상태 사전
        self.marker_states = {}

        # 마지막 처리 시간
        self.last_process_time = time.time()

    def detect_markers(self, image):
        """이미지에서 ArUco 마커를 검출하고 포즈를 추정합니다."""
        # 이미지가 컬러인 경우 그레이스케일로 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 and image.shape[2] > 1 else image

        # ArUco 마커 검출
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
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

            return corners, ids, rejected, np.array(rvecs), np.array(tvecs)
        else:
            return None, None, rejected, None, None

    def draw_detected_markers(self, image, corners, ids, rejected=None):
        """이미지에 검출된 ArUco 마커를 그리고, ID를 표시합니다."""
        if corners is not None and ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)

        # 거부된 마커 후보 시각화 (빨간색으로 표시)
        if rejected is not None and len(rejected) > 0:
            cv2.aruco.drawDetectedMarkers(image, rejected, borderColor=(0, 0, 255))

    # def draw_axes(self, image, rvec, tvec, length=None):
    #     """이미지에 ArUco 마커 좌표축을 그립니다."""
    #     if rvec is not None and tvec is not None:
    #         axis_length = length if length else self.marker_length * 0.5
    #         cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, axis_length)

    def draw_axes(self, image, rvec, tvec, length=None):
        """이미지에 ArUco 마커 좌표축을 그립니다."""
        if rvec is None or tvec is None:
            return

        try:
            # 데이터 타입을 명시적으로 float32로 변환
            rvec_float = np.array(rvec, dtype=np.float32)
            tvec_float = np.array(tvec, dtype=np.float32)

            axis_length = length if length else self.marker_length * 0.5
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec_float, tvec_float, axis_length)
        except Exception as e:
            print(f"[ERROR] 축 그리기 실패: {e}")

    def apply_lpf(self, marker_state, tvec, rvec):
        """저주파 통과 필터(LPF)를 적용합니다."""
        current_time = time.time()
        dt = current_time - marker_state.last_update_time

        # 시간이 너무 오래 지났으면 필터링 초기화
        if dt > 1.0:
            marker_state.filtered_tvec = tvec.copy()
            marker_state.filtered_rvec = rvec.copy()
            marker_state.last_update_time = current_time
            return

        # LPF 공식: y_k = (tau * y_k-1 + Ts * x_k) / (tau + Ts)
        tau = self.lpf_tau
        Ts = self.lpf_Ts

        # 위치 필터링
        for i in range(3):
            marker_state.filtered_tvec[i, 0] = (tau * marker_state.filtered_tvec[i, 0] + Ts * tvec[i, 0]) / (tau + Ts)

        # 회전 필터링 (회전 벡터 직접 필터링)
        for i in range(3):
            marker_state.filtered_rvec[i, 0] = (tau * marker_state.filtered_rvec[i, 0] + Ts * rvec[i, 0]) / (tau + Ts)

        marker_state.last_update_time = current_time

    def apply_ekf(self, marker_state, tvec, rvec):
        """확장 칼만 필터(EKF)를 적용합니다."""
        current_time = time.time()
        dt = current_time - marker_state.last_update_time

        # 시간이 너무 오래 지났으면 필터링 초기화
        if dt > 1.0:
            marker_state.filtered_tvec = tvec.copy()
            marker_state.filtered_rvec = rvec.copy()
            marker_state.vel = np.zeros((3, 1), dtype=np.float32)
            marker_state.rot_vel = np.zeros((3, 1), dtype=np.float32)
            marker_state.P = np.eye(12, dtype=np.float32) * 1000
            marker_state.last_update_time = current_time
            return

        # 상태 벡터: [x, y, z, vx, vy, vz, rx, ry, rz, wx, wy, wz]
        # 여기서 (x,y,z)는 위치, (vx,vy,vz)는 속도, (rx,ry,rz)는 회전 벡터, (wx,wy,wz)는 각속도

        # 1. 예측 단계
        # 상태 전이 행렬 F
        F = np.eye(12, dtype=np.float32)
        F[0, 3] = F[1, 4] = F[2, 5] = dt  # 위치 = 이전 위치 + 속도*dt
        F[6, 9] = F[7, 10] = F[8, 11] = dt  # 회전 = 이전 회전 + 각속도*dt

        # 프로세스 노이즈 공분산 행렬 Q
        Q = np.eye(12, dtype=np.float32) * self.ekf_process_noise
        Q[3:6, 3:6] *= 10  # 속도에 대한 노이즈 증가
        Q[9:12, 9:12] *= 10  # 각속도에 대한 노이즈 증가

        # 상태 벡터 구성
        x = np.zeros((12, 1), dtype=np.float32)
        x[0:3, 0] = marker_state.filtered_tvec[:, 0]
        x[3:6, 0] = marker_state.vel[:, 0]
        x[6:9, 0] = marker_state.filtered_rvec[:, 0]
        x[9:12, 0] = marker_state.rot_vel[:, 0]

        # 상태 예측
        x_pred = F @ x
        P_pred = F @ marker_state.P @ F.T + Q

        # 2. 업데이트 단계
        # 측정 행렬 H (측정값은 위치와 회전만)
        H = np.zeros((6, 12), dtype=np.float32)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 6] = H[4, 7] = H[5, 8] = 1

        # 측정 노이즈 공분산 행렬 R
        R = np.eye(6, dtype=np.float32)
        R[0:3, 0:3] *= self.ekf_measurement_noise_pos  # 위치 측정 노이즈
        R[3:6, 3:6] *= self.ekf_measurement_noise_rot  # 회전 측정 노이즈

        # 측정값
        z = np.zeros((6, 1), dtype=np.float32)
        z[0:3, 0] = tvec[:, 0]
        z[3:6, 0] = rvec[:, 0]

        # 칼만 게인 계산
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # 상태 업데이트
        y = z - H @ x_pred  # 측정 잔차
        x_updated = x_pred + K @ y
        marker_state.P = (np.eye(12) - K @ H) @ P_pred

        # 업데이트된 상태 저장
        marker_state.filtered_tvec = x_updated[0:3].reshape(3, 1)
        marker_state.vel = x_updated[3:6].reshape(3, 1)
        marker_state.filtered_rvec = x_updated[6:9].reshape(3, 1)
        marker_state.rot_vel = x_updated[9:12].reshape(3, 1)

        marker_state.last_update_time = current_time

    # UKF 구현 함수 추가
    def apply_ukf(self, marker_state, tvec, rvec):
        """UKF(Unscented Kalman Filter)를 적용합니다."""
        current_time = time.time()
        dt = current_time - marker_state.last_update_time

        # 필요한 속성이 없으면 동적으로 추가
        if not hasattr(marker_state, 'ukf_state'):
            setattr(marker_state, 'ukf_state', None)
        if not hasattr(marker_state, 'ukf_P'):
            setattr(marker_state, 'ukf_P', None)

        # 시간이 너무 오래 지났으면 필터링 초기화
        if dt > 1.0 or marker_state.ukf_state is None:
            # 상태 벡터: [x, y, z, vx, vy, vz, rx, ry, rz, wx, wy, wz]
            marker_state.ukf_state = np.zeros(12)
            marker_state.ukf_state[0:3] = tvec.flatten()
            marker_state.ukf_state[6:9] = rvec.flatten()

            # 공분산 행렬 초기화
            marker_state.ukf_P = np.eye(12) * 1000

            marker_state.filtered_tvec = tvec.copy()
            marker_state.filtered_rvec = rvec.copy()
            marker_state.last_update_time = current_time
            return

        # 1. 시그마 포인트 생성
        n = 12  # 상태 차원
        lambda_param = self.ukf_alpha ** 2 * (n + self.ukf_kappa) - n

        # 공분산 행렬의 제곱근 계산 (Cholesky 분해)
        try:
            L = np.linalg.cholesky((n + lambda_param) * marker_state.ukf_P)
        except np.linalg.LinAlgError:
            # 공분산 행렬이 양의 정부호가 아니면 대각 행렬로 초기화
            marker_state.ukf_P = np.eye(12) * 1000
            L = np.linalg.cholesky((n + lambda_param) * marker_state.ukf_P)

        # 시그마 포인트 계산
        X = np.zeros((2 * n + 1, n))
        X[0] = marker_state.ukf_state
        for i in range(n):
            X[i + 1] = marker_state.ukf_state + L[i]
            X[i + 1 + n] = marker_state.ukf_state - L[i]

        # 2. 시그마 포인트 전파 (예측 단계)
        X_pred = np.zeros_like(X)
        for i in range(2 * n + 1):
            # 간단한 상태 전이 모델: 위치 += 속도*dt, 회전 += 각속도*dt
            X_pred[i, 0:3] = X[i, 0:3] + X[i, 3:6] * dt
            X_pred[i, 3:6] = X[i, 3:6]  # 속도는 유지
            X_pred[i, 6:9] = X[i, 6:9] + X[i, 9:12] * dt
            X_pred[i, 9:12] = X[i, 9:12]  # 각속도는 유지

        # 가중치 계산
        Wm = np.zeros(2 * n + 1)
        Wc = np.zeros(2 * n + 1)
        Wm[0] = lambda_param / (n + lambda_param)
        Wc[0] = lambda_param / (n + lambda_param) + (1 - self.ukf_alpha ** 2 + self.ukf_beta)
        for i in range(1, 2 * n + 1):
            Wm[i] = Wc[i] = 1 / (2 * (n + lambda_param))

        # 예측된 상태와 공분산 계산
        x_pred = np.zeros(n)
        for i in range(2 * n + 1):
            x_pred += Wm[i] * X_pred[i]

        P_pred = np.zeros((n, n))
        for i in range(2 * n + 1):
            diff = X_pred[i] - x_pred
            P_pred += Wc[i] * np.outer(diff, diff)

        # 프로세스 노이즈 추가
        Q = np.eye(n) * self.ukf_process_noise
        Q[3:6, 3:6] *= 10  # 속도에 대한 노이즈 증가
        Q[9:12, 9:12] *= 10  # 각속도에 대한 노이즈 증가
        P_pred += Q

        # 3. 측정 업데이트
        # 측정값: [x, y, z, rx, ry, rz]
        z = np.zeros(6)
        z[0:3] = tvec.flatten()
        z[3:6] = rvec.flatten()

        # 예측된 측정값 계산
        Z_pred = np.zeros((2 * n + 1, 6))
        for i in range(2 * n + 1):
            Z_pred[i, 0:3] = X_pred[i, 0:3]
            Z_pred[i, 3:6] = X_pred[i, 6:9]

        # 예측된 측정값의 평균
        z_pred = np.zeros(6)
        for i in range(2 * n + 1):
            z_pred += Wm[i] * Z_pred[i]

        # 측정 공분산 및 교차 공분산 계산
        Pzz = np.zeros((6, 6))
        Pxz = np.zeros((n, 6))

        for i in range(2 * n + 1):
            diff_z = Z_pred[i] - z_pred
            diff_x = X_pred[i] - x_pred
            Pzz += Wc[i] * np.outer(diff_z, diff_z)
            Pxz += Wc[i] * np.outer(diff_x, diff_z)

        # 측정 노이즈 추가
        R = np.eye(6) * self.ukf_measurement_noise
        Pzz += R

        # 칼만 게인 계산
        K = Pxz @ np.linalg.inv(Pzz)

        # 상태 및 공분산 업데이트
        marker_state.ukf_state = x_pred + K @ (z - z_pred)
        marker_state.ukf_P = P_pred - K @ Pzz @ K.T

        # 필터링된 값 저장
        marker_state.filtered_tvec = marker_state.ukf_state[0:3].reshape(3, 1)
        marker_state.filtered_rvec = marker_state.ukf_state[6:9].reshape(3, 1)

        marker_state.last_update_time = current_time

    # def apply_particle_filter(self, marker_state, tvec, rvec):
    #     """파티클 필터를 적용합니다."""
    #     current_time = time.time()
    #     dt = current_time - marker_state.last_update_time
    #
    #     # 파티클 초기화 (첫 호출 또는 오랜 시간 경과)
    #     if marker_state.particles_pos is None or dt > 1.0:
    #         marker_state.particles_pos = np.tile(tvec, (self.particle_count, 1, 1)) + \
    #                                      np.random.normal(0, self.particle_noise_pos, (self.particle_count, 3, 1))
    #         marker_state.particles_rot = np.tile(rvec, (self.particle_count, 1, 1)) + \
    #                                      np.random.normal(0, self.particle_noise_rot, (self.particle_count, 3, 1))
    #         marker_state.particles_weights = np.ones(self.particle_count) / self.particle_count
    #         marker_state.filtered_tvec = tvec.copy()
    #         marker_state.filtered_rvec = rvec.copy()
    #         marker_state.last_update_time = current_time
    #         return
    #
    #     # 1. 파티클 예측 (모션 모델 적용)
    #     # 간단한 랜덤 워크 모델 사용
    #     marker_state.particles_pos += np.random.normal(0, self.particle_noise_pos * dt,
    #                                                    (self.particle_count, 3, 1))
    #     marker_state.particles_rot += np.random.normal(0, self.particle_noise_rot * dt,
    #                                                    (self.particle_count, 3, 1))
    #
    #     # 2. 가중치 업데이트 (측정값과의 유사도 계산)
    #     pos_diff = np.sum((marker_state.particles_pos - tvec) ** 2, axis=(1, 2))
    #     rot_diff = np.sum((marker_state.particles_rot - rvec) ** 2, axis=(1, 2))
    #
    #     # 위치와 회전 차이에 대한 가중치 계산 (가우시안 커널)
    #     pos_weights = np.exp(-0.5 * pos_diff / (self.particle_noise_pos ** 2))
    #     rot_weights = np.exp(-0.5 * rot_diff / (self.particle_noise_rot ** 2))
    #
    #     # 전체 가중치 계산
    #     weights = pos_weights * rot_weights
    #
    #     # 가중치 정규화
    #     if np.sum(weights) > 0:
    #         weights = weights / np.sum(weights)
    #     else:
    #         weights = np.ones(self.particle_count) / self.particle_count
    #
    #     marker_state.particles_weights = weights
    #
    #     # 3. 상태 추정 (가중 평균)
    #     marker_state.filtered_tvec = np.sum(marker_state.particles_pos * weights[:, np.newaxis, np.newaxis], axis=0)
    #     marker_state.filtered_rvec = np.sum(marker_state.particles_rot * weights[:, np.newaxis, np.newaxis], axis=0)
    #
    #     # 4. 리샘플링 (가중치에 따라 파티클 재생성)
    #     if 1.0 / np.sum(weights ** 2) < self.particle_count / 2:  # 효과적인 파티클 수가 절반 이하면 리샘플링
    #         indices = np.random.choice(self.particle_count, self.particle_count, p=weights)
    #         marker_state.particles_pos = marker_state.particles_pos[indices]
    #         marker_state.particles_rot = marker_state.particles_rot[indices]
    #         marker_state.particles_weights = np.ones(self.particle_count) / self.particle_count
    #
    #     marker_state.last_update_time = current_time

    def apply_particle_filter(self, marker_state, tvec, rvec):
        """파티클 필터를 적용합니다."""
        current_time = time.time()
        dt = current_time - marker_state.last_update_time

        # 필요한 속성이 없으면 동적으로 추가
        if not hasattr(marker_state, 'particles_pos'):
            marker_state.particles_pos = None
        if not hasattr(marker_state, 'particles_rot'):
            marker_state.particles_rot = None
        if not hasattr(marker_state, 'particles_weights'):
            marker_state.particles_weights = None
        if not hasattr(marker_state, 'particles_vel'):
            marker_state.particles_vel = None
        if not hasattr(marker_state, 'particles_rot_vel'):
            marker_state.particles_rot_vel = None

        # 파티클 초기화 (첫 호출 또는 오랜 시간 경과)
        if marker_state.particles_pos is None or dt > 0.5:  # 타임아웃 감소
            marker_state.particles_pos = np.tile(tvec, (self.particle_count, 1, 1)) + \
                                         np.random.normal(0, self.particle_noise_pos, (self.particle_count, 3, 1))
            marker_state.particles_rot = np.tile(rvec, (self.particle_count, 1, 1)) + \
                                         np.random.normal(0, self.particle_noise_rot, (self.particle_count, 3, 1))
            # 속도 상태 추가
            marker_state.particles_vel = np.zeros((self.particle_count, 3, 1))
            marker_state.particles_rot_vel = np.zeros((self.particle_count, 3, 1))
            marker_state.particles_weights = np.ones(self.particle_count) / self.particle_count
            marker_state.filtered_tvec = tvec.copy()
            marker_state.filtered_rvec = rvec.copy()
            marker_state.last_update_time = current_time
            return

        # 1. 파티클 예측 (개선된 모션 모델 적용)
        # 속도 상태 업데이트 (가속도 랜덤 노이즈 추가)
        accel_noise = np.random.normal(0, self.particle_noise_pos * 2.0, (self.particle_count, 3, 1))
        rot_accel_noise = np.random.normal(0, self.particle_noise_rot * 2.0, (self.particle_count, 3, 1))

        # 속도 업데이트
        marker_state.particles_vel += accel_noise * dt
        marker_state.particles_rot_vel += rot_accel_noise * dt

        # 위치 및 회전 업데이트 (속도 기반)
        marker_state.particles_pos += marker_state.particles_vel * dt
        marker_state.particles_rot += marker_state.particles_rot_vel * dt

        # 2. 가중치 업데이트 (측정값과의 유사도 계산)
        pos_diff = np.sum((marker_state.particles_pos - tvec) ** 2, axis=(1, 2))
        rot_diff = np.sum((marker_state.particles_rot - rvec) ** 2, axis=(1, 2))

        # 위치와 회전 차이에 대한 가중치 계산 (가우시안 커널, 회전에 더 높은 가중치)
        pos_weights = np.exp(-0.5 * pos_diff / (self.particle_noise_pos ** 2))
        rot_weights = np.exp(-0.5 * rot_diff / (self.particle_noise_rot ** 2))

        # 회전 가중치에 더 높은 중요도 부여
        weights = pos_weights * (rot_weights ** 1.5)

        # 가중치 정규화
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(self.particle_count) / self.particle_count

        marker_state.particles_weights = weights

        # 3. 상태 추정 (가중 평균)
        marker_state.filtered_tvec = np.sum(marker_state.particles_pos * weights[:, np.newaxis, np.newaxis], axis=0)
        marker_state.filtered_rvec = np.sum(marker_state.particles_rot * weights[:, np.newaxis, np.newaxis], axis=0)

        # 4. 리샘플링 (더 자주 수행)
        effective_particles = 1.0 / np.sum(weights ** 2)
        if effective_particles < self.particle_count * 0.7:  # 임계값 증가 (70%)
            indices = np.random.choice(self.particle_count, self.particle_count, p=weights)
            marker_state.particles_pos = marker_state.particles_pos[indices]
            marker_state.particles_rot = marker_state.particles_rot[indices]
            marker_state.particles_vel = marker_state.particles_vel[indices]
            marker_state.particles_rot_vel = marker_state.particles_rot_vel[indices]

            # 리샘플링 후 속도에 약간의 노이즈 추가 (다양성 유지)
            marker_state.particles_vel += np.random.normal(0, self.particle_noise_pos * 0.5,
                                                           (self.particle_count, 3, 1))
            marker_state.particles_rot_vel += np.random.normal(0, self.particle_noise_rot * 0.5,
                                                               (self.particle_count, 3, 1))

            marker_state.particles_weights = np.ones(self.particle_count) / self.particle_count

        marker_state.last_update_time = current_time

    def apply_multi_marker_average(self, all_marker_data):
        """다중 마커 평균화를 적용합니다."""
        if not all_marker_data or len(all_marker_data) <= 1:
            return all_marker_data  # 마커가 없거나 하나만 있으면 평균화 불필요

        # 모든 마커의 위치와 회전 정보 수집
        all_tvecs = np.array([data['tvec'] for data in all_marker_data])
        all_rvecs = np.array([data['rvec'] for data in all_marker_data])

        # 마커 간 거리 계산
        marker_count = len(all_marker_data)
        distances = np.zeros((marker_count, marker_count))

        for i in range(marker_count):
            for j in range(i + 1, marker_count):
                dist = np.linalg.norm(all_tvecs[i] - all_tvecs[j])
                distances[i, j] = distances[j, i] = dist

        # 이상치 제거 (평균 거리의 2배 이상 떨어진 마커 제외)
        if marker_count > 2:
            avg_distance = np.mean(distances[distances > 0])
            valid_markers = []

            for i in range(marker_count):
                marker_dists = distances[i, distances[i] > 0]
                if np.mean(marker_dists) < avg_distance * 2:
                    valid_markers.append(i)

            if len(valid_markers) > 0:
                all_tvecs = all_tvecs[valid_markers]
                all_rvecs = all_rvecs[valid_markers]

                # 유효한 마커만 선택
                valid_marker_data = [all_marker_data[i] for i in valid_markers]
            else:
                valid_marker_data = all_marker_data
        else:
            valid_marker_data = all_marker_data

        # 평균 위치 계산
        avg_tvec = np.mean(all_tvecs, axis=0)

        # 회전은 직접 평균하기 어려우므로 쿼터니언으로 변환 후 평균
        quats = []
        for data in valid_marker_data:
            quat = data.get('quaternion')
            if quat is not None:
                quats.append(quat)

        if quats:
            # 쿼터니언 평균 계산 (간단한 방법)
            avg_quat = np.mean(quats, axis=0)
            avg_quat = avg_quat / np.linalg.norm(avg_quat)  # 정규화

            # 평균 쿼터니언을 회전 벡터로 변환
            avg_rot = Rotation.from_quat(avg_quat)
            avg_rvec = avg_rot.as_rotvec().reshape(3, 1)

            # 평균 오일러 각 계산
            avg_euler = avg_rot.as_euler('zyx', degrees=True)

            # 평균 정보를 첫 번째 마커 데이터에 추가
            avg_marker_data = valid_marker_data[0].copy()
            avg_marker_data['tvec'] = avg_tvec
            avg_marker_data['rvec'] = avg_rvec
            avg_marker_data['quaternion'] = avg_quat
            avg_marker_data['euler_angles'] = avg_euler
            avg_marker_data['is_averaged'] = True
            avg_marker_data['source_markers'] = [data['id'] for data in valid_marker_data]

            # 원본 마커 데이터에 평균 정보 추가
            for data in valid_marker_data:
                data['avg_tvec'] = avg_tvec
                data['avg_rvec'] = avg_rvec

            return valid_marker_data + [avg_marker_data]

        return valid_marker_data

    def get_pose_data(self, corners, ids, rvecs, tvecs):
        """
        ArUco 마커의 포즈 정보를 담은 리스트를 반환합니다.
        선택한 필터링 방식을 적용합니다.
        """
        aruco_data = []

        if ids is None or corners is None or rvecs is None or tvecs is None:
            return aruco_data

        current_time = time.time()
        dt = current_time - self.last_process_time
        self.last_process_time = current_time

        for i, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            marker_corners = corners[i]
            rvec = rvecs[i]
            tvec = tvecs[i]

            # 회전 벡터를 쿼터니언으로 변환
            rotation = Rotation.from_rotvec(rvec.flatten())
            quat = rotation.as_quat()  # [x, y, z, w] 형식

            # 정규화
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 0:
                quat = quat / quat_norm

            # 오일러 각 계산
            euler_angles = rotation.as_euler('zyx', degrees=True)

            # 마커 상태 초기화 또는 가져오기
            if marker_id not in self.marker_states:
                self.marker_states[marker_id] = MarkerState(marker_id, tvec, rvec)
                self.marker_states[marker_id].quat = quat
                self.marker_states[marker_id].euler_angles = euler_angles
                self.marker_states[marker_id].filtered_tvec = tvec.copy()
                self.marker_states[marker_id].filtered_rvec = rvec.copy()
                self.marker_states[marker_id].filtered_quat = quat.copy()
                self.marker_states[marker_id].filtered_euler = euler_angles.copy()

            marker_state = self.marker_states[marker_id]

            # 히스토리 저장
            marker_state.history_tvec.append(tvec.copy())
            marker_state.history_rvec.append(rvec.copy())

            # 선택된 필터 적용
            if self.filter_type == FilterType.LPF:
                self.apply_lpf(marker_state, tvec, rvec)
            elif self.filter_type == FilterType.EKF:
                self.apply_ekf(marker_state, tvec, rvec)
            elif self.filter_type == FilterType.PARTICLE:
                self.apply_particle_filter(marker_state, tvec, rvec)
            elif self.filter_type == FilterType.UKF:
                self.apply_ukf(marker_state, tvec, rvec)
            else:
                # 필터 없음
                marker_state.filtered_tvec = tvec.copy()
                marker_state.filtered_rvec = rvec.copy()

            # 필터링된 회전 벡터를 쿼터니언과 오일러 각으로 변환
            filtered_rotation = Rotation.from_rotvec(marker_state.filtered_rvec.flatten())
            marker_state.filtered_quat = filtered_rotation.as_quat()
            marker_state.filtered_euler = filtered_rotation.as_euler('zyx', degrees=True)

            # 마커 데이터 구성
            marker_data = {
                'id': marker_id,
                'corners': marker_corners,
                'rvec': marker_state.filtered_rvec,
                'tvec': marker_state.filtered_tvec,
                'raw_rvec': rvec,
                'raw_tvec': tvec,
                'quaternion': marker_state.filtered_quat,
                'euler_angles': marker_state.filtered_euler,
                'rotation_z': marker_state.filtered_euler[0],
                'filter_type': self.filter_type.name
            }

            aruco_data.append(marker_data)

        # 다중 마커 평균화 적용 (옵션)
        if self.use_marker_average and len(aruco_data) > 1:
            aruco_data = self.apply_multi_marker_average(aruco_data)

        return aruco_data


def main():
    parser = argparse.ArgumentParser(description="고급 ArUco 마커 탐지기 (다양한 필터링 옵션)")
    parser.add_argument("--camera", type=int, default=config.USB_CAMERA_INDEX,
                        help=f"카메라 인덱스 (기본값: {config.USB_CAMERA_INDEX})")
    parser.add_argument("--width", type=int, default=config.CAMERA_DEFAULT_WIDTH,
                        help=f"카메라 프레임 너비 (기본값: {config.CAMERA_DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=config.CAMERA_DEFAULT_HEIGHT,
                        help=f"카메라 프레임 높이 (기본값: {config.CAMERA_DEFAULT_HEIGHT})")
    parser.add_argument("--fps", type=int, default=30,
                        help="카메라 목표 FPS (기본값: 30)")
    parser.add_argument("--calib", type=str, default=config.GLOBAL_CALIBRATION_FILE,
                        help="카메라 캘리브레이션 YAML 파일 경로")
    parser.add_argument("--aruco_dict", type=str, default=config.DEFAULT_ARUCO_TYPE,
                        help=f"ArUco 사전 타입 (기본값: {config.DEFAULT_ARUCO_TYPE})")
    parser.add_argument("--marker_length", type=float, default=config.USB_ARUCO_LENGTH,
                        help=f"마커 크기(미터) (기본값: {config.USB_ARUCO_LENGTH}m)")
    parser.add_argument("--filter", type=str, choices=['none', 'lpf', 'ekf', 'particle', 'ukf', 'multi_avg'],
                        default='lpf', help="필터링 방식 선택 (기본값: lpf)")
    parser.add_argument("--lpf_tau", type=float, default=0.3,
                        help="LPF 시정수 (기본값: 0.3, 높을수록 더 강한 필터링)")
    parser.add_argument("--ekf_process_noise", type=float, default=0.01,
                        help="EKF 프로세스 노이즈 (기본값: 0.01)")
    parser.add_argument("--ukf_alpha", type=float, default=0.1,
                        help="UKF 알파 파라미터 (기본값: 0.1)")
    parser.add_argument("--ukf_beta", type=float, default=2.0,
                        help="UKF 베타 파라미터 (기본값: 2.0)")
    parser.add_argument("--particle_count", type=int, default=100,
                        help="파티클 필터 파티클 수 (기본값: 100)")
    parser.add_argument("--save_dir", type=str, default="captured_frames",
                        help="프레임 저장 디렉토리 (기본값: captured_frames)")
    parser.add_argument("--show_raw", action='store_true',
                        help="필터링되지 않은 원시 포즈도 함께 표시")

    args = parser.parse_args()

    # 필터 타입 설정
    filter_type_map = {
        'none': FilterType.NONE,
        'lpf': FilterType.LPF,
        'ekf': FilterType.EKF,
        'ukf': FilterType.UKF,
        'particle': FilterType.PARTICLE,
        'multi_avg': FilterType.MULTI_MARKER_AVG
    }
    filter_type = filter_type_map[args.filter]

    # 필터 파라미터 설정
    filter_params = {
        'lpf_tau': args.lpf_tau,
        'lpf_Ts': 1.0 / args.fps,
        'ekf_process_noise': args.ekf_process_noise,
        'ekf_measurement_noise_pos': 0.1,
        'ekf_measurement_noise_rot': 0.2,
        'ukf_alpha': args.ukf_alpha,
        'ukf_beta': args.ukf_beta,
        'ukf_kappa': 0.0,
        'ukf_process_noise': 0.01,
        'ukf_measurement_noise': 0.1,
        'particle_count': args.particle_count,
        'particle_noise_pos': 0.01,
        'particle_noise_rot': 0.05
    }

    try:
        # 카메라 핸들러 초기화
        cam_handler = CameraHandler(
            index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            buffer_size=1,
            calibration_file=args.calib
        )

        # 카메라 매트릭스와 왜곡 계수 가져오기
        K, D = cam_handler.get_effective_camera_parameters()

        if K is None or D is None:
            print("[ERROR] 카메라 파라미터를 가져올 수 없습니다. 캘리브레이션 파일이 올바른지 확인하세요.")
            return

        # ArUco 프로세서 초기화
        aruco_dict_type = config.ARUCO_DICT.get(args.aruco_dict, cv2.aruco.DICT_6X6_250)
        aruco_processor = AdvancedArucoProcessor(
            camera_matrix=K,
            dist_coeffs=D,
            aruco_dict_type=aruco_dict_type,
            marker_length=args.marker_length,
            filter_type=filter_type,
            filter_params=filter_params
        )

        # 창 생성
        window_name = f"고급 ArUco 탐지 ({args.filter} 필터) - 카메라 {args.camera}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

        print(f"[INFO] 필터 타입: {args.filter}")
        print(f"[INFO] 'q'키: 종료, 's'키: 프레임 저장 ({args.save_dir})")

        frame_count = 0
        start_time = time.time()
        display_fps = 0

        while True:
            # 왜곡 보정된 프레임 캡처
            frame = cam_handler.capture_frame(undistort=True)

            if frame is None:
                print("[WARN] 프레임 캡처 실패, 다시 시도합니다.")
                time.sleep(0.1)
                continue

            # ArUco 마커 검출
            corners, ids, rejected, rvecs, tvecs = aruco_processor.detect_markers(frame)

            # 마커 정보 처리 및 시각화
            aruco_data_list = aruco_processor.get_pose_data(corners, ids, rvecs, tvecs)

            if aruco_data_list:
                print("\n--- 감지된 ArUco 마커 정보 ---")
                for marker_data in aruco_data_list:
                    marker_id = marker_data.get('id', 'N/A')
                    tvec_m = marker_data.get('tvec')
                    rot_z_deg = marker_data.get('rotation_z')
                    filter_type = marker_data.get('filter_type', 'UNKNOWN')

                    # 다중 마커 평균인 경우 특별 표시
                    if marker_data.get('is_averaged', False):
                        source_markers = marker_data.get('source_markers', [])
                        print(f" 평균 마커 (출처: {source_markers}): "
                              f"위치 = [{tvec_m[0, 0]:.3f}, {tvec_m[1, 0]:.3f}, {tvec_m[2, 0]:.3f}] | "
                              f"Z축 회전 = {rot_z_deg:.1f}°")
                    else:
                        print(f" ID {marker_id} ({filter_type}): "
                              f"위치 = [{tvec_m[0, 0]:.3f}, {tvec_m[1, 0]:.3f}, {tvec_m[2, 0]:.3f}] | "
                              f"Z축 회전 = {rot_z_deg:.1f}°")

                        # 원시 데이터도 표시 (옵션)
                        if args.show_raw:
                            raw_tvec = marker_data.get('raw_tvec')
                            raw_rvec = marker_data.get('raw_rvec')
                            raw_rot = Rotation.from_rotvec(raw_rvec.flatten())
                            raw_euler = raw_rot.as_euler('zyx', degrees=True)
                            print(
                                f"   원시 데이터: 위치 = [{raw_tvec[0, 0]:.3f}, {raw_tvec[1, 0]:.3f}, {raw_tvec[2, 0]:.3f}] | "
                                f"Z축 회전 = {raw_euler[0]:.1f}°")

                print("----------------------------------")

            # 마커 시각화
            aruco_processor.draw_detected_markers(frame, corners, ids, None)

            # 필터링된 포즈로 축 그리기
            for marker_data in aruco_data_list:
                if marker_data.get('is_averaged', False):
                    # 평균 마커는 다른 색으로 표시
                    rvec = marker_data.get('rvec')
                    tvec = marker_data.get('tvec')

                    # 원래 drawFrameAxes 대신 직접 그리기 (다른 색상 사용)
                    axis_length = args.marker_length * 0.7
                    cv2.drawFrameAxes(frame, K, D, rvec, tvec, axis_length)

                    # # 평균 마커임을 표시하는 텍스트
                    # imgpts, _ = cv2.projectPoints(np.array([[0, 0, 0]]), rvec, tvec, K, D)
                    # corner = tuple(imgpts[0, 0].astype(int))
                    # cv2.putText(frame, "AVG", (corner[0] + 10, corner[1] - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    try:
                        # 모든 입력을 명시적으로 float32로 변환
                        points_3d = np.array([[0, 0, 0]], dtype=np.float32)
                        rvec_float = np.array(rvec, dtype=np.float32)
                        tvec_float = np.array(tvec, dtype=np.float32)
                        K_float = np.array(K, dtype=np.float32)
                        D_float = np.array(D, dtype=np.float32)

                        imgpts, _ = cv2.projectPoints(points_3d, rvec_float, tvec_float, K_float, D_float)

                        # 이제 imgpts 사용
                        corner = tuple(imgpts[0, 0].astype(int))
                        cv2.putText(frame, f"ID {marker_id}", corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"[ERROR] projectPoints 오류: {e}")
                        # 오류 처리 로직

                else:
                    # 일반 마커
                    rvec = marker_data.get('rvec')
                    tvec = marker_data.get('tvec')
                    aruco_processor.draw_axes(frame, rvec, tvec)

                    # 원시 데이터도 표시 (옵션)
                    if args.show_raw:
                        raw_rvec = marker_data.get('raw_rvec')
                        raw_tvec = marker_data.get('raw_tvec')
                        # 원시 데이터는 점선이나 다른 색으로 표시할 수 있음
                        # 여기서는 간단히 작은 크기로 표시
                        aruco_processor.draw_axes(frame, raw_rvec, raw_tvec, length=args.marker_length * 0.3)

            # FPS 계산
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # FPS 및 필터 정보 표시
            cv2.putText(frame, f"FPS: {display_fps:.1f} | Filter: {args.filter.upper()}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 화면 표시
            cv2.imshow(window_name, frame)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF

            # 's' 키: 프레임 저장
            if key == ord('s'):
                import os
                import datetime

                if not os.path.exists(args.save_dir):
                    try:
                        os.makedirs(args.save_dir)
                        print(f"[INFO] 디렉토리 생성: {args.save_dir}")
                    except OSError as e:
                        print(f"[ERROR] 디렉토리 생성 실패: {e}")
                        continue

                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = os.path.join(args.save_dir, f'frame_{timestamp}.jpg')

                try:
                    cv2.imwrite(filename, frame)
                    print(f"[INFO] 프레임 저장됨: {filename}")
                except Exception as e:
                    print(f"[ERROR] 프레임 저장 실패: {e}")

            # 'q' 키: 종료
            elif key == ord('q'):
                print("[INFO] 'q' 키 입력 감지. 종료합니다.")
                break

            # 창이 닫혔는지 확인
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] 창이 닫혔습니다. 종료합니다.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] 키보드 인터럽트 감지. 종료합니다.")
    except Exception as e:
        print(f"[ERROR] 예기치 않은 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 리소스 정리
        if 'cam_handler' in locals():
            cam_handler.release_camera()
        cv2.destroyAllWindows()
        print("[INFO] 리소스 정리 완료. 프로그램 종료.")


if __name__ == "__main__":
    main()
