# Stroke model inference prototype

BrainOn 개발 초기에 만든 **FastAPI 추론 API 시제품**입니다. 이 저장소는 최종 의료영상 분할 모델의 학습·평가 코드를 담고 있지 않습니다. 현재 BrainOn의 nnU-Net·SegResNet 실험은 별도의 팀 모델 저장소에서 관리합니다.

## 구현 범위

- `GET /health`: 서비스 상태 확인
- `POST /predict`: 업로드한 이미지를 회색조 128×128로 변환한 뒤 TensorFlow/Keras 모델로 마스크 추론
- GCS에서 모델 파일을 내려받아 로드하는 코드
- Dockerfile 및 Cloud Build/Cloud Run 배포 설정 초안

## 저장소 구조

| 경로 | 내용 |
| :--- | :--- |
| `app/main.py` | FastAPI 엔드포인트 |
| `app/model.py` | GCS 모델 로딩, 이미지 전처리, 추론 |
| `Dockerfile`, `cloudbuild.yaml` | 컨테이너 빌드와 배포 설정 |
| `requirements.txt` | 시제품 의존성 |

## 상태와 한계

이 코드는 초기 API 흐름을 확인하기 위한 시제품입니다. 학습 코드, 데이터셋, 평가 결과, 모델 가중치는 포함하지 않습니다. `app/model.py`는 시작 시 GCS의 모델 파일을 불러오므로 해당 파일과 접근 권한이 없으면 실행할 수 없습니다. 입력 또한 최종 BrainOn의 CT·MRI·MRA NIfTI 추론 규격과 다릅니다.

BrainOn 프로젝트의 실제 모델 연구와 서비스 연결 경험은 [포트폴리오 요약](https://github.com/duddl6292/duddl6292/blob/main/docs/brainon-ai.md)에 정리했습니다.
