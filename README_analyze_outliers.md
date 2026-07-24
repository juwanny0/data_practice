# analyze_outliers 사용 설명서

`analyze_outliers.py`는 LeRobot 데이터셋의 관절 상태와 annotation을 비교해 각
에피소드를 `delete`, `validate`, `pass` 중 하나로 분류하는 프로그램이다.

이 결과는 자동 판정 보조 자료다. 특히 `validate`는 오류 확정이 아니라 사람이
영상을 확인할 후보라는 뜻이다.

## 1. 실행

작업 폴더에서 가상환경을 활성화한 뒤 실행한다.

```bash
cd /home/robotis/Downloads/data_practice
source venv/bin/activate
python analyze_outliers.py --dataset-dir "데이터셋_폴더"
```

예시:

```bash
python analyze_outliers.py \
  --dataset-dir "Task_000241_throw_garbage_into_the_trash_bin_lift17_LJC_LDY_fail_example_lerobot"
```

분석 결과는 기본적으로 해당 데이터셋의 `analysis_results/`에 저장된다.

## 2. 입력 데이터

최소한 다음 구조가 필요하다.

```text
데이터셋/
├── meta/info.json
├── data/chunk-*/episode_*.parquet
└── annotations/chunk-*/episode_*.json  # 없어도 실행 가능
```

- `info.json`: FPS, 로봇 종류, `observation.state`의 관절 이름
- Parquet: 에피소드별 관절 상태 시계열
- annotation: subtask/skill 개수와 시간 범위 분석에 사용

유효한 Parquet 에피소드가 최소 5개 필요하다.

## 3. 분석 과정

프로그램은 다음 신호를 순서대로 계산한다.

1. **파일 무결성**: 빈 데이터, state 누락, shape 오류, NaN/Inf, 중복 여부
2. **기본 움직임**: 길이, 전체 관절 이동량, 실제로 움직인 프레임 비율
3. **순간 충격**: 최대 속도·가속도·jerk가 모두 데이터셋의 강한 3-IQR 기준을
   넘는지 확인
4. **손 사용 모드**: 왼손/오른손 활동 비율로 `left_dominant`,
   `right_dominant`, `balanced` 구분
5. **모션 feature**: 관절별 속도, 가속도, 위치 범위, 활동률, 종료 변위와
   시간 구간별 활동량 계산
6. **Annotation 구조**: 일반적인 subtask/skill 구조와 다른 에피소드 탐색
7. **Isolation Forest**: 같은 손 모드 안에서 feature가 희귀한 에피소드 탐색
8. **PCA + HDBSCAN**: feature를 필요한 설명분산만큼 축소하고 밀도가 낮은
   수행 형태 탐색
9. **DTW**: 길이가 다른 관절 궤적의 시간 차이를 보정해 형태가 다른 수행 탐색
10. 여러 신호의 강도와 일치 여부를 조합해 최종 라벨 결정

복잡한 태스크는 정상 수행 방식이 다양하므로 annotation 구조를 더 중시하고,
단순 태스크는 전체 모션 feature를 더 직접적으로 사용한다.

## 4. 라벨 의미

- `delete`: 파일 손상, 정지 데이터, 심한 조기 종료 등 삭제 근거가 강함
- `validate`: 희귀하거나 의심스럽지만 정상일 수도 있어 영상 확인이 필요함
- `pass`: 현재 사용한 신호에서는 뚜렷한 문제가 발견되지 않음

`pass`는 영상 내용까지 완벽히 정상임을 보증하지 않는다. 카메라 지연처럼 영상과
관절 상태의 시간 관계가 필요한 문제는 `detect_camera_delay.py`로 별도 검사한다.

## 5. 출력 파일

```text
analysis_results/
├── episode_outlier_labels.json
├── outlier_summary.json
├── outlier_results.csv
└── episode_outlier_analysis.png
```

- `episode_outlier_labels.json`: 세 라벨별 에피소드 번호
- `outlier_summary.json`: 분석 조건, 임계값, 신호별 후보와 경고
- `outlier_results.csv`: 에피소드별 feature, 라벨, 판정 이유
- `episode_outlier_analysis.png`: 길이 분포와 PCA 공간의 간단한 시각화

터미널 실행 내용은 작업 폴더의 `terminal_execution.log`에도 기록된다.

## 6. 주요 옵션

```bash
python analyze_outliers.py --help
```

| 옵션 | 의미 | 기본값 |
|---|---|---:|
| `--output-dir PATH` | 결과 저장 위치 변경 | 데이터셋의 `analysis_results/` |
| `--task-feature-prefixes PREFIX ...` | 분석할 관절 이름 접두사 | `arm_ gripper_` |
| `--pca-variance-target FLOAT` | PCA가 보존할 설명분산 비율 | `0.90` |
| `--dtw-resample-length INT` | DTW 비교용 궤적 길이 | `60` |
| `--dtw-window INT` | DTW 시간 정렬 허용 폭 | `12` |
| `--skip-dtw` | 시간이 오래 걸리는 DTW 생략 | 사용 안 함 |
| `--no-write-results` | 파일을 저장하지 않고 터미널 결과만 확인 | 사용 안 함 |
| `--quiet` | 중간 진행 출력 최소화 | 사용 안 함 |

일반적인 사용에서는 기본값을 유지하는 것이 권장된다. `--skip-dtw`를 사용하면
실행은 빨라지지만 궤적 형태 이상을 판정하는 근거 하나가 빠진다.

