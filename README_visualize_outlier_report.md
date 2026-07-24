# visualize_outlier_report 사용 설명서

`visualize_outlier_report.py`는 `analyze_outliers.py`의 검수 과정을 7페이지짜리
PDF 보고서로 시각화하는 프로그램이다. 최종 라벨뿐 아니라 각 에피소드가 왜
후보가 되었는지 그래프로 확인할 수 있다.

## 1. 실행

작업 폴더에서 가상환경을 활성화한 뒤 실행한다.

```bash
cd /home/robotis/Downloads/data_practice
source venv/bin/activate
python visualize_outlier_report.py --dataset-dir "데이터셋_폴더"
```

예시:

```bash
python visualize_outlier_report.py \
  --dataset-dir "Task_000216_Sub3_Put_Carrot_Plate_L2R_Speed40_Lift18_KBK_NJW_fail_example_lerobot"
```

기본 출력 파일:

```text
데이터셋_폴더/analysis_results/outlier_analysis_report.pdf
```

## 2. 필요한 파일

이 프로그램은 다음 코드와 데이터가 필요하다.

```text
프로젝트/
├── analyze_outliers.py
├── visualize_outlier_report.py
├── execution_logging.py
└── 데이터셋/
    ├── meta/info.json
    ├── data/chunk-*/episode_*.parquet
    └── annotations/chunk-*/episode_*.json  # 선택 사항
```

시각화 프로그램은 기존 결과 파일을 단순히 읽는 것이 아니라
`analyze_outliers.py`를 내부에서 다시 실행한다. 따라서 분석 프로그램과 동일하게
Parquet 및 metadata에 접근할 수 있어야 하며, PDF 생성에도 분석 시간이 든다.

다음 파일이 있으면 보고서에 추가로 반영하지만 없어도 PDF는 생성된다.

- `meta/manual_review_summary.json`: 사람이 검수한 결과와 모델 라벨 비교
- `analysis_results/camera_delay_delete_episodes.json`: 별도로 정리된 카메라
  지연 삭제 대상 표시

## 3. PDF 구성

1. **검증 요약**: 전체 라벨 수, 신호별 검출 수, 손 사용 모드와 태스크 복잡도
2. **무결성·길이·무동작**: 에피소드 길이, 전체 관절 이동량, 움직인 프레임 비율
3. **수행 모드와 태스크 구조**: 좌우 손 활동 비율, 시간 구간별 활동,
   annotation 구조
4. **Isolation Forest**: 손 사용 모드별 isolation score와 주요 feature
5. **PCA + HDBSCAN**: 저차원 수행 분포, 군집 noise와 보수적 후보 비교
6. **DTW**: 모드 내부 궤적 차이와 최종 후보의 동작 형태
7. **최종 교차판정**: 후보별 검출 신호, `delete`·`validate`·`pass` 결과

## 4. 그래프 색상

- 빨강: `delete` 또는 강한 이상 신호
- 주황: `validate`
- 초록: `pass`
- 파랑: 오른손 우세 모드
- 보라: 왼손 우세 모드
- 청록: 양손 균형 모드

그래프에 표시된 에피소드 번호로 원본 영상을 찾아 최종 검수하면 된다.

## 5. 출력 위치 변경

```bash
python visualize_outlier_report.py \
  --dataset-dir "데이터셋_폴더" \
  --output-pdf "./원하는_폴더/report.pdf"
```

사용 가능한 옵션은 다음 명령으로 확인한다.

```bash
python visualize_outlier_report.py --help
```

| 옵션 | 의미 | 기본값 |
|---|---|---|
| `--dataset-dir PATH` | 분석할 LeRobot 데이터셋 | 코드의 기본 데이터셋 |
| `--output-pdf PATH` | 생성할 PDF 경로 | 데이터셋의 `analysis_results/` |

## 6. 해석할 때 주의할 점

- PDF는 이상 후보를 설명하는 검수 자료이며 작업 실패를 완전히 자동 확정하지 않는다.
- `validate`는 정상 수행의 변형일 수 있으므로 해당 영상을 직접 확인해야 한다.
- 카메라 지연의 상세 측정값은 이 PDF가 아니라 `detect_camera_delay.py`의
  `camera_delay_results.csv`와 `camera_delay_summary.json`에서 확인한다.
- 실행 기록은 프로젝트 폴더의 `terminal_execution.log`에 누적된다.

