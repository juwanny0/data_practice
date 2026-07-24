# detect_camera_delay 사용 설명서

`detect_camera_delay.py`는 LeRobot의 MP4 영상이 관절 상태보다 늦게 기록된
에피소드를 찾는 프로그램이다. 영상의 PTS와 프레임 수가 정상이어도 영상 내용
자체가 늦는 문제를 검출하기 위해 사용한다.

## 1. 실행

작업 폴더에서 가상환경을 활성화한 뒤 실행한다.

```bash
cd /home/robotis/Downloads/data_practice
source venv/bin/activate
python detect_camera_delay.py "데이터셋_폴더"
```

예시:

```bash
python detect_camera_delay.py \
  "Task_000216_Sub3_Put_Carrot_Plate_L2R_Speed40_Lift18_KBK_NJW_fail_example_lerobot"
```

아무 옵션도 주지 않으면 전체 에피소드와 `meta/info.json`에 등록된 모든
카메라를 분석한다.

## 2. 입력 데이터

다음 구조가 필요하다.

```text
데이터셋/
├── meta/info.json
├── data/chunk-*/episode_*.parquet
└── videos/chunk-*/VIDEO_KEY/episode_*.mp4
```

- `info.json`: FPS, 관절 이름, video key
- Parquet: `observation.state` 관절 상태 시계열
- MP4: Parquet 에피소드와 번호가 일치하는 카메라 영상

## 3. 검출 원리

1. 영상을 기본 `96×64` grayscale로 축소하고 밝기·대비를 정규화한다.
2. 국소적인 사람/물체 변화의 영향을 제한한 프레임 차이로
   **영상 움직임 신호**를 만든다.
3. 손목 카메라는 해당 팔 관절, 고정 카메라는 양팔 관절의 속도로
   **관절 움직임 신호**를 만든다. 그리퍼는 제외한다.
4. 두 신호의 교차상관을 여러 시간차에서 계산한다.
5. 상관이 가장 높은 시간차, 0초 대비 상관 증가량, peak의 뚜렷함을 함께
   확인한다.
6. 영상이 관절보다 늦고 모든 신뢰도 기준을 통과할 때만 지연으로 판정한다.

양수 `delay_frames`는 영상이 관절 상태보다 늦다는 뜻이다. 예를 들어 15 FPS에서
`+30 frames`는 약 2초의 영상 지연이다. 음수 값은 영상 선행을 뜻하지만 현재
자동 지연 이상으로 확정하지 않는다.

## 4. 상태값 의미

- `delay`: 신뢰도 조건을 통과한 영상 지연
- `pass`: 분석은 가능했지만 지연 판정 기준 미충족
- `insufficient_motion`: 영상 또는 대응 관절 움직임이 부족해 판단하기 어려움
- `recording_error`: 영상은 움직이지만 전체 관절 상태가 거의 고정된 녹화 불일치
- `error`: 파일 누락 등으로 해당 episode-camera 분석 실패

고정 카메라는 사람, 물체, 반대쪽 팔 움직임이 섞이므로 손목 카메라보다 더 엄격한
판정 기준을 사용한다.

## 5. 출력 파일

결과는 기본적으로 데이터셋의 `analysis_results/`에 저장된다.

```text
analysis_results/
├── camera_delay_results.csv
├── camera_delay_summary.json
├── camera_delay_delete_episodes.json
└── camera_delay_cache/
```

- `camera_delay_results.csv`: 모든 episode-camera 쌍의 상세 측정값
- `camera_delay_summary.json`: 지연/녹화 오류 에피소드와 실행 조건 요약
- `camera_delay_delete_episodes.json`: 1차 검수에서 삭제할 확정 지연 에피소드
- `camera_delay_cache/`: 재실행 시 영상 디코딩 시간을 줄이는 캐시

터미널 실행 내용은 작업 폴더의 `terminal_execution.log`에도 기록된다.

`camera_delay_delete_episodes.json`의 `delete_episode_indices`만
`camera_delay` 사유로 삭제한다. `pass`, `insufficient_motion`,
`recording_error`, `error`는 이 목록에 포함하지 않는다. 삭제 후 남은 규격화
데이터셋에 `analyze_outliers.py`를 2차 검사로 별도 실행한다. CSV와 summary
JSON은 판정 근거 확인용이며 2차 검사의 필수 입력은 아니다.

## 6. 일부만 빠르게 검사하기

특정 에피소드:

```bash
python detect_camera_delay.py "데이터셋_폴더" --episodes 72-76
```

여러 번호와 범위를 함께 지정:

```bash
python detect_camera_delay.py "데이터셋_폴더" --episodes 1,3,8-10
```

특정 카메라만 지정:

```bash
python detect_camera_delay.py "데이터셋_폴더" \
  --cameras observation.images.cam_left_wrist
```

사용 가능한 정확한 video key는 데이터셋의 `meta/info.json`에서 확인한다.

## 7. 주요 옵션

```bash
python detect_camera_delay.py --help
```

| 옵션 | 의미 | 기본값 |
|---|---|---:|
| `--output-dir PATH` | 결과 저장 위치 변경 | 데이터셋의 `analysis_results/` |
| `--episodes SPEC` | 분석할 에피소드 번호/범위 | 전체 |
| `--cameras KEY ...` | 분석할 video key | 전체 |
| `--analysis-seconds FLOAT` | 시작 부분의 관절 분석 길이 | `8.0`초 |
| `--max-lag-seconds FLOAT` | 탐색할 최대 시간차 | `3.0`초 |
| `--min-delay-seconds FLOAT` | 이상으로 볼 최소 영상 지연 | `0.5`초 |
| `--width`, `--height` | 분석용 영상 해상도 | `96`, `64` |
| `--workers INT` | 동시에 처리할 에피소드 수 | `2` |
| `--decoder-threads INT` | 영상 디코더별 thread 수 | `1` |
| `--cache-dir PATH` | 캐시 위치 변경 | 결과 폴더 내부 |
| `--no-cache` | 캐시를 사용하지 않음 | 사용 안 함 |
| `--quiet` | 진행 출력 최소화 | 사용 안 함 |

상관 임계값과 smoothing 옵션도 제공되지만, 특별한 검증 없이 기본값을 낮추면
정상 영상을 지연으로 오인할 수 있으므로 일반적으로 변경하지 않는 것이 좋다.

## 8. 해석할 때 주의할 점

- 움직임이 너무 적으면 지연이 없다는 뜻이 아니라 **판단할 신호가 부족한 것**이다.
- 이 프로그램은 영상 내용과 관절 움직임의 시간차를 찾는다. 물체 인식 실패,
  작업 성공 여부, 카메라 화질 저하는 별도의 문제다.
- 실제 파일 삭제 전에는 `camera_delay_delete_episodes.json`과 상세 진단 결과를
  함께 확인하는 것이 안전하다.
