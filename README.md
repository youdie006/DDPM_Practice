# Denoising Diffusion Probabilistic Models (DDPM)

## 개요
KAIST CS492(D): Diffusion Models and Their Applications (Fall 2024)
Programming Assignment 1
## Task 1: 2D Swiss-Roll DDPM

### 구현 내용
#### 1-1. `network.py` - SimpleNet 구현
- TimeLinear 레이어들을 연결한 noise prediction 네트워크 구성
- 네트워크 구조: [dim_in] → [dim_hids] → [dim_out]
- 마지막 레이어 제외하고 ReLU activation 적용

#### 1-2. `ddpm.py` - Forward/Reverse Process 구현
- `q_sample()`: Forward diffusion process (x0 → xt)
  - 수식: q(xt|x0) = N(xt; √(ᾱt)x0, (1-ᾱt)I)
- `p_sample()`: Reverse process 한 스텝 (xt → xt-1)
  - 평균과 분산을 계산하여 denoising
- `p_sample_loop()`: 전체 reverse process (xT → x0)
  - 순수 노이즈에서 시작하여 반복적으로 denoising

#### 1-3. `ddpm.py` - Loss Function 구현
- `compute_loss()`: Simplified noise matching loss
  - L = E[||ε - εθ(xt, t)||²]

### 실행 결과

#### 1. Target & Prior 분포
![Target and Prior](./output/output1.png)
- Swiss-roll 타겟 분포와 가우시안 prior 분포 시각화

#### 2. Forward Process (q(x_t))
![Forward Process](./output/output2.png)
- t=0부터 t=450까지 점진적으로 노이즈가 추가되는 과정

#### 3. 학습 Loss Curve
![Loss Curve](./output/output4.png)
- 5000 iteration 학습
- 최종 loss: 0.2512
- 학습 속도: 140.79it/s (총 35초 소요)

#### 4. DDPM 샘플링 결과
![Sampling Result](./output/output3.png)
- 4999 iteration에서의 샘플링 결과
- Swiss-roll 패턴을 성공적으로 재현

#### 5. 최종 평가
![Final Evaluation](./output/output5.png)
- **Chamfer Distance: 15.2181** (목표: < 20) 
- 타겟 분포와 생성된 샘플이 잘 일치

## Task 2: Image Diffusion

### 구현 내용
#### `scheduler.py` - DDPMScheduler 구현
- `add_noise()`: Forward diffusion process (Task 1의 q_sample과 동일)
- `step()`: Reverse process 한 스텝 (Task 1의 p_sample과 동일)

### 실행 결과
(진행 예정)

## 진행 상황
- [O] Task 1: 2D Swiss-Roll DDPM 구현 완료
- [O] Task 2: Image Diffusion 구현 완료 (학습 예정)

## 원본 저장소
https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment1-DDPM

## 저작권
모든 저작권은 원본 저장소(KAIST Visual AI Group)에 있음