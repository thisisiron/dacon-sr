# Dacon AIhub Super Resolution

## Description
- Team: 오예!
- 구성원: gradient, Capshulization
- Score: 24.95184 (2nd)

## 개발 환경 및 라이브러리
- 개발 환경
    ```
    Ubuntu version  : 18.04
    GPU             : V100 (4개)
    Cuda version    : 11.1
    Python version  : 3.8
    Pytorch version : 1.12.1
    ```

1. Code download
    ```
    git clone https://github.com/thisisiron/dacon-sr.git
    cd dacon-sr
    ```
2. 가상환경 구축 및 라이브러리 다운로드
    ```
    virtualenv dacon
    source dacon/bin/activate
    pip install -r requirements.txt
    ```

## Data Download & Image Preprocessing
1. Data download [[link]](https://dacon.io/competitions/official/235977/data)
    ```
    ├── dacon-sr
    │   ├── open.zip
    │   │ 
    ```
2. open.zip 압축 해제
    ```
    unzip open.zip
    ```
    ```
    ├── dacon-sr
    │   ├── train
    │   │   ├── lr
    │   │   ├── hr
    │   ├── test
    │   │   ├── lr
    │   ├── train.csv
    │   ├── test.csv
    ```
3. Image Preprocessing (Patch 만들기)
    ```
    python preprocessing.py
    ```
4. `./data/train`에 `hr`과 `lr`에 npy로 파일 생성
    ```
    ├── dacon-sr
    │   ├── data
    │   │   ├── train
    │   │   │   ├── lr
    │   │   │   │   ├── p128x128_r512x512_s32x32
    │   │   │   ├── hr
    │   │   │   │   ├── p512x512_r2048x2048_s128x128
    │   ├── train.csv
    │   ├── test.csv
    │   ├── train
    │   ├── test
    ```
 
## Training
<sub> 실행 명령어는 아래 Command를 참고해주세요 </sub>
1. SwinIR + L1 Loss 학습 진행 (Epoch 34까지 진행, 해당 Epoch 근처 Test 성능 향상이 거의 없었음)
2. 1번에서 진행한 실험 weight를 이용하여 아래 두 가지 실험 진행
    1. SwinIR + L1 Loss + L2 Loss (Epoch 5까지 진행)
    2. SwinIR + L1 Loss + LR 1e-5 (Epoch 11까지 진행)

### Command
- EXP_NAME: experiments에 생성된 실험 디렉토리 (실행날짜 + name 옵션에 설정한 NAME으로 구성)
    - Ex. `experiments/10051136_swinir_no-acti_epoch26_lr1e5`
- ckpt_NUM: 해당 실험 디렉토리에서 weight 아래에 있는 ckpt_NUM 디렉토리
    - Ex. `10051136_swinir_no-acti_epoch26_lr1e5/weight/ckpt_10`

1. SwinIR + L1 Loss
    ```
    python -m torch.distributed.run --nproc_per_node 4 main.py --gene swinir \
                                                               --cfg models/swinir.yaml \
                                                               --data p128x128_r512x512_s32x32 \
                                                               --data_hr p512x512_r2048x2048_s128x128 \
                                                               --batch 4 \
                                                               --name NAME
    ```
2. SwinIR + L1 Loss + L2 Loss (10:5)
    ```
    python -m torch.distributed.run --nproc_per_node 4 main.py --gene swinir \
                                                               --cfg models/swinir.yaml \
                                                               --data p128x128_r512x512_s32x32 \
                                                               --data_hr p512x512_r2048x2048_s128x128 \
                                                               --batch 4 \
                                                               --lamb_l2 5 \
                                                               --name NAME
    ```
3. SwinIR + L1 Loss + LR 1e-5
    ```
    python -m torch.distributed.run --nproc_per_node 4 main.py --gene swinir \
                                                               --cfg models/swinir.yaml \
                                                               --data p128x128_r512x512_s32x32 \
                                                               --data_hr p512x512_r2048x2048_s128x128 \
                                                               --batch 4 \
                                                               --lr 1e-5
                                                               --weight experiments/EXP_NAME/weight/ckpt_NUM/gene \
                                                               --name EXP_NAME
    ```


## Inference
LR 512 이미지를 HR 2048로 예측 (LR 512 -> HR 2048)
```
python infer.py --img-size 2048 --stride 512 --batch 1 --weight experiments/EXP_NAME/weight/ckpt_NUM/gene 
```

Model weight download [[link]](https://drive.google.com/drive/folders/10sDalvd7nDtGv_rqHvOa73__UlhEiXZK?usp=sharing)

## Ensemble
1. ensemble.ipynb 실행
2. 앙상블할 결과를 알맞게 subm_dirs 리스트에 작성
3. 앙상블 진행
