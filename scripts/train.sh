#!/usr/bin/env bash
TIME=$(date "+%Y%m%d_%H%M%S")
MODEL=SK_Decoder

PYTHON=python
DATAROOT=/data


CHAIRS_ROOT=${DATAROOT}/FlyingChairs_release/data
SINTEL_ROOT=${DATAROOT}/Sintel
THINGS_ROOT=${DATAROOT}/FlyingThings3D
KITTI_ROOT=${DATAROOT}/kitti15
HD1K_ROOT=${DATAROOT}/hd1k


CHAIRS_SAVE_DIR=results/chairs/${MODEL}$-${TIME} 
THINGS_SAVE_DIR=results/things/${MODEL}$-${TIME} 
SINTEL_SAVE_DIR=results/sintel/${MODEL}-${TIME} 
KITTI_SAVE_DIR=results/kitti/${MODEL}$-${TIME} 

# chairs
CUDA_VISIBLE_DEVICES=0,1 ${PYTHON} train.py \
--name gma-chairs \
--stage chairs \
--validation chairs \
--output ${CHAIRS_SAVE_DIR} \
--num_steps 120000 \
--lr 0.00025 \
--image_size 368 496 \
--wdecay 0.0001 \
--gpus 0 1 \
--batch_size 8 \
--val_freq 10000 \
--print_freq 100 \
--mixed_precision \
--model_name ${MODEL} \
--chairs_root ${CHAIRS_ROOT} \
--things_root ${THINGS_ROOT} \
--sintel_root ${SINTEL_ROOT} \
--kitti_root ${KITTI_ROOT} \
--hd1k_root ${HD1K_ROOT} \
--UpdateBlock SKUpdateBlock6_Deep_nopoolres_AllDecoder \
--k_conv 1 15 \
--PCUpdater_conv 1 7 \


# # # # things
CUDA_VISIBLE_DEVICES=0,1 ${PYTHON} train.py \
--name gma-things \
--stage things \
--validation sintel \
--output ${THINGS_SAVE_DIR} \
--restore_ckpt ${CHAIRS_SAVE_DIR}/best.pth \
--num_steps 150000 \
--lr 0.000175 \
--image_size 400 720 \
--wdecay 0.0001 \
--gpus 0 1 \
--batch_size 6 \
--val_freq 5000 \
--print_freq 100 \
--mixed_precision \
--model_name ${MODEL} \
--chairs_root ${CHAIRS_ROOT} \
--things_root ${THINGS_ROOT} \
--sintel_root ${SINTEL_ROOT} \
--kitti_root ${KITTI_ROOT} \
--hd1k_root ${HD1K_ROOT} \
--UpdateBlock SKUpdateBlock6_Deep_nopoolres_AllDecoder \
--k_conv 1 15 \
--PCUpdater_conv 1 7 \


# sintel
CUDA_VISIBLE_DEVICES=0,1 ${PYTHON} train.py \
--name gma-sintel \
--stage sintel \
--validation sintel \
--output ${SINTEL_SAVE_DIR} \
--restore_ckpt ${THINGS_SAVE_DIR}/best.pth \
--num_steps 180000 \
--lr 0.000175 \
--image_size 368 768 \
--wdecay 0.00001 \
--gamma 0.85 \
--gpus 0 1 \
--batch_size 6 \
--val_freq 5000 \
--print_freq 100 \
--mixed_precision \
--model_name ${MODEL} \
--chairs_root ${CHAIRS_ROOT} \
--things_root ${THINGS_ROOT} \
--sintel_root ${SINTEL_ROOT} \
--kitti_root ${KITTI_ROOT} \
--hd1k_root ${HD1K_ROOT} \
--UpdateBlock SKUpdateBlock6_Deep_nopoolres_AllDecoder \
--k_conv 1 15 \
--PCUpdater_conv 1 7 \


# kitti
CUDA_VISIBLE_DEVICES=0,1 ${PYTHON} train.py \
--name gma-kitti \
--stage kitti \
--validation kitti \
--output ${KITTI_SAVE_DIR} \
--restore_ckpt ${SINTEL_SAVE_DIR}/best.pth \
--num_steps 50000 \
--lr 0.000175 \
--image_size 288 960 \
--wdecay 0.00001 \
--gamma 0.85 \
--gpus 0 1 \
--batch_size 6 \
--val_freq 5000 \
--print_freq 100 \
--mixed_precision \
--model_name ${MODEL} \
--chairs_root ${CHAIRS_ROOT} \
--things_root ${THINGS_ROOT} \
--sintel_root ${SINTEL_ROOT} \
--kitti_root ${KITTI_ROOT} \
--hd1k_root ${HD1K_ROOT} \
--UpdateBlock SKUpdateBlock6_Deep_nopoolres_AllDecoder \
--k_conv 1 15 \
--PCUpdater_conv 1 7 \

