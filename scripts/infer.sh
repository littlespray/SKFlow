#!/usr/bin/env bash
CHAIRS_SAVE_DIR=results/chairs/debug
THINGS_SAVE_DIR=results/things/debug
SINTEL_SAVE_DIR=results/sintel/debug
KITTI_SAVE_DIR=results/kitti/debug
PYTHON=python

DATAROOT=/data
CHAIRS_ROOT=${DATAROOT}/FlyingChairs_release/data
SINTEL_ROOT=${DATAROOT}/Sintel
KITTI_ROOT=${DATAROOT}/kitti15

MODEL=SK_Decoder
# best sintel
CKPT=skflow-sintel.pth


CUDA_VISIBLE_DEVICES=0 \
${PYTHON} evaluate.py \
--dataset sintel \
--iters 15 \
--model_name ${MODEL} \
--chairs_root ${CHAIRS_ROOT} \
--sintel_root ${SINTEL_ROOT} \
--kitti_root ${KITTI_ROOT} \
--UpdateBlock SKUpdateBlock6_Deep_nopoolres_AllDecoder \
--model ${CKPT} \
--k_conv 1 15 \
--PCUpdater_conv 1 7 \
