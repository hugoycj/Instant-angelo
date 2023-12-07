INPUT_DIR=/mnt/nas/share-all/caizebin/03.dataset/nerf/instantangelo/dragon
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
starttime=$(date +%Y-%m-%d\ %H:%M:%S)
echo "---sfm---"
python scripts/run_colmap.py $INPUT_DIR
sfm_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---sparse_visualize---"
colmap model_converter \
    --input_path $INPUT_DIR/sparse/0 \
    --output_path $INPUT_DIR/sparse/0/points3D.ply --output_type PLY

exp_dir="/mnt/nas/share-all/caizebin/07.cache/nerf/nerfsudio/output"
echo "---angelo_recon---"
python trainer.py \
    --config configs/neuralangelo-colmap_sparse_trainer.yaml \
    --gpu 1 \
    --train \
    --exp_dir ${exp_dir} \
    dataset.root_dir=$INPUT_DIR
angelo_recon_time=`date +"%Y-%m-%d %H:%M:%S"`

echo 'start time:' $starttime
echo 'sfm time:' $sfm_time
echo 'sparse_visualize finished:' $model_converter_time
echo 'angelo_recon finished:' $angelo_recon_time
