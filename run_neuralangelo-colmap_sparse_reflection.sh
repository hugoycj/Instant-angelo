INPUT_DIR=$1

starttime=$(date +%Y-%m-%d\ %H:%M:%S)
echo "---sfm---"
python scripts/run_colmap.py $INPUT_DIR
sfm_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---sparse_visualize---"
colmap model_converter \
    --input_path $INPUT_DIR/sparse/0 \
    --output_path $INPUT_DIR/sparse/0/points3D.ply --output_type PLY
sparse_visualize_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---angelo_recon---"
python launch.py --config configs/neuralangelo-colmap_sparse-wreflection.yaml --gpu 0 --train     dataset.root_dir=$INPUT_DIR
angelo_recon_time=`date +"%Y-%m-%d %H:%M:%S"`

echo 'start time:' $starttime
echo 'sfm time:' $sfm_time
echo 'sparse_visualize finished:' $model_converter_time
echo 'angelo_recon finished:' $angelo_recon_time
