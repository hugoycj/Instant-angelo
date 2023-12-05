INPUT_DIR=$1

starttime=$(date +%Y-%m-%d\ %H:%M:%S)
echo "---sfm---"
python scripts/run_colmap.py $INPUT_DIR
sfm_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---model_converter---"
colmap model_converter \
    --input_path $INPUT_DIR/sparse/0 \
    --output_path $INPUT_DIR/sparse/0 \
    --output_type TXT

colmap model_converter \
    --input_path $INPUT_DIR/sparse/0 \
    --output_path $INPUT_DIR/sparse/0/points3D.ply --output_type PLY
model_converter_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---colmap2mvsnet---"
python third_party/Vis-MVSNet/colmap2mvsnet.py --dense_folder $INPUT_DIR --max_d 256 --convert_format
colmap2mvsnet_time=`date +"%Y-%m-%d %H:%M:%S"`

# echo "---mvsnet_inference---"
mkdir -p $INPUT_DIR/dense/mvsnet_fusion
python third_party/Vis-MVSNet/test.py --data_root $INPUT_DIR/dense \
        --dataset_name general --num_src 4 \
        --max_d 256 --load_path third_party/Vis-MVSNet/pretrained_model/vis \
        --write_result --result_dir $INPUT_DIR/dense/mvsnet_fusion
mvsnet_inference_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---mvsnet_fusion---"
python third_party/Vis-MVSNet/fusion.py --data $INPUT_DIR/dense/mvsnet_fusion \
        --pair $INPUT_DIR/dense/pair.txt
mvsnet_fusion_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---angelo_recon---"
python launch.py --config configs/neuralangelo-colmap_dense.yaml --gpu 0 --train     dataset.root_dir=$INPUT_DIR
angelo_recon_time=`date +"%Y-%m-%d %H:%M:%S"`

echo 'start time:' $starttime
echo 'sfm time:' $sfm_time
echo 'model_converter finished:' $model_converter_time
echo 'colmap2mvsnet finished:' $colmap2mvsnet_time
echo 'mvsnet_inference finished:' $mvsnet_inference_time
echo 'mvsnet_fusion finished:' $mvsnet_fusion_time
echo 'angelo_recon finished:' $angelo_recon_time
