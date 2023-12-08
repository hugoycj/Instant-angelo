export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1
docker build -t zebincai/nerf:v1 -f dev.dockerfile .
