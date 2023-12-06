export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1
docker build -t zebincai/nerf:latest -f dev.dockerfile .
