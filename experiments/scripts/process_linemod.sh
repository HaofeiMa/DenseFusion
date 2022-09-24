mkdir ./datasets/linemod/LINEMOD/data
mkdir ./datasets/linemod/LINEMOD/models
mkdir ./datasets/linemod/LINEMOD/segnet_results

mv ./datasets/linemod/LINEMOD/timer ./datasets/linemod/LINEMOD/data/01
mv ./datasets/linemod/LINEMOD/data/01/JPEGImages ./datasets/linemod/LINEMOD/data/01/rgb

mv ./datasets/linemod/LINEMOD/data/01/labels ./datasets/linemod/LINEMOD/segnet_results/01_label
mv ./datasets/linemod/LINEMOD/data/01/registeredScene.ply ./datasets/linemod/LINEMOD/models/obj_01.ply