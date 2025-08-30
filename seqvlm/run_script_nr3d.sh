export PYTHONPATH=$(dirname "$PWD")
python evaluate_nr3d.py \
--exp_name visprog_test \
--image_path ../data/nr3d_preprocessed \
--vlm_model doubao-vision