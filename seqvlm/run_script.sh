export PYTHONPATH=$(dirname "$PWD")
python evaluate.py \
--exp_name visprog_test \
--image_path ../data/scanrefer_preprocessed \
--vlm_model doubao-vision