pip install -r requirements.txt

to run evaluation
./run_exec.sh --scan_path src/data/test_images_16_bit --output_path detections/results.txt
python evaluation.py detections/ src/data/labels_test_16_bit.txt --iou 0.5
